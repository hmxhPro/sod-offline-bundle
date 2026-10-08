"""
app/api/llm_annotation.py
--------------------------
LLM自动标注API端点

提供：
1. 创建LLM标注任务
2. 查询任务状态和进度
3. 获取任务结果
4. 取消运行中的任务
"""

from __future__ import annotations

import asyncio
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select

from app.core.config import settings
from app.core.logging import logger
from app.db.models import (
    AnnotationIterationRecord,
    CategoryRecord,
    DatasetImageRecord,
    LLMAnnotationTaskRecord,
    OperationLog,
    TrainedModelRecord,
    TrainingJobRecord,
)
from app.db.session import AsyncSessionLocal
from app.services.llm_annotation import (
    PROVIDER,
    create_annotator,
    filter_multimodal_models,
    iter_annotate_images_batch,
    list_available_models,
    save_annotation_to_disk,
    validate_multimodal_model,
)

router = APIRouter()

# 全局任务执行器（存储运行中的任务）
_running_tasks: dict[str, asyncio.Task] = {}


# ── Request / Response Schemas ───────────────────────────────────────────────


class CreateAnnotationTaskRequest(BaseModel):
    """创建LLM标注任务的请求。

    提供商、模型、API密钥等全部由后端 .env 配置（固定为 Qwen 系列），
    前端不提供也不需要这些字段。
    """

    category_id: str = Field(..., description="类别ID")
    prompt: str = Field(
        ...,
        description="目标描述提示词，例如：'小型无人机' 或 '红色安全帽'",
        min_length=1,
        max_length=500,
    )
    model: Optional[str] = Field(
        default=None,
        max_length=120,
        description="本次任务使用的通义千问多模态/视觉模型；为空则使用后端默认 LLM_MODEL",
    )
    max_concurrent: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="最大并发API请求数（默认使用后端配置 LLM_MAX_CONCURRENT）",
    )
    only_pending: bool = Field(
        default=True,
        description="是否只标注状态为pending的图片（默认true）",
    )


class CreateAnnotationIterationRequest(BaseModel):
    """创建一轮可迭代 LLM 标注。"""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="本轮标注要求，例如：'标注画面中的船只'",
    )
    selection_mode: str = Field(
        default="pending_only",
        description="pending_only=只标未标注；no_target_only=重标未检测到目标；failed_only=重试检测失败；all=全部重标",
    )
    write_policy: str = Field(
        default="skip_existing",
        description="skip_existing=保留已有标注；overwrite=覆盖已有标注",
    )
    model: Optional[str] = Field(
        default=None,
        max_length=120,
        description="本轮使用的通义千问多模态/视觉模型；为空则使用后端默认 LLM_MODEL",
    )
    max_concurrent: Optional[int] = Field(default=None, ge=1, le=10)
    assist_model_id: Optional[str] = Field(
        default=None,
        max_length=36,
        description="可选：使用已训练模型辅助标注（先用YOLO预标，再让LLM修正/补充）",
    )


class AnnotationIterationItem(BaseModel):
    """一轮 LLM 标注的用户可见状态。"""

    id: str
    category_id: str
    iteration_no: int
    prompt: str
    status: str
    selection_mode: str
    write_policy: str
    llm_task_id: Optional[str] = None
    training_job_id: Optional[str] = None
    assist_model_id: Optional[str] = None
    total_images: int
    success_count: int
    no_target_count: int = 0
    failed_count: int
    total_boxes: int
    result_summary: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AnnotationConfigResponse(BaseModel):
    """后端 LLM 标注配置（供前端只读展示，绝不包含密钥）"""

    provider: str
    model: str
    configured: bool
    max_concurrent: int


class AnnotationModelsResponse(BaseModel):
    """可用 LLM 模型列表（只返回模型 ID，不返回密钥或原始响应）。"""

    provider: str
    current_model: str
    current_model_available: bool = True
    models: List[str]
    all_models_count: int = 0


class AnnotationTaskItem(BaseModel):
    """标注任务信息"""

    id: str
    category_id: str
    llm_provider: str
    llm_model: str
    prompt: str
    status: str
    progress: float
    total_images: int
    processed_images: int
    success_count: int
    no_target_count: int = 0
    failed_count: int
    total_boxes: int
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class CreateAnnotationIterationResponse(BaseModel):
    iteration: AnnotationIterationItem
    task: AnnotationTaskItem


class AnnotationTaskResult(BaseModel):
    """标注任务详细结果"""

    task: AnnotationTaskItem
    result_summary: Optional[dict] = None


class DeleteTaskResult(BaseModel):
    """删除标注任务的结果摘要——如实回传实际发生了什么，供前端据实告知用户，
    而非由前端提前断言（数据集是否清空、是否受保护由后端整库判定）。"""

    task_id: str
    protected: bool            # 数据集受保护（已训练/在训）→ 仅删任务记录，数据全留
    purged_images: int         # 清理掉的未成功标注图片数
    dataset_deleted: bool      # 数据集是否因清空而被一并删除
    dataset_name: Optional[str] = None


def _ann_path(category_id: str, image_id: str) -> Path:
    return settings.ANNOTATIONS_DIR / category_id / f"{image_id}.txt"


def _select_images_for_iteration(
    rows: List[DatasetImageRecord],
    write_policy: str,
) -> List[DatasetImageRecord]:
    """Apply write policy before queueing costly LLM calls."""
    if write_policy != "skip_existing":
        return list(rows)
    selected: list[DatasetImageRecord] = []
    for img in rows:
        if img.annotation_status == "annotated" and _ann_path(img.category_id, img.id).exists():
            continue
        selected.append(img)
    return selected


# ── API Endpoints ────────────────────────────────────────────────────────────


@router.get(
    "/annotation-config",
    response_model=AnnotationConfigResponse,
    summary="获取后端LLM标注配置（只读）",
)
async def get_annotation_config() -> AnnotationConfigResponse:
    """返回后端固定的标注配置，供前端展示当前使用的模型及是否已配置密钥。

    出于安全考虑，**绝不**返回 API 密钥本身，仅返回是否已配置（configured）。
    """
    return AnnotationConfigResponse(
        provider=PROVIDER,
        model=settings.LLM_MODEL,
        configured=settings.llm_configured,
        max_concurrent=settings.LLM_MAX_CONCURRENT,
    )


@router.get(
    "/annotation-models",
    response_model=AnnotationModelsResponse,
    summary="获取当前密钥可访问的LLM模型列表",
)
async def get_annotation_models() -> AnnotationModelsResponse:
    """实时从后端配置的 OpenAI 兼容端点获取模型列表。"""
    if not settings.llm_configured:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "后端未配置 LLM_API_KEY，请在 backend/.env 中设置后重启服务。",
        )
    try:
        models = await list_available_models()
        # 规则识别视觉模型（-vl-/qvq/omni 命名 + 白名单兜底），而非仅靠死白名单
        from app.services.llm_annotation import filter_multimodal_models
        multimodal_models = filter_multimodal_models(models)
    except Exception as exc:
        logger.warning(f"获取 LLM 模型列表失败: {exc}")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"获取模型列表失败：{exc}",
        ) from exc
    return AnnotationModelsResponse(
        provider=PROVIDER,
        current_model=settings.LLM_MODEL,
        current_model_available=(settings.LLM_MODEL in models),
        models=multimodal_models,
        all_models_count=len(models),
    )


class SetDefaultModelRequest(BaseModel):
    """设置默认标注模型请求。probe=True 时先做一次调用探测，不可用则拒绝设置。"""

    model: str
    probe: bool = True


def _persist_env_default_model(model: str) -> None:
    """把默认模型写回 backend/.env（UTF-8 无 BOM），重启后仍生效。"""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    line = f"LLM_MODEL={model}"
    try:
        if env_path.exists():
            text = env_path.read_text(encoding="utf-8")
            new_text, n = re.subn(r"(?m)^LLM_MODEL=.*$", line, text)
            if n == 0:
                new_text = text.rstrip("\n") + f"\n{line}\n"
            env_path.write_text(new_text, encoding="utf-8")
        else:
            env_path.write_text(line + "\n", encoding="utf-8")
    except Exception as exc:
        logger.warning(f"写入 .env 失败，默认模型仅对当前进程生效：{exc}")


@router.put(
    "/annotation-config/model",
    response_model=AnnotationConfigResponse,
    summary="设置默认标注模型（探测可用后立即生效并持久化到 backend/.env）",
)
async def set_default_annotation_model(req: SetDefaultModelRequest) -> AnnotationConfigResponse:
    from app.services.llm_annotation import is_multimodal_model, probe_model

    model = (req.model or "").strip()
    if not model:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "模型名不能为空")
    if not is_multimodal_model(model):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{model} 不是视觉(VL)模型，无法用于图片标注。",
        )
    if req.probe and settings.llm_configured:
        ok, detail = await probe_model(model)
        if not ok:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"模型 {model} 探测失败，未修改默认：{detail}",
            )
    settings.LLM_MODEL = model
    _persist_env_default_model(model)
    logger.info(f"默认标注模型已设置为 {model}")
    return AnnotationConfigResponse(
        provider=PROVIDER,
        model=settings.LLM_MODEL,
        configured=settings.llm_configured,
        max_concurrent=settings.LLM_MAX_CONCURRENT,
    )


@router.post(
    "/categories/{category_id}/annotation-iterations",
    response_model=CreateAnnotationIterationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建一轮可迭代LLM标注",
)
async def create_annotation_iteration(
    category_id: str,
    body: CreateAnnotationIterationRequest,
) -> CreateAnnotationIterationResponse:
    """Create the next LLM annotation round for an existing dataset.

    This is the iterative layer on top of the existing annotation task runner: it
    records a user-facing round number, selects images according to the requested
    mode, then launches the same background LLM task used by the old endpoints.
    """
    if not settings.llm_configured:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "后端未配置 LLM_API_KEY，请在 backend/.env 中设置后重启服务。",
        )
    selection_mode = (body.selection_mode or "pending_only").strip()
    write_policy = (body.write_policy or "skip_existing").strip()
    if selection_mode not in {"pending_only", "all", "no_target_only", "failed_only"}:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "selection_mode 只能是 pending_only / no_target_only / failed_only / all。",
        )
    if write_policy not in {"skip_existing", "overwrite"}:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "write_policy 只能是 skip_existing / overwrite。",
        )

    task_id = uuid.uuid4().hex
    iteration_id = uuid.uuid4().hex
    max_concurrent = body.max_concurrent or settings.LLM_MAX_CONCURRENT
    try:
        selected_model = validate_multimodal_model(body.model)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    try:
        async with AsyncSessionLocal() as session:
            cat = await session.get(CategoryRecord, category_id)
            if cat is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "类别不存在")

            # 验证辅助模型（如果指定）
            assist_model_id = body.assist_model_id
            if assist_model_id:
                assist_model = await session.get(TrainedModelRecord, assist_model_id)
                if assist_model is None:
                    raise HTTPException(
                        status.HTTP_404_NOT_FOUND,
                        f"辅助模型 {assist_model_id} 不存在",
                    )
                logger.info(
                    f"本轮标注将使用辅助模型: {assist_model.name} v{assist_model.version}"
                )

            stmt = select(DatasetImageRecord).where(DatasetImageRecord.category_id == category_id)
            if selection_mode == "pending_only":
                stmt = stmt.where(DatasetImageRecord.annotation_status == "pending")
            elif selection_mode == "no_target_only":
                stmt = stmt.where(DatasetImageRecord.annotation_status == "no_target")
            elif selection_mode == "failed_only":
                stmt = stmt.where(DatasetImageRecord.annotation_status == "failed")
            rows = (await session.execute(stmt)).scalars().all()
            images = _select_images_for_iteration(list(rows), write_policy)
            if not images:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "本轮没有可标注的图片，请先上传新素材，或选择重标全部并允许覆盖已有标注。",
                )

            max_no = await session.scalar(
                select(func.max(AnnotationIterationRecord.iteration_no)).where(
                    AnnotationIterationRecord.category_id == category_id
                )
            )
            iteration = AnnotationIterationRecord(
                id=iteration_id,
                category_id=category_id,
                iteration_no=int(max_no or 0) + 1,
                prompt=body.prompt.strip(),
                status="pending",
                selection_mode=selection_mode,
                write_policy=write_policy,
                llm_task_id=task_id,
                assist_model_id=assist_model_id,
                total_images=len(images),
            )
            task = LLMAnnotationTaskRecord(
                id=task_id,
                category_id=category_id,
                llm_provider=PROVIDER,
                llm_model=selected_model,
                prompt=body.prompt.strip(),
                status="pending",
                total_images=len(images),
                result_summary={
                    "iteration_id": iteration_id,
                    "iteration_no": iteration.iteration_no,
                    "selection_mode": selection_mode,
                    "write_policy": write_policy,
                },
            )
            session.add(iteration)
            session.add(task)
            await session.commit()
            await session.refresh(iteration)
            await session.refresh(task)

        asyncio_task = asyncio.create_task(
            _run_annotation_task(
                task_id=task_id,
                category_id=category_id,
                prompt=body.prompt.strip(),
                images=images,
                max_concurrent=max_concurrent,
                model_name=selected_model,
                iteration_id=iteration_id,
                write_policy=write_policy,
            )
        )
        _running_tasks[task_id] = asyncio_task
        return CreateAnnotationIterationResponse(
            iteration=AnnotationIterationItem.model_validate(iteration),
            task=AnnotationTaskItem.model_validate(task),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"创建迭代标注失败: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用") from exc


@router.get(
    "/categories/{category_id}/annotation-iterations",
    response_model=List[AnnotationIterationItem],
    summary="列出某个数据集的LLM标注轮次",
)
async def list_annotation_iterations(category_id: str) -> List[AnnotationIterationItem]:
    try:
        async with AsyncSessionLocal() as session:
            cat = await session.get(CategoryRecord, category_id)
            if cat is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "类别不存在")
            rows = (
                await session.execute(
                    select(AnnotationIterationRecord)
                    .where(AnnotationIterationRecord.category_id == category_id)
                    .order_by(AnnotationIterationRecord.iteration_no.desc())
                )
            ).scalars().all()
            return [AnnotationIterationItem.model_validate(r) for r in rows]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"列出标注轮次失败: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用") from exc


@router.post(
    "/annotation-tasks",
    response_model=AnnotationTaskItem,
    status_code=status.HTTP_201_CREATED,
    summary="创建LLM自动标注任务",
)
async def create_annotation_task(
    body: CreateAnnotationTaskRequest,
) -> AnnotationTaskItem:
    """
    创建一个LLM自动标注任务，异步批量标注指定类别的图片。

    提供商/模型/密钥均取自后端 .env 配置（固定为 Qwen），前端无需传入。

    工作流程：
    1. 校验后端已配置 LLM_API_KEY
    2. 验证类别存在
    3. 查询待标注的图片列表
    4. 创建任务记录
    5. 异步执行标注（后台任务）
    6. 立即返回任务信息

    前端可通过轮询 GET /annotation-tasks/{task_id} 查看进度
    """
    # 未配置密钥时直接拒绝，避免创建必然失败的任务。
    if not settings.llm_configured:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "后端未配置 LLM_API_KEY，请在 backend/.env 中设置后重启服务。",
        )

    task_id = uuid.uuid4().hex
    try:
        llm_model = validate_multimodal_model(body.model)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    max_concurrent = body.max_concurrent or settings.LLM_MAX_CONCURRENT

    try:
        async with AsyncSessionLocal() as session:
            # 验证类别存在
            cat = await session.get(CategoryRecord, body.category_id)
            if cat is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "类别不存在")

            # 查询待标注的图片
            if body.only_pending:
                stmt = select(DatasetImageRecord).where(
                    DatasetImageRecord.category_id == body.category_id,
                    DatasetImageRecord.annotation_status == "pending",
                )
            else:
                stmt = select(DatasetImageRecord).where(
                    DatasetImageRecord.category_id == body.category_id
                )

            rows = (await session.execute(stmt)).scalars().all()

            if not rows:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "该类别下没有待标注的图片",
                )

            # 创建任务记录
            task = LLMAnnotationTaskRecord(
                id=task_id,
                category_id=body.category_id,
                llm_provider=PROVIDER,
                llm_model=llm_model,
                prompt=body.prompt,
                status="pending",
                total_images=len(rows),
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)

            # 启动后台标注任务
            asyncio_task = asyncio.create_task(
                _run_annotation_task(
                    task_id=task_id,
                    category_id=body.category_id,
                    prompt=body.prompt,
                    images=rows,
                    max_concurrent=max_concurrent,
                    model_name=llm_model,
                )
            )
            _running_tasks[task_id] = asyncio_task

            return AnnotationTaskItem.model_validate(task)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"创建标注任务失败: {exc}")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用"
        ) from exc


@router.get(
    "/annotation-tasks/{task_id}",
    response_model=AnnotationTaskResult,
    summary="获取标注任务详情",
)
async def get_annotation_task(task_id: str) -> AnnotationTaskResult:
    """查询标注任务的状态、进度和结果"""
    try:
        async with AsyncSessionLocal() as session:
            task = await session.get(LLMAnnotationTaskRecord, task_id)
            if task is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在")

            return AnnotationTaskResult(
                task=AnnotationTaskItem.model_validate(task),
                result_summary=task.result_summary,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"查询标注任务失败: {exc}")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用"
        ) from exc


@router.get(
    "/annotation-tasks",
    response_model=List[AnnotationTaskItem],
    summary="列出所有标注任务",
)
async def list_annotation_tasks(
    category_id: Optional[str] = None,
) -> List[AnnotationTaskItem]:
    """列出标注任务（可按类别筛选）"""
    try:
        async with AsyncSessionLocal() as session:
            stmt = select(LLMAnnotationTaskRecord)
            if category_id:
                stmt = stmt.where(
                    LLMAnnotationTaskRecord.category_id == category_id
                )
            stmt = stmt.order_by(LLMAnnotationTaskRecord.created_at.desc())

            rows = (await session.execute(stmt)).scalars().all()
            return [AnnotationTaskItem.model_validate(r) for r in rows]
    except Exception as exc:
        logger.error(f"列出标注任务失败: {exc}")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用"
        ) from exc


@router.post(
    "/annotation-tasks/{task_id}/cancel",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="取消运行中的标注任务",
)
async def cancel_annotation_task(task_id: str):
    """取消正在运行的标注任务"""
    # 取消asyncio任务
    if task_id in _running_tasks:
        _running_tasks[task_id].cancel()
        del _running_tasks[task_id]

    # 更新数据库状态
    try:
        async with AsyncSessionLocal() as session:
            task = await session.get(LLMAnnotationTaskRecord, task_id)
            if task is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在")

            if task.status not in ("pending", "running"):
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST, "任务已完成，无法取消"
                )

            task.status = "cancelled"
            task.finished_at = datetime.utcnow()
            iteration = await session.scalar(
                select(AnnotationIterationRecord).where(
                    AnnotationIterationRecord.llm_task_id == task_id
                )
            )
            if iteration:
                iteration.status = "cancelled"
                iteration.finished_at = task.finished_at
            await session.commit()

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"取消标注任务失败: {exc}")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用"
        ) from exc

    return None


@router.delete(
    "/annotation-tasks/{task_id}",
    response_model=DeleteTaskResult,
    summary="删除标注任务，并清理未成功标注的数据",
)
async def delete_annotation_task(task_id: str) -> DeleteTaskResult:
    """删除一个**已结束**（失败 / 已取消 / 已完成）的标注任务。

    删除任务记录的同时，清理该数据集中**未成功标注**的图片（原图 + 标注文件）；
    已成功标注的图片予以保留，仍可用于训练。若清理后数据集已无任何图片，则连同这个
    空数据集一并删除，避免遗留垃圾。

    约束：
    - 进行中的任务（等待中 / 标注中）需先取消，才能删除（409）。
    - 同一数据集还有别的进行中任务时，需先取消那些任务再删（409），避免误删其数据。
    - 已生成模型、已训练（含待人工确认 needs_review）、或正在训练的数据集受保护：
      仅删除任务记录，绝不动其任何数据。
    - 与单图删除 / 删除类别一致：先提交数据库，再清理磁盘文件。

    返回实际结果摘要（受保护 / 清理张数 / 是否删除了数据集），供前端如实告知用户。
    """
    from app.api.dataset import _ann_path, _raw_dir, _recompute_counts
    from app.services.training_manager import training_manager

    _running_tasks.pop(task_id, None)  # 终态任务通常已不在此，防御性清理

    category_id: Optional[str] = None
    dataset_name: Optional[str] = None
    purged_image_ids: List[str] = []
    dataset_deleted = False
    protected = False

    try:
        async with AsyncSessionLocal() as session:
            # 行锁：并发的重复删除在此串行化，第二个请求读到 None → 干净的 404，
            # 而不是双双闯入销毁逻辑、报出误导性的 503。
            task = await session.get(
                LLMAnnotationTaskRecord, task_id, with_for_update=True
            )
            if task is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在")
            if task.status in ("pending", "running"):
                raise HTTPException(
                    status.HTTP_409_CONFLICT, "任务正在进行，请先取消后再删除。"
                )
            category_id = task.category_id

            cat_row = await session.get(CategoryRecord, category_id)
            dataset_name = cat_row.name if cat_row is not None else None

            # 同一数据集若还有别的进行中标注任务，拒绝删除——否则会把它们尚未标注的
            # 图片一并清掉、甚至连数据集一起删除。要求先取消那些任务。
            other_active = await session.scalar(
                select(func.count())
                .select_from(LLMAnnotationTaskRecord)
                .where(
                    LLMAnnotationTaskRecord.category_id == category_id,
                    LLMAnnotationTaskRecord.id != task_id,
                    LLMAnnotationTaskRecord.status.in_(("pending", "running")),
                )
            ) or 0
            if other_active:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "该数据集还有正在进行的标注任务，请先取消后再删除。",
                )

            # 数据集是否受保护：已生成模型 / 有训练历史 / 正在训练 → 仅删任务记录。
            model_cnt = await session.scalar(
                select(func.count())
                .select_from(TrainedModelRecord)
                .where(TrainedModelRecord.category_id == category_id)
            ) or 0
            # 训练历史：低于部署门槛的训练会留下 status='needs_review' 且不写
            # TrainedModelRecord，必须一并纳入保护，否则其冻结数据集/权重会被误删。
            job_cnt = await session.scalar(
                select(func.count())
                .select_from(TrainingJobRecord)
                .where(
                    TrainingJobRecord.category_id == category_id,
                    TrainingJobRecord.status.in_(
                        ("pending", "running", "finished", "needs_review")
                    ),
                )
            ) or 0
            training_active = False
            active_id = training_manager.active_job_id
            if active_id is not None:
                job = await session.get(TrainingJobRecord, active_id)
                # 行尚未落库 == 训练正在启动（start() 先发布 active_job_id 再插入行）：
                # 此刻无法确认归属，保守按受保护处理（窗口仅毫秒级）。
                training_active = job is None or (
                    job.category_id == category_id
                    and job.status in ("pending", "running")
                )
            protected = bool(model_cnt) or bool(job_cnt) or training_active

            if protected:
                await session.delete(task)
                await session.commit()
            else:
                # 删除“未成功标注”的图片行（annotation_status != 'annotated'）。
                unsuccessful = (
                    await session.execute(
                        select(DatasetImageRecord).where(
                            DatasetImageRecord.category_id == category_id,
                            DatasetImageRecord.annotation_status != "annotated",
                        )
                    )
                ).scalars().all()
                purged_image_ids = [img.id for img in unsuccessful]
                for img in unsuccessful:
                    await session.delete(img)
                await session.flush()
                await _recompute_counts(session, category_id)

                remaining = await session.scalar(
                    select(func.count())
                    .select_from(DatasetImageRecord)
                    .where(DatasetImageRecord.category_id == category_id)
                ) or 0

                if remaining == 0:
                    # 数据集已空 → 整体删除（含本任务及该类别其余历史记录）。
                    no_sync = {"synchronize_session": False}
                    await session.execute(
                        delete(LLMAnnotationTaskRecord)
                        .where(LLMAnnotationTaskRecord.category_id == category_id)
                        .execution_options(**no_sync)
                    )
                    await session.execute(
                        delete(AnnotationIterationRecord)
                        .where(AnnotationIterationRecord.category_id == category_id)
                        .execution_options(**no_sync)
                    )
                    await session.execute(
                        delete(TrainingJobRecord)
                        .where(TrainingJobRecord.category_id == category_id)
                        .execution_options(**no_sync)
                    )
                    cat = await session.get(CategoryRecord, category_id)
                    if cat is not None:
                        await session.delete(cat)
                    dataset_deleted = True
                else:
                    # 仍有已标注图片 → 数据集保留，仅删除本任务记录。
                    await session.delete(task)

                await session.commit()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"删除标注任务失败 {task_id}: {exc}")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用"
        ) from exc

    # ── 数据库提交成功后再清理磁盘（顺序与 delete_image / delete_category 一致）──
    if category_id:
        if dataset_deleted:
            for d in (
                settings.DATASETS_DIR / category_id,
                settings.ANNOTATIONS_DIR / category_id,
            ):
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
        else:
            raw_dir = _raw_dir(category_id)
            for image_id in purged_image_ids:
                for p in raw_dir.glob(f"{image_id}.*"):
                    p.unlink(missing_ok=True)
                _ann_path(category_id, image_id).unlink(missing_ok=True)

    # 审计（best-effort，失败不影响主流程）
    try:
        async with AsyncSessionLocal() as session:
            session.add(OperationLog(
                action="delete_annotation_task",
                detail=(
                    f"task={task_id} category={category_id} protected={protected} "
                    f"purged_images={len(purged_image_ids)} dataset_deleted={dataset_deleted}"
                ),
                affected_count=len(purged_image_ids),
            ))
            await session.commit()
    except Exception as exc:
        logger.warning(f"记录删除任务审计失败 {task_id}: {exc}")

    return DeleteTaskResult(
        task_id=task_id,
        protected=protected,
        purged_images=len(purged_image_ids),
        dataset_deleted=dataset_deleted,
        dataset_name=dataset_name,
    )


# ── Background Task Executor ─────────────────────────────────────────────────


async def _run_annotation_task(
    task_id: str,
    category_id: str,
    prompt: str,
    images: List[DatasetImageRecord],
    max_concurrent: int,
    *,
    model_name: Optional[str] = None,
    iteration_id: Optional[str] = None,
    write_policy: str = "overwrite",
):
    """
    后台执行标注任务

    步骤：
    1. 更新任务状态为running
    2. 批量调用LLM API标注（提供商/模型/密钥取自后端配置）
    3. 保存标注结果到磁盘
    4. 更新数据库图片状态
    5. 更新任务完成状态
    """
    selected_model = validate_multimodal_model(model_name)
    logger.info(
        f"开始LLM标注任务 {task_id}: {len(images)}张图片, "
        f"提示词='{prompt}', 模型={PROVIDER}/{selected_model}"
    )

    try:
        # 更新任务状态为running
        async with AsyncSessionLocal() as session:
            task = await session.get(LLMAnnotationTaskRecord, task_id)
            if task is None:
                return
            task.status = "running"
            task.started_at = datetime.utcnow()
            if iteration_id:
                iteration = await session.get(AnnotationIterationRecord, iteration_id)
                if iteration:
                    iteration.status = "running"
                    iteration.started_at = task.started_at
            await session.commit()

        # 先探测解析出真正可调用的视觉模型：指定/默认模型无调用权限时自动降级
        # 为账号里可用的 VL 模型；全部不可用则立刻失败并给出可读原因，
        # 避免整批图片逐张 403 后被误报成「未识别到目标」。
        from app.services.llm_annotation import resolve_usable_model

        resolved_model = await resolve_usable_model(selected_model)
        if resolved_model != selected_model:
            logger.warning(
                f"任务 {task_id}: 模型 {selected_model} 不可用，改用 {resolved_model}"
            )
            selected_model = resolved_model
            async with AsyncSessionLocal() as session:
                task = await session.get(LLMAnnotationTaskRecord, task_id)
                if task:
                    task.llm_model = selected_model
                    await session.commit()

        # 创建LLM标注器（API Key/Base 来自后端配置，模型使用本任务选择）
        annotator = create_annotator(selected_model)

        # 批量标注
        image_dicts = [
            {
                "id": img.id,
                "stored_path": img.stored_path,
                "width": img.width,
                "height": img.height,
            }
            for img in images
        ]

        # 保存标注结果（每张图片完成后立即写库，前端进度不会长时间卡在 0%）
        success_count = 0
        no_target_count = 0
        failed_count = 0
        total_boxes = 0
        last_error: Optional[str] = None

        async with AsyncSessionLocal() as session:
            async for result in iter_annotate_images_batch(
                annotator=annotator,
                images=image_dicts,
                prompt=prompt,
                category_id=category_id,
                max_concurrent=max_concurrent,
            ):
                if result.success:
                    img = await session.get(DatasetImageRecord, result.image_id)
                    if not result.boxes:
                        # For LLM auto-annotation, an empty result means "target not
                        # found", not a positive annotated sample. Do not write an
                        # empty YOLO label here; otherwise the training page shows
                        # many selected images with no boxes and the category looks
                        # trainable even though no target was found.
                        if img:
                            img.annotation_status = "no_target"
                            img.box_count = 0
                        _ann_path(category_id, result.image_id).unlink(missing_ok=True)
                        no_target_count += 1
                    elif (
                        write_policy == "skip_existing"
                        and img
                        and img.annotation_status == "annotated"
                        and _ann_path(category_id, result.image_id).exists()
                    ):
                        # The row may have become annotated after this task was
                        # queued; count it as processed but do not overwrite it.
                        success_count += 1
                        total_boxes += img.box_count or 0
                    else:
                        # 保存到磁盘
                        save_annotation_to_disk(
                            category_id=category_id,
                            image_id=result.image_id,
                            boxes=result.boxes,
                        )

                        # 更新数据库
                        if img:
                            img.annotation_status = "annotated"
                            img.box_count = len(result.boxes)

                        success_count += 1
                        total_boxes += len(result.boxes)
                else:
                    img = await session.get(DatasetImageRecord, result.image_id)
                    if img:
                        img.annotation_status = "failed"
                        img.box_count = 0
                    failed_count += 1
                    last_error = result.error or last_error

                # 更新进度
                task = await session.get(LLMAnnotationTaskRecord, task_id)
                if task:
                    task.processed_images = success_count + no_target_count + failed_count
                    task.progress = task.processed_images / task.total_images if task.total_images else 1.0
                    task.success_count = success_count
                    task.failed_count = failed_count
                    task.total_boxes = total_boxes
                    task.result_summary = {
                        "success": success_count,
                        "no_target": no_target_count,
                        "failed": failed_count,
                        "total_boxes": total_boxes,
                        "phase": "annotating",
                    }
                if iteration_id:
                    iteration = await session.get(AnnotationIterationRecord, iteration_id)
                    if iteration:
                        iteration.status = "running"
                        iteration.success_count = success_count
                        iteration.failed_count = failed_count
                        iteration.total_boxes = total_boxes
                        iteration.result_summary = {
                            "processed_images": success_count + no_target_count + failed_count,
                            "success": success_count,
                            "no_target": no_target_count,
                            "failed": failed_count,
                            "total_boxes": total_boxes,
                            "write_policy": write_policy,
                            "task_id": task_id,
                        }

                await session.commit()

            # 重新计算类别统计
            from app.api.dataset import _recompute_counts

            await _recompute_counts(session, category_id)
            await session.commit()

        # 标记任务完成
        async with AsyncSessionLocal() as session:
            task = await session.get(LLMAnnotationTaskRecord, task_id)
            if task:
                task.status = "finished"
                task.finished_at = datetime.utcnow()
                task.result_summary = {
                    "success": success_count,
                    "no_target": no_target_count,
                    "failed": failed_count,
                    "total_boxes": total_boxes,
                    **({"iteration_id": iteration_id, "write_policy": write_policy} if iteration_id else {}),
                }
                # 全部失败时把真实原因写进任务，避免前端只显示「未识别到目标」
                if success_count == 0 and failed_count > 0:
                    task.error = (
                        f"全部 {failed_count} 张图片标注失败；"
                        f"最后错误：{(last_error or '未知')[:300]}"
                    )
            if iteration_id:
                iteration = await session.get(AnnotationIterationRecord, iteration_id)
                if iteration:
                    iteration.status = "ready_for_training" if success_count > 0 else "failed"
                    iteration.finished_at = datetime.utcnow()
                    iteration.success_count = success_count
                    iteration.failed_count = failed_count
                    iteration.total_boxes = total_boxes
                    iteration.error = None if success_count > 0 else "本轮未成功标注任何图片。"
                    iteration.result_summary = {
                        "success": success_count,
                        "no_target": no_target_count,
                        "failed": failed_count,
                        "total_boxes": total_boxes,
                        "task_id": task_id,
                    }
            await session.commit()

        logger.info(
            f"LLM标注任务完成 {task_id}: 检测成功{success_count}, "
            f"未检测到目标{no_target_count}, 检测失败{failed_count}, 总框数{total_boxes}"
        )

    except asyncio.CancelledError:
        logger.info(f"LLM标注任务被取消 {task_id}")
        async with AsyncSessionLocal() as session:
            task = await session.get(LLMAnnotationTaskRecord, task_id)
            if task:
                task.status = "cancelled"
                task.finished_at = datetime.utcnow()
            if iteration_id:
                iteration = await session.get(AnnotationIterationRecord, iteration_id)
                if iteration:
                    iteration.status = "cancelled"
                    iteration.finished_at = datetime.utcnow()
            await session.commit()
        raise

    except Exception as exc:
        logger.error(f"LLM标注任务失败 {task_id}: {exc}")
        async with AsyncSessionLocal() as session:
            task = await session.get(LLMAnnotationTaskRecord, task_id)
            if task:
                task.status = "failed"
                task.error = str(exc)
                task.finished_at = datetime.utcnow()
            if iteration_id:
                iteration = await session.get(AnnotationIterationRecord, iteration_id)
                if iteration:
                    iteration.status = "failed"
                    iteration.error = str(exc)
                    iteration.finished_at = datetime.utcnow()
            await session.commit()

    finally:
        # 清理任务引用
        _running_tasks.pop(task_id, None)
