"""
app/api/auto_annotate.py
-------------------------
一站式自动标注端点。

用户只需：①说明要标注什么（自然语言或直接名词）②上传图片或视频。系统据此：
1. 自动从输入中提取目标名词作为数据集类别名，并加上时间戳保证唯一；
2. 把图片落盘（视频则后台抽帧成图片）；
3. 复用现有 LLM 标注流程逐张标注，完成后该数据集即可直接训练。

设计要点：
- ``UploadFile`` 不能跨请求存活，所有上传内容必须在请求内落盘；耗时的视频解码放到
  后台任务里用 ``asyncio.to_thread`` 执行，避免阻塞事件循环。
- 复用 ``app/api/dataset.py::_store_one_image / _recompute_counts / _raw_dir`` 与
  ``app/api/llm_annotation.py::_run_annotation_task``，与手动上传/标注产出完全一致的
  磁盘 + 数据库表示，下游训练逻辑零改动。
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.api.dataset import _guard_upload_batch, _raw_dir, _recompute_counts, _store_one_image
from app.api.llm_annotation import _run_annotation_task, _running_tasks
from app.core.config import settings
from app.core.logging import logger
from app.db.models import (
    AnnotationIterationRecord,
    CategoryRecord,
    DatasetImageRecord,
    LLMAnnotationTaskRecord,
)
from app.db.session import AsyncSessionLocal
from app.services.frame_extractor import extract_frames
from app.services.llm_annotation import PROVIDER, extract_target_noun, validate_multimodal_model
from app.utils.video_utils import get_video_info

router = APIRouter()

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
_VALID_DENSITIES = {"sparse", "medium", "dense"}


class AutoAnnotateResponse(BaseModel):
    """一站式自动标注创建结果。"""

    task_id: str
    iteration_id: Optional[str] = None
    category_id: str
    category_name: str   # 带时间戳的唯一名（如 船只_20260620153012）
    noun: str            # 提取出的目标名词（如 船只）
    mode: str            # images | video
    total_images: int    # 图片模式为已入库张数；视频模式抽帧前为 0


# ── helpers ──────────────────────────────────────────────────────────────────


async def _make_unique_category_name(noun: str) -> str:
    """生成带时间戳的唯一类别名：``{noun}_{YYYYMMDDHHMMSS}``，撞名追加序号。"""
    base = f"{noun}_{datetime.now():%Y%m%d%H%M%S}"
    candidate = base
    n = 0
    async with AsyncSessionLocal() as session:
        while True:
            dup = await session.scalar(
                select(CategoryRecord).where(CategoryRecord.name == candidate)
            )
            if dup is None:
                return candidate
            n += 1
            candidate = f"{base}_{n}"


async def _save_upload_video(f: UploadFile) -> Path:
    """把上传视频流式落盘到 UPLOAD_DIR，带大小上限。"""
    suffix = Path(f.filename or "").suffix.lower()
    if suffix not in _VIDEO_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"不支持的视频格式 '{suffix}'。支持：{', '.join(sorted(_VIDEO_EXTENSIONS))}",
        )
    dest = settings.UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    total = 0
    try:
        async with aiofiles.open(dest, "wb") as out:
            while chunk := await f.read(4 * 1024 * 1024):
                total += len(chunk)
                if total > settings.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        f"视频文件超过上限 {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB。",
                    )
                await out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.error(f"视频保存失败 {f.filename}: {exc}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "视频保存失败。") from exc
    return dest


async def _cleanup_category(category_id: str) -> None:
    """创建后失败时的尽力清理：删类别行 + 图片行 + 数据集/标注目录。"""
    try:
        async with AsyncSessionLocal() as session:
            from sqlalchemy import delete

            await session.execute(
                delete(DatasetImageRecord).where(DatasetImageRecord.category_id == category_id)
            )
            rec = await session.get(CategoryRecord, category_id)
            if rec is not None:
                await session.delete(rec)
            await session.commit()
    except Exception as exc:
        logger.warning(f"清理类别 {category_id} 失败: {exc}")
    for d in (settings.DATASETS_DIR / category_id, settings.ANNOTATIONS_DIR / category_id):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


# ── endpoint ─────────────────────────────────────────────────────────────────


@router.post(
    "/auto-annotate",
    response_model=AutoAnnotateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="一站式自动标注：说目标 + 传图片/视频 → 自动建带时间戳数据集并标注",
)
async def auto_annotate(
    target: str = Form(..., description="要标注/检测的目标（自然语言或直接名词）"),
    files: List[UploadFile] = File(..., description="图片（可多张）或单个视频"),
    video_density: str = Form("medium", description="视频抽帧密度：sparse | medium | dense"),
    model: Optional[str] = Form(None, description="本次标注使用的通义千问多模态/视觉模型"),
) -> AutoAnnotateResponse:
    if not settings.llm_configured:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "自动标注服务暂未开通，请联系管理员开通后再使用。",
        )
    if not (target or "").strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "请先填写要标注的目标。")
    if not files:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "请上传图片或视频。")
    try:
        selected_model = validate_multimodal_model(model)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    _guard_upload_batch(files, verb="上传")
    density = video_density if video_density in _VALID_DENSITIES else "medium"

    # 分流：有视频文件 → 视频模式（取第一个视频）；否则按图片处理。
    video_file = next(
        (f for f in files if Path(f.filename or "").suffix.lower() in _VIDEO_EXTENSIONS),
        None,
    )
    mode = "video" if video_file is not None else "images"

    # 1) 提取目标名词
    noun = (await extract_target_noun(target)).strip()
    if not noun:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "无法从描述中识别出要标注的目标，请换一个更具体的说法（如：船只、安全帽）。",
        )

    # 2) 建带时间戳的唯一类别（并发同秒同名时自动追加序号重试，避免直接报冲突）
    category_id = str(uuid.uuid4())
    base_name = await _make_unique_category_name(noun)
    category_name = None
    try:
        for attempt in range(5):
            candidate = base_name if attempt == 0 else f"{base_name}_{attempt}"
            try:
                async with AsyncSessionLocal() as session:
                    session.add(CategoryRecord(
                        id=category_id, name=candidate,
                        description=f"AI 自动标注：{target.strip()}", status="draft",
                    ))
                    await session.commit()
                category_name = candidate
                break
            except IntegrityError:
                continue  # 撞名，换序号重试
        if category_name is None:
            raise HTTPException(status.HTTP_409_CONFLICT, "数据集创建冲突，请稍后重试。")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"创建自动标注类别失败: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc

    # 3) 在请求内把上传内容落盘
    video_path: Optional[Path] = None
    total_images = 0
    try:
        if mode == "video":
            video_path = await _save_upload_video(video_file)
            try:
                get_video_info(video_path)  # 提前校验是坏视频则立即报错
            except Exception as exc:
                video_path.unlink(missing_ok=True)
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY, f"视频无法读取：{exc}"
                ) from exc
        else:
            async with AsyncSessionLocal() as session:
                stored = []
                for f in files:
                    rec = await _store_one_image(session, category_id, f)
                    if rec is not None:
                        stored.append(rec)
                if not stored:
                    raise HTTPException(
                        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        "没有有效图片，请上传 jpg/png 等常见图片或视频。",
                    )
                await session.flush()
                await _recompute_counts(session, category_id)
                await session.commit()
                total_images = len(stored)

        # 4) 建第 1 轮标注记录 + 标注任务记录
        task_id = uuid.uuid4().hex
        iteration_id = uuid.uuid4().hex
        async with AsyncSessionLocal() as session:
            session.add(AnnotationIterationRecord(
                id=iteration_id,
                category_id=category_id,
                iteration_no=1,
                prompt=target.strip(),
                status="pending",
                selection_mode="pending_only",
                write_policy="overwrite",
                llm_task_id=task_id,
                total_images=total_images,
                result_summary={"phase": "extracting"} if mode == "video" else None,
            ))
            session.add(LLMAnnotationTaskRecord(
                id=task_id, category_id=category_id,
                llm_provider=PROVIDER, llm_model=selected_model,
                prompt=target.strip(), status="pending",
                total_images=total_images,
                result_summary={
                    "phase": "extracting",
                    "iteration_id": iteration_id,
                    "iteration_no": 1,
                } if mode == "video" else {"iteration_id": iteration_id, "iteration_no": 1},
            ))
            await session.commit()
    except HTTPException:
        if video_path:
            video_path.unlink(missing_ok=True)
        await _cleanup_category(category_id)
        raise
    except Exception as exc:
        if video_path:
            video_path.unlink(missing_ok=True)
        await _cleanup_category(category_id)
        logger.error(f"自动标注准备失败: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc

    # 5) 起后台任务（抽帧 + 标注），立即返回
    job = asyncio.create_task(_run_auto_annotate_job(
        task_id=task_id, category_id=category_id, prompt=target.strip(),
        video_path=video_path, density=density,
        max_concurrent=settings.LLM_MAX_CONCURRENT,
        model_name=selected_model,
        iteration_id=iteration_id,
    ))
    _running_tasks[task_id] = job

    return AutoAnnotateResponse(
        task_id=task_id, iteration_id=iteration_id,
        category_id=category_id, category_name=category_name,
        noun=noun, mode=mode, total_images=total_images,
    )


# ── background job ───────────────────────────────────────────────────────────


async def _run_auto_annotate_job(
    task_id: str,
    category_id: str,
    prompt: str,
    video_path: Optional[Path],
    density: str,
    max_concurrent: int,
    model_name: str,
    iteration_id: Optional[str] = None,
) -> None:
    """后台：视频先抽帧成 pending 图片，然后复用现有标注流程逐张标注。"""
    try:
        # 视频：抽帧（放线程池，避免阻塞事件循环）→ 建 pending 行
        if video_path is not None:
            raw_dir = _raw_dir(category_id)
            try:
                frames = await asyncio.to_thread(extract_frames, video_path, raw_dir, density)
            except Exception as exc:
                logger.error(f"视频抽帧失败 {task_id}: {exc}")
                await _mark_failed(task_id, f"视频抽帧失败：{exc}", iteration_id)
                return

            # to_thread 抽帧不可中断：若抽帧期间任务已被取消，丢弃已抽帧并退出。
            async with AsyncSessionLocal() as session:
                task = await session.get(LLMAnnotationTaskRecord, task_id)
                if task is not None and task.status == "cancelled":
                    for fr in frames:
                        try:
                            Path(fr["stored_path"]).unlink(missing_ok=True)
                        except OSError:
                            pass
                    logger.info(f"自动标注任务在抽帧后检测到已取消，已清理抽帧 {task_id}")
                    return

            # 建 pending 行；若 DB 写入失败，清理已落盘帧避免磁盘泄露。
            try:
                async with AsyncSessionLocal() as session:
                    for fr in frames:
                        session.add(DatasetImageRecord(
                            id=fr["image_id"], category_id=category_id,
                            filename=fr["filename"], stored_path=fr["stored_path"],
                            width=fr["width"], height=fr["height"],
                            annotation_status="pending", box_count=0,
                        ))
                    await session.flush()
                    await _recompute_counts(session, category_id)
                    await session.commit()
            except Exception as exc:
                for fr in frames:
                    try:
                        Path(fr["stored_path"]).unlink(missing_ok=True)
                    except OSError:
                        pass
                logger.error(f"保存抽帧记录失败 {task_id}: {exc}")
                await _mark_failed(task_id, f"保存抽帧失败：{exc}", iteration_id)
                return
            # 帧已落盘，删源视频省磁盘
            try:
                video_path.unlink(missing_ok=True)
            except OSError:
                pass

        # 取该类别的 pending 图片；同步 total_images 供进度计算
        async with AsyncSessionLocal() as session:
            rows = (await session.execute(
                select(DatasetImageRecord).where(
                    DatasetImageRecord.category_id == category_id,
                    DatasetImageRecord.annotation_status == "pending",
                )
            )).scalars().all()
            images = list(rows)  # expire_on_commit=False，分离后仍可读属性
            task = await session.get(LLMAnnotationTaskRecord, task_id)
            if task is not None:
                task.total_images = len(images)
            if iteration_id:
                iteration = await session.get(AnnotationIterationRecord, iteration_id)
                if iteration is not None:
                    iteration.total_images = len(images)
                    iteration.result_summary = {"phase": "annotating", "task_id": task_id}
            await session.commit()

        if not images:
            await _mark_failed(task_id, "没有可标注的图片。", iteration_id)
            return

        # 复用现有标注执行器：内部会置 running→批量标注→保存→finished
        await _run_annotation_task(
            task_id=task_id, category_id=category_id, prompt=prompt,
            images=images, max_concurrent=max_concurrent,
            model_name=model_name,
            iteration_id=iteration_id,
            write_policy="overwrite",
        )
    except asyncio.CancelledError:
        async with AsyncSessionLocal() as session:
            task = await session.get(LLMAnnotationTaskRecord, task_id)
            if task and task.status in ("pending", "running"):
                task.status = "cancelled"
                task.finished_at = datetime.utcnow()
                if iteration_id:
                    iteration = await session.get(AnnotationIterationRecord, iteration_id)
                    if iteration:
                        iteration.status = "cancelled"
                        iteration.finished_at = task.finished_at
                await session.commit()
        raise
    except Exception as exc:
        logger.error(f"自动标注任务失败 {task_id}: {exc}")
        await _mark_failed(task_id, str(exc), iteration_id)
    finally:
        # 视频源文件抽帧后即可删除（成功路径已删，这里兜底覆盖失败/取消路径）。
        if video_path is not None:
            try:
                video_path.unlink(missing_ok=True)
            except OSError:
                pass
        _running_tasks.pop(task_id, None)


async def _mark_failed(task_id: str, error: str, iteration_id: Optional[str] = None) -> None:
    try:
        async with AsyncSessionLocal() as session:
            task = await session.get(LLMAnnotationTaskRecord, task_id)
            if task and task.status not in ("finished", "cancelled"):
                task.status = "failed"
                task.error = error
                task.finished_at = datetime.utcnow()
            if iteration_id:
                iteration = await session.get(AnnotationIterationRecord, iteration_id)
                if iteration and iteration.status not in ("ready_for_training", "cancelled"):
                    iteration.status = "failed"
                    iteration.error = error
                    iteration.finished_at = datetime.utcnow()
            await session.commit()
    except Exception as exc:
        logger.warning(f"标记任务失败状态失败 {task_id}: {exc}")
