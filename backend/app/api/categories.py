"""
app/api/categories.py
----------------------
Category CRUD for the YOLOE training workflow (REQ2).

A category is a user-created class to train; its name becomes the trained
model name. Rows live in PostgreSQL (yoloe_categories); per-category raw
images + working annotations live on disk under DATASETS_DIR / ANNOTATIONS_DIR.
"""

from __future__ import annotations

import shutil
import uuid
from typing import List

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.core.logging import logger
from app.db.models import (
    CategoryRecord, DatasetImageRecord, LLMAnnotationTaskRecord,
    TrainedModelRecord, TrainingJobRecord,
)
from app.db.session import AsyncSessionLocal
from app.models.schemas import CategoryCreate, CategoryItem
from app.services.training_manager import training_manager

router = APIRouter()

ACTIVE_TRAIN_STATUSES = frozenset({"pending", "running"})


@router.post(
    "/categories",
    response_model=CategoryItem,
    status_code=status.HTTP_201_CREATED,
    summary="创建一个训练类别",
)
async def create_category(body: CategoryCreate) -> CategoryItem:
    name = body.name.strip()
    if not name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "类别名不能为空。")
    try:
        async with AsyncSessionLocal() as session:
            dup = await session.scalar(
                select(CategoryRecord).where(CategoryRecord.name == name)
            )
            if dup is not None:
                raise HTTPException(status.HTTP_409_CONFLICT, f"类别「{name}」已存在。")
            rec = CategoryRecord(
                id=str(uuid.uuid4()),
                name=name,
                description=body.description,
                status="draft",
            )
            session.add(rec)
            await session.commit()
            await session.refresh(rec)
            return CategoryItem.model_validate(rec)
    except HTTPException:
        raise
    except IntegrityError:
        raise HTTPException(status.HTTP_409_CONFLICT, f"类别「{name}」已存在。")
    except Exception as exc:
        logger.warning(f"create_category failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc


@router.get(
    "/categories",
    response_model=List[CategoryItem],
    summary="列出全部训练类别（新→旧）",
)
async def list_categories() -> List[CategoryItem]:
    try:
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(CategoryRecord).order_by(CategoryRecord.created_at.desc())
                )
            ).scalars().all()
    except Exception as exc:
        logger.warning(f"list_categories failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc
    return [CategoryItem.model_validate(r) for r in rows]


@router.get(
    "/categories/{category_id}",
    response_model=CategoryItem,
    summary="获取单个类别",
)
async def get_category(category_id: str) -> CategoryItem:
    try:
        async with AsyncSessionLocal() as session:
            rec = await session.get(CategoryRecord, category_id)
    except Exception as exc:
        logger.warning(f"get_category failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc
    if rec is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "类别不存在。")
    return CategoryItem.model_validate(rec)


@router.delete(
    "/categories/{category_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除类别（含图片/标注/数据集；保留已训练权重文件）",
)
async def delete_category(category_id: str):
    if training_manager.active_job_id is not None:
        # Cheap guard; the active job might belong to this category.
        async with AsyncSessionLocal() as session:
            job = await session.get(TrainingJobRecord, training_manager.active_job_id)
        if job is not None and job.category_id == category_id and job.status in ACTIVE_TRAIN_STATUSES:
            raise HTTPException(status.HTTP_409_CONFLICT, "该类别正在训练中，请等待完成后再删除。")

    # 取消该类别正在运行的（自动）标注任务，避免删除后后台任务把数据写回已删类别。
    try:
        from app.api.llm_annotation import _running_tasks

        async with AsyncSessionLocal() as session:
            active_task_ids = (
                await session.execute(
                    select(LLMAnnotationTaskRecord.id).where(
                        LLMAnnotationTaskRecord.category_id == category_id,
                        LLMAnnotationTaskRecord.status.in_(("pending", "running")),
                    )
                )
            ).scalars().all()
        for tid in active_task_ids:
            t = _running_tasks.pop(tid, None)
            if t is not None:
                t.cancel()
    except Exception as exc:
        logger.warning(f"取消类别 {category_id} 的标注任务失败: {exc}")

    try:
        async with AsyncSessionLocal() as session:
            rec = await session.get(CategoryRecord, category_id)
            if rec is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "类别不存在。")
            await session.execute(
                delete(DatasetImageRecord).where(DatasetImageRecord.category_id == category_id)
            )
            await session.execute(
                delete(TrainedModelRecord).where(TrainedModelRecord.category_id == category_id)
            )
            await session.execute(
                delete(TrainingJobRecord).where(TrainingJobRecord.category_id == category_id)
            )
            await session.execute(
                delete(LLMAnnotationTaskRecord).where(LLMAnnotationTaskRecord.category_id == category_id)
            )
            await session.delete(rec)
            await session.commit()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"delete_category failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc

    # Best-effort on-disk cleanup (raw images + working annotations + datasets).
    for d in (settings.DATASETS_DIR / category_id, settings.ANNOTATIONS_DIR / category_id):
        if d.exists():
            try:
                shutil.rmtree(d)
            except OSError as exc:
                logger.warning(f"failed to remove {d}: {exc}")
    logger.info(f"deleted category {category_id}")
    return None
