"""
app/api/training.py
--------------------
Start training jobs + query training history/status (REQ2).

POST /api/categories/{id}/train      -> 202, spawns a training subprocess
GET  /api/training/jobs              -> training history (newest first)
GET  /api/training/jobs/{job_id}     -> status/progress (polled by the UI)
POST /api/training/jobs/{job_id}/cancel -> terminate a running job
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select

from app.core.logging import logger
from app.db.models import CategoryRecord, DatasetImageRecord, TrainingJobRecord
from app.db.session import AsyncSessionLocal
from app.models.schemas import TrainingJobItem, TrainRequest, TrainResponse
from app.services.training_manager import TrainingBusyError, training_manager

router = APIRouter()


def _job_item(rec) -> TrainingJobItem:
    """Build a TrainingJobItem, surfacing val_is_train from the params JSON.

    val_is_train is stored inside TrainingJobRecord.params (not its own column),
    so it isn't auto-mapped by model_validate — lift it out explicitly (M-1)."""
    item = TrainingJobItem.model_validate(rec)
    item.val_is_train = bool((rec.params or {}).get("val_is_train"))
    return item


@router.post(
    "/categories/{category_id}/train",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="基于类别的已标注数据集启动训练",
)
async def start_training(category_id: str, body: TrainRequest) -> TrainResponse:
    # Gather category + annotated images.
    try:
        async with AsyncSessionLocal() as session:
            cat = await session.get(CategoryRecord, category_id)
            if cat is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "类别不存在。")
            rows = (
                await session.execute(
                    select(DatasetImageRecord).where(
                        DatasetImageRecord.category_id == category_id,
                        DatasetImageRecord.annotation_status == "annotated",
                    )
                )
            ).scalars().all()
            category = {"id": cat.id, "name": cat.name}
            images = [{"id": r.id, "stored_path": r.stored_path} for r in rows]
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"start_training lookup failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc

    if not images:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "该类别没有已标注的图片，请先完成标注再训练。",
        )

    params = {
        "epochs": body.epochs,
        "imgsz": body.imgsz,
        "batch": body.batch,
        "base_model": body.base_model,
        "mosaic": body.mosaic,
        "scale": body.scale,
        "close_mosaic": body.close_mosaic,
    }
    try:
        job_id = await training_manager.start(category, images, params)
    except TrainingBusyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    return TrainResponse(
        job_id=job_id,
        category_id=category_id,
        model_name=category["name"],
        status="pending",
    )


@router.get(
    "/training/jobs",
    response_model=List[TrainingJobItem],
    summary="训练任务历史（新→旧）",
)
async def list_training_jobs(
    category_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> List[TrainingJobItem]:
    try:
        async with AsyncSessionLocal() as session:
            stmt = select(TrainingJobRecord).order_by(TrainingJobRecord.created_at.desc())
            if category_id:
                stmt = stmt.where(TrainingJobRecord.category_id == category_id)
            stmt = stmt.limit(limit).offset(offset)
            rows = (await session.execute(stmt)).scalars().all()
    except Exception as exc:
        logger.warning(f"list_training_jobs failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc
    return [_job_item(r) for r in rows]


@router.get(
    "/training/jobs/{job_id}",
    response_model=TrainingJobItem,
    summary="获取训练任务状态/进度",
)
async def get_training_job(job_id: str) -> TrainingJobItem:
    try:
        async with AsyncSessionLocal() as session:
            rec = await session.get(TrainingJobRecord, job_id)
    except Exception as exc:
        logger.warning(f"get_training_job failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc
    if rec is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "训练任务不存在。")
    return _job_item(rec)


@router.post(
    "/training/jobs/{job_id}/cancel",
    summary="取消正在运行的训练任务",
)
async def cancel_training_job(job_id: str) -> dict:
    ok = await training_manager.cancel(job_id)
    if not ok:
        raise HTTPException(status.HTTP_409_CONFLICT, "该任务未在运行，无法取消。")
    return {"job_id": job_id, "action": "cancel", "status": "accepted"}
