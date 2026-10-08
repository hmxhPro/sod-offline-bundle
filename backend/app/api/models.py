"""
app/api/models.py
-----------------
Trained-model registry endpoints — backs the REQ3 model list.

GET    /api/models             -> all trained models, newest first
GET    /api/models/{model_id}  -> one model (detail for hover/inspect)
DELETE /api/models/{model_id}  -> drop the registry row + evict from cache
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select

from app.core.logging import logger
from app.db.models import TrainedModelRecord
from app.db.session import AsyncSessionLocal
from app.models.schemas import TrainedModelItem

router = APIRouter()


def _model_item(rec) -> TrainedModelItem:
    """Build a TrainedModelItem, surfacing val_is_train embedded in the metrics
    JSON (no dedicated column — see training_manager._insert_trained_model). M-1."""
    item = TrainedModelItem.model_validate(rec)
    item.val_is_train = bool((rec.metrics or {}).get("val_is_train"))
    return item


@router.get(
    "/models",
    response_model=List[TrainedModelItem],
    summary="列出全部已训练模型（新→旧）",
)
async def list_models(
    category_id: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> List[TrainedModelItem]:
    try:
        async with AsyncSessionLocal() as session:
            stmt = select(TrainedModelRecord).order_by(TrainedModelRecord.created_at.desc())
            if category_id:
                stmt = stmt.where(TrainedModelRecord.category_id == category_id)
            stmt = stmt.limit(limit).offset(offset)
            rows = (await session.execute(stmt)).scalars().all()
    except Exception as exc:
        logger.warning(f"list_models failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc
    return [_model_item(r) for r in rows]


@router.get(
    "/models/{model_id}",
    response_model=TrainedModelItem,
    summary="获取单个已训练模型详情",
)
async def get_model(model_id: str) -> TrainedModelItem:
    try:
        async with AsyncSessionLocal() as session:
            rec = await session.get(TrainedModelRecord, model_id)
    except Exception as exc:
        logger.warning(f"get_model failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc
    if rec is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "模型不存在。")
    return _model_item(rec)


@router.delete(
    "/models/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除模型注册项（保留权重文件）",
)
async def delete_model(model_id: str):
    try:
        async with AsyncSessionLocal() as session:
            rec = await session.get(TrainedModelRecord, model_id)
            if rec is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "模型不存在。")
            weights_path = rec.weights_path
            await session.delete(rec)
            await session.commit()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"delete_model failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc

    # Evict from the in-memory image-detection cache (best effort).
    try:
        from app.services.image_detector import get_registry
        if weights_path:
            get_registry().evict(weights_path)
    except Exception as exc:
        logger.debug(f"model cache evict skipped: {exc}")
    return None
