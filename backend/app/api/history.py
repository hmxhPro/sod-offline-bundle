"""
app/api/history.py
------------------
GET    /api/tasks                — list past tasks (paginated, optional date filter)
GET    /api/task/{task_id}/frames — list saved frame filenames for a task
DELETE /api/task/{task_id}       — remove a single task (DB row + result frames + ZIP + upload)
DELETE /api/tasks                — wipe all history (DB + every result dir + every upload)

History rows live in PostgreSQL (table: detection_tasks). Frame files live
on disk under RESULTS_DIR/{task_id}/.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import func, select

from app.core.config import settings
from app.core.logging import logger
from app.db.models import OperationLog, TaskRecord
from app.db.session import AsyncSessionLocal
from app.models.schemas import TaskHistoryItem
from app.services.task_manager import task_manager

router = APIRouter()

# Statuses that prevent a task from being deleted — the user must cancel
# (or wait for) an in-flight task before purging its records.
ACTIVE_STATUSES = frozenset({"pending", "running", "paused", "packaging"})


async def _log_operation(action: str, detail: str, affected_count: int) -> None:
    """Append an audit row for a destructive operation (R-10). Best-effort: a
    DB outage must not block the delete itself, so failures are swallowed."""
    try:
        async with AsyncSessionLocal() as session:
            session.add(OperationLog(
                action=action, detail=detail[:2000], affected_count=affected_count,
            ))
            await session.commit()
    except Exception as exc:
        logger.warning(f"operation_log write failed ({action}): {exc}")


def _reject_unsafe_id(task_id: str) -> None:
    if "/" in task_id or "\\" in task_id or ".." in task_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid task_id.",
        )


def _remove_results_dir(task_id: str) -> None:
    """Wipe RESULTS_DIR/{task_id}/ — frames + ZIP. Best-effort."""
    task_dir = settings.RESULTS_DIR / task_id
    if task_dir.exists():
        try:
            shutil.rmtree(task_dir)
        except OSError as exc:
            logger.warning(f"failed to remove results dir {task_dir}: {exc}")


def _remove_upload(video_id: str) -> None:
    """Delete uploads/{video_id}.* (extension unknown). Best-effort."""
    if not video_id:
        return
    for p in settings.UPLOAD_DIR.glob(f"{video_id}.*"):
        try:
            p.unlink()
        except OSError as exc:
            logger.warning(f"failed to remove upload {p}: {exc}")


@router.get(
    "/tasks",
    response_model=List[TaskHistoryItem],
    summary="List past detection tasks (newest first)",
)
async def list_history(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    date: Optional[str] = Query(
        None,
        description="Filter to a single day (YYYY-MM-DD, server local time).",
    ),
) -> List[TaskHistoryItem]:
    """Return rows from `detection_tasks`, newest first."""
    try:
        async with AsyncSessionLocal() as session:
            stmt = select(TaskRecord).order_by(TaskRecord.created_at.desc())
            if date:
                try:
                    day = datetime.fromisoformat(date).date()
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Invalid date format: {date} (expected YYYY-MM-DD)",
                    ) from exc
                stmt = stmt.where(func.date(TaskRecord.created_at) == day)
            stmt = stmt.limit(limit).offset(offset)
            rows = (await session.execute(stmt)).scalars().all()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"history list_history failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="History database unavailable.",
        ) from exc

    return [TaskHistoryItem.model_validate(r) for r in rows]


@router.get(
    "/task/{task_id}/frames",
    response_model=List[str],
    summary="List saved annotated frame filenames for a task",
)
async def list_frames(task_id: str) -> List[str]:
    """
    Return JPG filenames in `RESULTS_DIR/{task_id}/`, sorted ascending.
    The frontend builds image URLs via /api/frame/{task_id}/{filename}.
    """
    _reject_unsafe_id(task_id)
    task_dir = settings.RESULTS_DIR / task_id
    if not task_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No saved frames for task '{task_id}'.",
        )
    return sorted(p.name for p in task_dir.glob("frame_*.jpg"))


@router.delete(
    "/task/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a single task: DB row + result frames + ZIP + upload",
)
async def delete_task(task_id: str) -> Response:
    """
    Hard-delete one task. Refuses if the task is still active (pending /
    running / paused / packaging) — the caller should cancel it first.

    Removes:
      - the row in `detection_tasks`
      - RESULTS_DIR/{task_id}/ (frame JPGs + ZIP)
      - uploads/{video_id}.* — only if no other task still references it
      - in-memory state in task_manager
    """
    _reject_unsafe_id(task_id)

    in_mem = task_manager.get_task(task_id)
    if in_mem is not None and in_mem.status in ACTIVE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"任务正在 '{in_mem.status}' 中，请先取消再删除。",
        )

    video_id: Optional[str] = None
    upload_safe_to_delete = False
    db_row_existed = False

    try:
        async with AsyncSessionLocal() as session:
            row = await session.get(TaskRecord, task_id)
            if row is not None:
                db_row_existed = True
                video_id = row.video_id
                if video_id:
                    other_uses = await session.scalar(
                        select(func.count(TaskRecord.task_id)).where(
                            TaskRecord.video_id == video_id,
                            TaskRecord.task_id != task_id,
                        )
                    )
                    upload_safe_to_delete = (other_uses or 0) == 0
                await session.delete(row)
                await session.commit()
    except Exception as exc:
        logger.warning(f"DB delete failed for task {task_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="History database unavailable.",
        ) from exc

    # Fall back to in-memory video_id when no DB row was present
    # (task was created in this process but never persisted).
    if video_id is None and in_mem is not None:
        video_id = in_mem.video_id
        upload_safe_to_delete = True

    if not db_row_existed and in_mem is None and not (settings.RESULTS_DIR / task_id).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )

    _remove_results_dir(task_id)
    if upload_safe_to_delete and video_id:
        _remove_upload(video_id)
    task_manager.remove_task(task_id)

    await _log_operation(
        "delete_task",
        f"task_id={task_id} video_id={video_id} upload_purged={upload_safe_to_delete}",
        affected_count=1,
    )
    logger.info(f"deleted task {task_id} (video_id={video_id}, upload_purged={upload_safe_to_delete})")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/tasks",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete ALL history: every DB row + each task's result dir + upload",
)
async def delete_all_history(
    confirm: str = Query(
        "",
        description="必须传 confirm=DELETE_ALL 以确认这是不可逆的批量删除。",
    ),
) -> Response:
    """
    Nuclear cleanup of detection-task history. Refuses if any task is still
    active in memory, and requires an explicit `confirm=DELETE_ALL` token to
    guard against an accidental double-click (R-10).

    Deletion is now **per row**: each deleted task's RESULTS_DIR/{task_id}/ and
    its upload are removed individually. The previous implementation wiped the
    whole RESULTS_DIR/UPLOAD_DIR via iterdir(), which also destroyed the shared
    image-detection outputs under RESULTS_DIR/imgdet/ (not task history at all).
    We never touch parent dirs or the imgdet subtree.
    """
    if confirm != "DELETE_ALL":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="危险操作未确认：请带 confirm=DELETE_ALL 重试（将永久删除全部检测历史）。",
        )

    active = [st.task_id for st in task_manager.list_tasks() if st.status in ACTIVE_STATUSES]
    if active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"仍有 {len(active)} 个任务正在进行，请先取消后再清空历史。",
        )

    deleted: list[tuple[str, Optional[str]]] = []   # (task_id, video_id)
    try:
        async with AsyncSessionLocal() as session:
            rows = (await session.execute(select(TaskRecord))).scalars().all()
            for r in rows:
                deleted.append((r.task_id, r.video_id))
                await session.delete(r)
            await session.commit()
    except Exception as exc:
        logger.warning(f"DB bulk delete failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="History database unavailable.",
        ) from exc

    # Remove ONLY the per-task artifacts we just deleted rows for — never an
    # iterdir() of the parent (that would nuke RESULTS_DIR/imgdet/ and any other
    # non-task content). Each task's upload goes too (all referencing rows are
    # being deleted in this wipe).
    for task_id, video_id in deleted:
        _remove_results_dir(task_id)
        if video_id:
            _remove_upload(video_id)

    # Drop in-memory state too — including tasks created in this process but
    # never persisted to DB.
    for st in list(task_manager.list_tasks()):
        task_manager.remove_task(st.task_id)
    for task_id, _ in deleted:
        task_manager.remove_task(task_id)

    await _log_operation(
        "delete_all_history",
        f"deleted {len(deleted)} task(s); imgdet/ and non-task dirs preserved",
        affected_count=len(deleted),
    )
    logger.info(f"deleted ALL history: {len(deleted)} task(s) (per-row; imgdet preserved)")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
