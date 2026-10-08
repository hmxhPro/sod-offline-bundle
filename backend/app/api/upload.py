"""
app/api/upload.py
-----------------
POST /api/upload  – Accept a video file and return its metadata.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, HTTPException, UploadFile, File, status

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import UploadResponse
from app.utils.video_utils import get_video_info

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a video file for detection",
)
async def upload_video(
    file: UploadFile = File(..., description="Video file (mp4, avi, mov, etc.)")
) -> UploadResponse:
    """
    Accept a video upload and persist it to the upload directory.

    Returns a `video_id` that must be passed to POST /api/detect.
    """
    # ── Validate file extension ────────────────────────────────────────────
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    video_id = str(uuid.uuid4())
    dest_path = settings.UPLOAD_DIR / f"{video_id}{suffix}"

    # ── Disk water-mark: refuse new uploads when free space is low (P-1) ──
    try:
        free = shutil.disk_usage(settings.UPLOAD_DIR).free
    except OSError:
        free = None
    if free is not None and free < settings.MIN_FREE_DISK_BYTES:
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail="磁盘空间不足，暂时无法接收上传，请联系管理员清理后重试。",
        )

    # ── Stream to disk with a hard size cap (P-1) ─────────────────────────
    # An unbounded write lets one upload fill the disk. Accumulate bytes and
    # abort + unlink the moment the cap is exceeded. The HTTPException is caught
    # by the `except HTTPException` arm below (ordered before the generic
    # handler) so the partial file is removed and 413 is returned, not 500.
    max_bytes = settings.MAX_UPLOAD_BYTES
    total_bytes = 0
    try:
        async with aiofiles.open(dest_path, "wb") as out_file:
            while chunk := await file.read(4 * 1024 * 1024):  # 4 MB chunks
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"视频文件超过上限 {max_bytes // (1024 * 1024)} MB。",
                    )
                await out_file.write(chunk)
    except HTTPException:
        dest_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        logger.error(f"Upload failed for {file.filename}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        ) from exc

    # ── Read video metadata ───────────────────────────────────────────────
    try:
        info = get_video_info(dest_path)
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File saved but could not be read as a valid video: {exc}",
        ) from exc

    logger.info(
        f"Video uploaded: {video_id} | {file.filename} | "
        f"{total_bytes / 1e6:.1f} MB | {info['total_frames']} frames"
    )

    return UploadResponse(
        video_id=video_id,
        filename=file.filename or "",
        size_bytes=total_bytes,
        duration_seconds=info["duration_seconds"],
        fps=info["fps"],
        total_frames=info["total_frames"],
    )
