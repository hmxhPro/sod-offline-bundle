"""
app/api/detect.py
-----------------
POST /api/detect  – Start a detection task (async).
GET  /api/task/{task_id}  – Query task status and results.
GET  /api/stream/{task_id}  – SSE stream of frame results.
GET  /api/download/{task_id}  – Download results ZIP.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse

from app.core.config import settings
from app.core import state as app_state
from app.core.logging import logger
from app.db.models import TaskRecord, TrainedModelRecord
from app.db.session import AsyncSessionLocal
from app.models.schemas import (
    DetectRequest,
    DetectResponse,
    StreamEvent,
    TaskState,
    TaskStatusResponse,
    TaskStatus,
)
from app.services.download_naming import (
    MODEL_PROMPT_PREFIX,
    build_zip_filename,
    content_disposition,
)
from app.services.pipeline import run_detection_pipeline
from app.services.prompt_normalizer import normalize_prompt
from app.services.color_filter import get_preset_color_filter
from app.services.task_manager import task_manager

router = APIRouter()

# Keep strong references to background tasks so they aren't GC'd mid-run
_background_tasks: set = set()


# ────────────────────────────────────────────────────────────────────────────
# POST /api/detect
# ────────────────────────────────────────────────────────────────────────────

@router.post(
    "/detect",
    response_model=DetectResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a video detection task",
)
async def start_detection(body: DetectRequest) -> DetectResponse:
    """
    Queue a detection task for the specified video_id and prompt.

    Processing runs asynchronously in the background.
    Use `GET /api/stream/{task_id}` for real-time results via SSE,
    or poll `GET /api/task/{task_id}` for status.
    """
    # ── Validate video_id  ─────────────────────────────────────────────────
    video_files = list(settings.UPLOAD_DIR.glob(f"{body.video_id}.*"))
    if not video_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video '{body.video_id}' not found. "
                   f"Please upload first via POST /api/upload.",
        )
    video_path = video_files[0]

    # ── Resolve detection mode: trained model XOR natural-language prompt ──
    prompt = (body.prompt or "").strip()
    model_id = (body.model_id or "").strip() or None
    if bool(model_id) == bool(prompt):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="请二选一：提供 prompt（自然语言检测）或 model_id（使用已训练模型）。",
        )

    weights_path = None

    if model_id:
        # ── Trained-model mode: classes are baked into the weights ─────────
        try:
            async with AsyncSessionLocal() as session:
                rec = await session.get(TrainedModelRecord, model_id)
        except Exception as exc:
            logger.warning(f"detect model lookup failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="数据库不可用。",
            ) from exc
        if rec is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "所选模型不存在。")
        # Fail fast (before queuing) if the weights file is missing — otherwise
        # the task would only fail on its first detection frame.
        if not rec.weights_path or not Path(rec.weights_path).exists():
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "模型权重文件不存在，无法用于检测。",
            )

        weights_path = rec.weights_path
        class_names = list(rec.class_names.values()) if isinstance(rec.class_names, dict) else []
        # No prompt → no normalization. The model's native model.names provide
        # the labels; keep dino_prompt / original_prompt / label_mapping EMPTY so
        # the pipeline's relabel step does NOT overwrite them.
        dino_prompt = ""
        original_prompt = ""
        label_mapping = {}
        color_filters = []
        vlm_query = "、".join(str(c) for c in class_names if c)
        # MODEL_PROMPT_PREFIX is the contract that lets the download endpoint
        # recover the model name from the persisted prompt (download_naming.py).
        display_prompt = f"{MODEL_PROMPT_PREFIX}{rec.name}"
        # create_task persists body.prompt — store a readable label for history.
        body = body.model_copy(update={"prompt": display_prompt})
        logger.info(
            f"Detection via trained model: id={model_id} name='{rec.name}' "
            f"classes={class_names} weights={weights_path}"
        )
    else:
        # ── Natural-language mode (existing open-vocabulary behavior) ──────
        # Open-vocabulary detection needs the preloaded singleton detector. If
        # preload failed (missing weights / GPU OOM), fail fast with 503 instead
        # of triggering an expensive failing reload on every request (R-8). The
        # trained-model branch above is independent (loads its own weights).
        if not app_state.model_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="检测模型尚未就绪（加载失败或加载中），请稍后重试，或查看 /readyz。",
            )
        normalized = normalize_prompt(prompt)
        dino_prompt = normalized.dino_prompt
        vlm_query = normalized.vlm_query
        original_prompt = prompt
        display_prompt = prompt

        # Build label mapping: English -> Chinese
        label_mapping = {}
        if normalized.targets:
            for target in normalized.targets:
                zh_label = target.get("zh", "")
                en_phrases = target.get("en", [])
                if zh_label and en_phrases:
                    for en in en_phrases:
                        # Remove trailing period and convert to lowercase
                        en_clean = en.strip().rstrip('.').lower()
                        label_mapping[en_clean] = zh_label

        # Get color filters from LLM or use presets
        color_filters = normalized.color_filters
        if not color_filters and normalized.targets:
            # Try to get preset color filters for known targets
            for target in normalized.targets:
                zh_label = target.get("zh", "")
                preset = get_preset_color_filter(zh_label)
                if preset:
                    color_filters.extend(preset)
                    logger.info(f"Using preset color filter for '{zh_label}'")

        logger.info(
            f"Prompt normalized: '{prompt}' → dino='{dino_prompt}' | "
            f"label_mapping={label_mapping} | "
            f"color_filters={len(color_filters)} rules"
        )

    # ── Determine VLM setting ─────────────────────────────────────────────
    enable_vlm = body.enable_vlm if body.enable_vlm is not None else settings.VLM_ENABLED

    # ── Create task ────────────────────────────────────────────────────────
    task_state = await task_manager.create_task(body)

    # ── Launch background coroutine ────────────────────────────────────────
    task = asyncio.create_task(
        run_detection_pipeline(
            task_id=task_state.task_id,
            video_path=video_path,
            prompt=dino_prompt,
            vlm_query=vlm_query,
            enable_vlm=enable_vlm,
            task_manager=task_manager,
            detection_interval=body.detection_interval,
            box_threshold=body.box_threshold,
            text_threshold=body.text_threshold,
            original_prompt=original_prompt,
            label_mapping=label_mapping,
            color_filters=color_filters,
            weights_path=weights_path,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    logger.info(
        f"Detection task queued: {task_state.task_id} | "
        f"video={body.video_id} | prompt='{display_prompt}'"
    )

    return DetectResponse(
        task_id=task_state.task_id,
        video_id=body.video_id,
        prompt=display_prompt,
        status=TaskStatus.PENDING,
    )


# ────────────────────────────────────────────────────────────────────────────
# GET /api/task/{task_id}
# ────────────────────────────────────────────────────────────────────────────

@router.get(
    "/task/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get task status and progress (no frame data)",
)
async def get_task(task_id: str) -> TaskStatusResponse:
    """
    Return the current state of a detection task.

    - `status`: pending | running | finished | failed
    - `progress`: 0.0 – 1.0
    - `zip_ready`: true when the download ZIP is available

    Frame results are streamed via SSE; this endpoint only returns status.
    Falls back to the PostgreSQL archive when the task is not in memory
    (e.g. after a backend restart).
    """
    state = task_manager.get_task(task_id)
    if state is not None:
        return TaskStatusResponse(
            task_id=state.task_id,
            video_id=state.video_id,
            prompt=state.prompt,
            status=state.status,
            progress=state.progress,
            total_frames=state.total_frames,
            processed_frames=state.processed_frames,
            detection_frame_count=state.detection_frame_count,
            error=state.error,
            zip_ready=state.zip_ready,
            early_terminated=state.early_terminated,
            termination_reason=state.termination_reason,
        )

    # In-memory miss: try the DB archive.
    try:
        from app.db.session import AsyncSessionLocal
        from app.db.models import TaskRecord

        async with AsyncSessionLocal() as session:
            row = await session.get(TaskRecord, task_id)
    except Exception as exc:
        logger.warning(f"DB lookup failed for task {task_id}: {exc}")
        row = None

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )
    return TaskStatusResponse(
        task_id=row.task_id,
        video_id=row.video_id,
        prompt=row.prompt,
        status=TaskStatus(row.status),
        progress=row.progress,
        total_frames=row.total_frames,
        processed_frames=row.processed_frames,
        error=row.error,
        zip_ready=row.zip_ready,
        early_terminated=row.early_terminated,
        termination_reason=row.termination_reason,
    )


# ────────────────────────────────────────────────────────────────────────────
# GET /api/stream/{task_id}  – Server-Sent Events
# ────────────────────────────────────────────────────────────────────────────

@router.get(
    "/stream/{task_id}",
    summary="Stream detection results frame by frame via SSE",
    response_class=StreamingResponse,
)
async def stream_task(task_id: str):
    """
    Server-Sent Events (SSE) endpoint.

    Events:
      - `frame`   – one frame processed; includes base64-encoded result image
      - `done`    – processing complete
      - `error`   – processing failed

    Each SSE message has the format:
        data: <JSON StreamEvent>\\n\\n
    """
    state = task_manager.get_task(task_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )

    async def event_generator():
        # Heartbeat keeps SSE connections alive through any silent period
        # (ZIP packaging for long videos can take minutes). 15 s is well
        # below typical browser / proxy idle timeouts.
        HEARTBEAT_SECONDS = 15.0
        queue = task_manager._queues.get(task_id)
        try:
            while True:
                if queue is None:
                    # Queue was already cleaned up — fall through and stop.
                    break
                try:
                    event_type, payload = await asyncio.wait_for(
                        queue.get(), timeout=HEARTBEAT_SECONDS
                    )
                except asyncio.TimeoutError:
                    if task_manager._queues.get(task_id) is not queue:
                        # Our queue was retired (push_done / push_error ran)
                        # but this consumer never saw the single terminal
                        # sentinel — a zombie generator from a half-dead
                        # connection can steal it from the shared queue.
                        # Emit a synthetic terminal event so a live client
                        # isn't parked on the orphaned queue forever.
                        if state.status == TaskStatus.FAILED:
                            evt = StreamEvent(
                                event_type="error",
                                task_id=task_id,
                                error=state.error or "Task failed.",
                            )
                        elif state.status == TaskStatus.EARLY_TERMINATED:
                            evt = StreamEvent(
                                event_type="early_terminated",
                                task_id=task_id,
                                progress=state.progress,
                                total_frames=state.total_frames,
                                processed_frames=state.processed_frames,
                                detection_frame_count=state.detection_frame_count,
                                error=state.termination_reason
                                      or "Early termination triggered",
                            )
                        elif state.status == TaskStatus.CANCELLED:
                            # Must NOT fall through to "done": clients treat
                            # done as finished + downloadable, but a cancelled
                            # task has no ZIP (download returns 409).
                            evt = StreamEvent(
                                event_type="cancelled",
                                task_id=task_id,
                                progress=state.progress,
                                total_frames=state.total_frames,
                                processed_frames=state.processed_frames,
                                detection_frame_count=state.detection_frame_count,
                            )
                        else:
                            evt = StreamEvent(
                                event_type="done",
                                task_id=task_id,
                                progress=1.0,
                                total_frames=state.total_frames,
                                processed_frames=state.processed_frames,
                                detection_frame_count=state.detection_frame_count,
                            )
                        yield f"data: {evt.model_dump_json()}\n\n"
                        break
                    # SSE comment line — ignored by EventSource but keeps
                    # the TCP connection from being closed by proxies.
                    yield ": keepalive\n\n"
                    continue

                if event_type == "frame":
                    evt = StreamEvent(
                        event_type="frame",
                        task_id=task_id,
                        frame_result=payload,
                        progress=state.progress,
                        total_frames=state.total_frames,
                        processed_frames=state.processed_frames,
                        detection_frame_count=state.detection_frame_count,
                    )
                elif event_type == "packaging":
                    evt = StreamEvent(
                        event_type="packaging",
                        task_id=task_id,
                        progress=1.0,
                        total_frames=state.total_frames,
                        processed_frames=state.processed_frames,
                        detection_frame_count=state.detection_frame_count,
                    )
                elif event_type in ("paused", "resumed", "cancelled"):
                    evt = StreamEvent(
                        event_type=event_type,
                        task_id=task_id,
                        progress=state.progress,
                        total_frames=state.total_frames,
                        processed_frames=state.processed_frames,
                        detection_frame_count=state.detection_frame_count,
                    )
                elif event_type == "early_terminated":
                    evt = StreamEvent(
                        event_type="early_terminated",
                        task_id=task_id,
                        progress=state.progress,
                        total_frames=state.total_frames,
                        processed_frames=state.processed_frames,
                        detection_frame_count=state.detection_frame_count,
                        error=str(payload),  # termination reason
                    )
                elif event_type == "done":
                    evt = StreamEvent(
                        event_type="done",
                        task_id=task_id,
                        progress=1.0,
                        total_frames=state.total_frames,
                        processed_frames=state.processed_frames,
                        detection_frame_count=state.detection_frame_count,
                    )
                elif event_type == "error":
                    evt = StreamEvent(
                        event_type="error",
                        task_id=task_id,
                        error=str(payload),
                    )
                else:
                    continue

                yield f"data: {evt.model_dump_json()}\n\n"

                if event_type in ("done", "error", "early_terminated"):
                    break
        finally:
            # Do NOT clean up the queue here. The queue's lifetime is
            # owned by task_manager and ends with push_done / push_error.
            # A page refresh closes this generator (cancellation) — if
            # we deleted the queue here, the pipeline would silently
            # drop subsequent events and a reconnecting tab would see
            # nothing, making the task appear frozen.
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",     # Disable nginx buffering
            "Connection": "keep-alive",
        },
    )


# ────────────────────────────────────────────────────────────────────────────
# GET /api/download/{task_id}
# ────────────────────────────────────────────────────────────────────────────

@router.get(
    "/frame/{task_id}/{filename}",
    summary="Serve a single annotated frame image",
    response_class=FileResponse,
)
async def get_frame(task_id: str, filename: str):
    """Return a single annotated frame JPEG by filename."""
    # Prevent path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename.")
    img_path = settings.RESULTS_DIR / task_id / filename
    if not img_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Frame not found.")
    return FileResponse(path=str(img_path), media_type="image/jpeg")


@router.get(
    "/download/{task_id}",
    summary="Download the detection results ZIP archive",
    response_class=FileResponse,
)
async def download_results(task_id: str):
    """
    Download a ZIP file containing all annotated frames,
    results.json, and results.csv for the specified task.

    Only available once the task status is `finished` (or `early_terminated`).
    Falls back to the PostgreSQL archive + on-disk ZIP when the task is no
    longer in memory — e.g. after a backend restart, or once the in-memory
    LRU has evicted an old terminal task (A-1 / R-5). The ZIP only ever lives
    on disk, so download depends on disk + DB, not on live process state.
    """
    # Path-traversal guard (the task_id indexes a directory below).
    if "/" in task_id or "\\" in task_id or ".." in task_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid task_id.")

    _TERMINAL = {TaskStatus.FINISHED, TaskStatus.EARLY_TERMINATED}

    state = task_manager.get_task(task_id)

    # The DB row supplies the download filename's metadata (prompt +
    # created_at) and is the status authority when the task is no longer in
    # memory — e.g. after a backend restart, or once the in-memory LRU has
    # evicted an old terminal task (A-1 / R-5).
    try:
        async with AsyncSessionLocal() as session:
            row = await session.get(TaskRecord, task_id)
    except Exception as exc:
        logger.warning(f"download DB lookup failed for {task_id}: {exc}")
        row = None

    if state is not None:
        if state.status not in _TERMINAL:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Task is not finished yet (status={state.status}). "
                       "Wait for 'finished' or 'early_terminated' before downloading.",
            )
    else:
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task '{task_id}' not found.",
            )
        if row.status not in ("finished", "early_terminated"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Task is not finished yet (status={row.status}).",
            )

    zip_path = settings.RESULTS_DIR / task_id / "results.zip"
    if not zip_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ZIP file not found. The task may have failed during packaging.",
        )

    # Name the download after what was detected: "<model name>_<ts>.zip" for
    # trained-model tasks, "<prompt/classes>_<ts>.zip" for natural-language
    # tasks. Falls back to the ZIP's mtime when the DB row is unavailable.
    # The header is built by hand (not FileResponse's filename=) so non-ASCII
    # names keep an ASCII filename= fallback for curl -OJ.
    filename = build_zip_filename(
        prompt=state.prompt if state is not None else (row.prompt or ""),
        created_at=row.created_at if row is not None else None,
        fallback_ts=zip_path.stat().st_mtime,
    )

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        headers={
            "Content-Disposition": content_disposition(
                filename, ascii_fallback=f"detection_{task_id[:8]}.zip"
            ),
        },
    )


# ────────────────────────────────────────────────────────────────────────────
# POST /api/task/{task_id}/cancel | pause | resume
# ────────────────────────────────────────────────────────────────────────────

_ACTIVE_STATUSES = {
    TaskStatus.PENDING,
    TaskStatus.RUNNING,
    TaskStatus.PAUSED,
}


def _require_active_task(task_id: str) -> TaskState:
    state = task_manager.get_task(task_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )
    if state.status not in _ACTIVE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Task is not active (status={state.status}).",
        )
    return state


@router.post(
    "/task/{task_id}/cancel",
    summary="Request cancellation of a running detection task",
)
async def cancel_task(task_id: str) -> dict:
    _require_active_task(task_id)
    ok = task_manager.request_cancel(task_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cancel control unavailable for this task.",
        )
    return {"task_id": task_id, "action": "cancel", "status": "accepted"}


@router.post(
    "/task/{task_id}/pause",
    summary="Pause a running detection task",
)
async def pause_task(task_id: str) -> dict:
    state = _require_active_task(task_id)
    if state.status == TaskStatus.PAUSED:
        return {"task_id": task_id, "action": "pause", "status": "already_paused"}
    if state.status != TaskStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only running tasks can be paused.",
        )
    ok = task_manager.request_pause(task_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Pause control unavailable for this task.",
        )
    success = await task_manager.set_paused(task_id)
    if success:
        await task_manager.push_paused(task_id)
    return {"task_id": task_id, "action": "pause", "status": "accepted"}


@router.post(
    "/task/{task_id}/resume",
    summary="Resume a paused detection task",
)
async def resume_task(task_id: str) -> dict:
    state = _require_active_task(task_id)
    if state.status == TaskStatus.RUNNING:
        return {"task_id": task_id, "action": "resume", "status": "already_running"}
    if state.status != TaskStatus.PAUSED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only paused tasks can be resumed.",
        )
    ok = task_manager.request_resume(task_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Resume control unavailable for this task.",
        )
    success = await task_manager.set_resumed(task_id)
    if success:
        await task_manager.push_resumed(task_id)
    return {"task_id": task_id, "action": "resume", "status": "accepted"}


@router.post(
    "/task/{task_id}/terminate",
    summary="Manually terminate a running task and package results",
)
async def terminate_task(task_id: str) -> dict:
    """
    Manually terminate a running detection task.

    Unlike cancel (which discards results), terminate will:
    - Stop processing immediately
    - Package all processed frames into a ZIP
    - Make the ZIP available for download
    """
    state = _require_active_task(task_id)
    if state.status not in [TaskStatus.RUNNING, TaskStatus.PAUSED]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only running or paused tasks can be terminated.",
        )

    # Use the same cancel mechanism but mark for early termination
    ok = task_manager.request_terminate(task_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Terminate control unavailable for this task.",
        )

    return {"task_id": task_id, "action": "terminate", "status": "accepted"}
