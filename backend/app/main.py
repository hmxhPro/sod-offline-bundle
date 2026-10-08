"""
app/main.py
-----------
FastAPI application entry point.

Registers all routers, configures CORS, and sets up startup/shutdown hooks.
"""

from __future__ import annotations

import asyncio
import os
import signal

# Silence FFmpeg / libav decoder chatter (SEI truncated, NAL warnings, etc.)
# Must be set before cv2 (and therefore the ffmpeg backend) is imported.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import detect, history, upload
from app.api import categories, dataset, training, image_detect, llm_annotation
from app.api import auto_annotate
from app.api import models as yoloe_models
from app.core import state as app_state
from app.core.config import settings
from app.core.errors import register_error_handling
from app.core.logging import logger, setup_logging


# Statuses that mean a detection task is still in flight (kept in sync with
# history.ACTIVE_STATUSES). TaskStatus is a str-Enum, so membership against
# these string literals works for both enum and raw-string values.
_ACTIVE_TASK_STATUSES = frozenset({"pending", "running", "paused", "packaging"})


def _reap_orphan_training(pid: int) -> bool:
    """Best-effort kill of a leftover training process group from a previous
    backend generation (R-1).

    Before signalling, confirm the pid is actually a `train_yoloe` process via
    /proc/<pid>/cmdline so a reused pid (a different, innocent process) is never
    killed. Tries the process group first (the child is its own group leader
    when spawned with start_new_session), then the bare pid as a fallback for
    legacy orphans spawned before that change. Returns True if a signal was sent.
    """
    if not pid or pid <= 0:
        return False
    # Windows has no /proc and no os.killpg; training subprocesses from a dead
    # backend generation don't survive a restart there anyway (they were spawned
    # in this process's own tree). Skip cleanly rather than fail on the
    # POSIX-only calls below.
    if os.name == "nt":
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read()
    except OSError:
        return False  # process gone or not inspectable
    if b"train_yoloe" not in cmdline:
        return False  # pid reused by an unrelated process — do not touch
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return False
    logger.info(f"Reaped orphan training process pid={pid}")
    return True


# ────────────────────────────────────────────────────────────────────────────
# Lifespan: startup & shutdown
# ────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan:
      - Startup: configure logging, ensure directories exist, preload model.
      - Shutdown: log graceful stop.
    """
    # ── Startup ───────────────────────────────────────────────────────────
    setup_logging(debug=settings.DEBUG)
    settings.ensure_dirs()

    logger.info("=" * 60)
    logger.info("Video Detection Agent starting up")
    logger.info(f"  Model:  {settings.DETECTION_MODEL}")
    logger.info(f"  Device: {settings.DEVICE}")
    logger.info(f"  Upload: {settings.UPLOAD_DIR}")
    logger.info(f"  Results:{settings.RESULTS_DIR}")
    logger.info("=" * 60)

    # ── DB: ensure schema, sweep stale active rows ───────────────────────
    # If the previous backend process crashed mid-task, those rows are
    # frozen at running/paused/packaging in the DB — mark them failed so
    # the history view reflects reality.
    try:
        from sqlalchemy import select, update, func as sql_func
        from app.db.session import AsyncSessionLocal, init_db
        from app.db.models import TaskRecord, TrainingJobRecord

        await init_db()
        stale_job_pids: list[int] = []
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                update(TaskRecord)
                .where(TaskRecord.status.in_(
                    ["pending", "running", "paused", "packaging"]
                ))
                .values(
                    status="failed",
                    error="服务非正常重启，任务中断",
                    finished_at=sql_func.now(),
                )
            )
            # Training subprocesses don't survive a backend restart. Read their
            # PIDs FIRST so we can reap any still-running orphans (R-1), then
            # sweep the rows to failed.
            stale_job_pids = [
                int(p) for p in (
                    await session.execute(
                        select(TrainingJobRecord.pid).where(
                            TrainingJobRecord.status.in_(["pending", "running"]),
                            TrainingJobRecord.pid.is_not(None),
                        )
                    )
                ).scalars().all()
                if p
            ]
            train_result = await session.execute(
                update(TrainingJobRecord)
                .where(TrainingJobRecord.status.in_(["pending", "running"]))
                .values(
                    status="failed",
                    error="服务非正常重启，训练中断",
                    finished_at=sql_func.now(),
                )
            )
            await session.commit()
            if result.rowcount:
                logger.info(f"Swept {result.rowcount} stale active task(s) → failed")
            if train_result.rowcount:
                logger.info(f"Swept {train_result.rowcount} stale training job(s) → failed")

        # Reap orphaned training process groups left by the previous generation.
        reaped = sum(1 for pid in stale_job_pids if _reap_orphan_training(pid))
        if reaped:
            logger.info(f"Reaped {reaped} orphan training process(es)")
    except Exception as exc:
        logger.warning(
            f"DB unavailable, history persistence disabled: {exc}\n"
            "Detection still works; install/start PostgreSQL to enable history."
        )

    # Preload the detection model (warm start). On success flag model_ready so
    # /readyz reports ready and the open-vocabulary detect path serves requests;
    # on failure leave it False so those fast-fail with 503 instead of retrying
    # an expensive load on every request (R-8).
    try:
        from app.services.detector import get_detector
        get_detector()
        app_state.model_ready = True
        logger.info("Detection model preloaded; model_ready=True")
    except Exception as exc:
        app_state.model_ready = False
        logger.warning(
            f"Model preload failed: {exc}\n"
            "Open-vocabulary detection will return 503 until the model loads; "
            "see /readyz."
        )

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    # Real graceful stop (R-2): ask in-flight detection tasks to package what
    # they have, reap training subprocess groups, then release the DB pool.
    logger.info("Shutting down: draining detection tasks & killing training procs")
    try:
        from app.services.task_manager import task_manager
        for st in task_manager.list_tasks():
            if st.status in _ACTIVE_TASK_STATUSES:
                task_manager.request_terminate(st.task_id)
        # Bounded wait for the background pipeline coroutines to wrap up.
        from app.api.detect import _background_tasks
        pending = [t for t in list(_background_tasks) if not t.done()]
        if pending:
            await asyncio.wait(pending, timeout=30)
    except Exception as exc:
        logger.warning(f"shutdown: detection-task drain skipped: {exc}")

    try:
        from app.services.training_manager import training_manager
        await training_manager.terminate_all(grace=10)
    except Exception as exc:
        logger.warning(f"shutdown: terminate_all failed: {exc}")

    try:
        from app.db.session import engine
        await engine.dispose()
    except Exception as exc:
        logger.warning(f"shutdown: engine.dispose failed: {exc}")

    logger.info("Shutdown complete.")


# ────────────────────────────────────────────────────────────────────────────
# Application factory
# ────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Video Object Detection Agent",
        description=(
            "Upload a video and describe what to detect in natural language. "
            "The system uses Grounding DINO + ByteTrack to detect and track "
            "objects frame-by-frame, streaming results in real time."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    # Allow all origins in development; restrict to your domain in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Error handling + request correlation (A-2 / O-5) ──────────────────
    # Registers the request-context middleware (request_id + X-Request-ID +
    # access log) and the unified {code,message,detail,request_id} handlers.
    register_error_handling(app)

    # ── API Routers ───────────────────────────────────────────────────────
    app.include_router(upload.router, prefix="/api", tags=["Upload"])
    app.include_router(detect.router, prefix="/api", tags=["Detection"])
    app.include_router(history.router, prefix="/api", tags=["History"])
    # YOLOE custom-training workflow (REQ1/REQ2/REQ3)
    app.include_router(categories.router, prefix="/api", tags=["YOLOE Categories"])
    app.include_router(dataset.router, prefix="/api", tags=["YOLOE Dataset"])
    app.include_router(training.router, prefix="/api", tags=["YOLOE Training"])
    app.include_router(yoloe_models.router, prefix="/api", tags=["YOLOE Models"])
    app.include_router(image_detect.router, prefix="/api", tags=["YOLOE Image Detection"])
    # LLM自动标注 (REQ4)
    app.include_router(llm_annotation.router, prefix="/api", tags=["LLM Annotation"])
    # 一站式自动标注：说目标 + 传图片/视频 → 自动建带时间戳数据集并标注
    app.include_router(auto_annotate.router, prefix="/api", tags=["Auto Annotation"])

    # ── Health checks ─────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health() -> dict:
        # Kept for backward compatibility (shallow probe). Prefer /livez + /readyz.
        return {
            "status": "ok",
            "model": settings.DETECTION_MODEL,
            "device": settings.DEVICE,
        }

    @app.get("/livez", tags=["Health"])
    async def livez() -> dict:
        """Liveness: the process is up and the event loop is responsive.
        Orchestrators probe this to decide whether to RESTART the container."""
        return {"status": "alive"}

    @app.get("/readyz", tags=["Health"])
    async def readyz():
        """Readiness: dependencies are usable (DB reachable, model loaded, GPU
        present when required). Orchestrators probe this to decide whether to
        send TRAFFIC. Returns 503 with a per-check breakdown when not ready (O-3).
        """
        from sqlalchemy import text

        checks: dict[str, str] = {}
        try:
            from app.db.session import AsyncSessionLocal
            async with AsyncSessionLocal() as s:
                await s.execute(text("SELECT 1"))
            checks["db"] = "ok"
        except Exception as exc:
            checks["db"] = f"fail: {type(exc).__name__}"

        checks["model"] = "ok" if app_state.model_ready else "loading"

        if settings.DEVICE.startswith("cuda"):
            try:
                import torch
                checks["gpu"] = "ok" if torch.cuda.is_available() else "fail"
            except Exception:
                checks["gpu"] = "fail"
        else:
            checks["gpu"] = "skipped"

        ok = (
            checks["db"] == "ok"
            and checks["model"] == "ok"
            and checks["gpu"] in ("ok", "skipped")
        )
        return JSONResponse(checks, status_code=200 if ok else 503)

    return app


app = create_app()


# ────────────────────────────────────────────────────────────────────────────
# Entry point (for direct `python -m app.main` or `python main.py`)
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,  # Keep single worker for GPU state sharing
        log_level="debug" if settings.DEBUG else "info",
        timeout_graceful_shutdown=30,  # give the shutdown hook time to drain (R-2)
    )
