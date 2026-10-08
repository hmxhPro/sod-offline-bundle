"""
app/core/logging.py
-------------------
Loguru-based logging configuration.

Three responsibilities:
  1. Console + daily-rotating file sinks (the file sink is the post-mortem
     record — see logs/app_<date>.log).
  2. Route the standard library `logging` (uvicorn / fastapi / sqlalchemy and
     any third-party lib, plus unhandled-exception tracebacks surfaced by the
     server) INTO loguru, so everything lands in the same file (O-5).
  3. Carry a per-request `request_id` on every log line so a user-facing error
     can be traced back to its server-side log entries (O-5). The id is stored
     in a ContextVar set by the request-context middleware (see app/core/errors.py).
"""

import logging
import sys
from contextvars import ContextVar

from loguru import logger

from app.core.config import settings


# Per-request correlation id. Defaults to "-" outside of any request (startup,
# background tasks). The middleware in app/core/errors.py sets/reset this; the
# patcher below copies it onto every loguru record's `extra`.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class InterceptHandler(logging.Handler):
    """Redirect standard-library logging records into loguru.

    Canonical loguru recipe: find the loguru level matching the stdlib level,
    walk up the stack past the logging module so the logged call-site is the
    real caller, and forward the record (with its exc_info) to loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _inject_request_id(record) -> None:
    """loguru patcher: stamp the current request_id onto every record.

    Uses setdefault so an explicit `logger.bind(request_id=...)` (e.g. in the
    global exception handler, which runs after the middleware has reset the
    ContextVar) is preserved.
    """
    record["extra"].setdefault("request_id", request_id_var.get())


def setup_logging(debug: bool = False) -> None:
    """Configure loguru and route stdlib logging into it."""
    level = "DEBUG" if debug else settings.LOG_LEVEL

    logger.remove()  # Remove default handler
    logger.configure(patcher=_inject_request_id)

    log_dir = settings.LOG_DIR.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Console handler with colored output
    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<magenta>{extra[request_id]}</magenta> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler (rotates daily, keeps 7 days). Always DEBUG for detail.
    logger.add(
        str(log_dir / "app_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        rotation="00:00",
        retention="7 days",
        compression="zip",
        enqueue=True,            # safe across threads / the training subprocess
        backtrace=True,          # full traceback frames on exceptions
        diagnose=debug,          # variable values only in debug (avoid leaking in prod)
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {extra[request_id]} | {message}"
        ),
    )

    # ── Route stdlib logging → loguru ────────────────────────────────────────
    # Root catches everything by default; then point the noisy named loggers at
    # our handler explicitly and stop them propagating (uvicorn installs its own
    # handlers, which we override here so their output also reaches the file).
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "sqlalchemy.engine",
        "asyncio",
    ):
        std_logger = logging.getLogger(name)
        std_logger.handlers = [InterceptHandler()]
        std_logger.propagate = False


__all__ = ["logger", "setup_logging", "request_id_var", "InterceptHandler"]
