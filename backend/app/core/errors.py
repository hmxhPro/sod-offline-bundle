"""
app/core/errors.py
-------------------
Unified error handling (A-2) + request correlation (O-5).

- A request-context ASGI middleware assigns every request a `request_id`,
  exposes it on `request.state` + the logging ContextVar, injects it as the
  `X-Request-ID` response header, and logs one access line per request.
- Three exception handlers turn anything non-2xx into the SAME JSON envelope
  `{code, message, detail, request_id}` (see schemas.ErrorResponse):
    * HTTPException        → keep the localized message, map status → code
    * RequestValidationError (422) → friendly message + field list as detail
    * Exception (500)      → log the traceback, return a generic safe message
                             (raw error only when DEBUG)

Implemented as a pure ASGI middleware (not BaseHTTPMiddleware) so the SSE
detection stream (/api/stream/{id}) is never buffered.
"""

from __future__ import annotations

import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.logging import logger, request_id_var
from app.models.schemas import ErrorResponse


# HTTP status → short, machine-readable code (frontend may branch on it).
STATUS_CODE_MAP: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    503: "service_unavailable",
}

# Fallback Chinese message when an HTTPException carries a non-string detail.
_DEFAULT_MESSAGES: dict[str, str] = {
    "not_found": "请求的资源不存在。",
    "validation_error": "请求参数有误。",
    "service_unavailable": "服务暂不可用，请稍后重试。",
    "internal_error": "系统内部错误，请稍后重试。",
}


def _rid(request: Request) -> str:
    """The request_id stamped by the middleware (falls back to '-')."""
    return getattr(request.state, "request_id", "-")


def _json_error(status_code: int, body: ErrorResponse) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
        headers={"X-Request-ID": body.request_id},
    )


# ── Exception handlers ───────────────────────────────────────────────────────

async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Wrap HTTPException, preserving its (usually localized) detail message."""
    rid = _rid(request)
    code = STATUS_CODE_MAP.get(exc.status_code, "error")
    if isinstance(exc.detail, str):
        message, detail = exc.detail, None
    else:
        message, detail = _DEFAULT_MESSAGES.get(code, "请求出错。"), exc.detail
    return _json_error(
        exc.status_code,
        ErrorResponse(code=code, message=message, detail=detail, request_id=rid),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 → a readable message + the field-level errors as detail (fixes the
    frontend showing `[object Object]` for FastAPI's default array body)."""
    rid = _rid(request)
    return _json_error(
        422,
        ErrorResponse(
            code="validation_error",
            message="请求参数有误。",
            detail=jsonable_encoder(exc.errors()),
            request_id=rid,
        ),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all 500: log the full traceback (correlated by request_id) and
    return a safe generic message. The raw error is exposed only under DEBUG."""
    rid = _rid(request)
    logger.bind(request_id=rid).opt(exception=exc).error(
        f"Unhandled error on {request.method} {request.url.path}"
    )
    return _json_error(
        500,
        ErrorResponse(
            code="internal_error",
            message="系统内部错误，请稍后重试。",
            detail=(f"{type(exc).__name__}: {exc}" if settings.DEBUG else None),
            request_id=rid,
        ),
    )


# ── Request-context middleware (pure ASGI; SSE-safe) ─────────────────────────

class RequestContextMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Reuse an inbound X-Request-ID (e.g. from a reverse proxy) or mint one.
        rid = uuid4().hex[:12]
        for k, v in scope.get("headers") or []:
            if k == b"x-request-id" and v:
                rid = v.decode("latin-1")
                break

        # Expose to exception handlers via request.state and to loguru via the
        # ContextVar (so every log line inside this request carries the id).
        scope.setdefault("state", {})["request_id"] = rid
        token = request_id_var.set(rid)
        start = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = list(message.get("headers") or [])
                if not any(k == b"x-request-id" for k, _ in headers):
                    headers.append((b"x-request-id", rid.encode("latin-1")))
                    message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            # Unhandled — ServerErrorMiddleware (outermost) will format the 500
            # via unhandled_exception_handler, which logs the traceback. Here we
            # only record timing, then re-raise.
            dur_ms = (time.perf_counter() - start) * 1000
            logger.bind(request_id=rid).warning(
                f"{scope['method']} {scope['path']} -> 500 ({dur_ms:.0f}ms) [unhandled]"
            )
            raise
        finally:
            request_id_var.reset(token)

        dur_ms = (time.perf_counter() - start) * 1000
        line = f"{scope['method']} {scope['path']} -> {status['code']} ({dur_ms:.0f}ms)"
        if status["code"] >= 500:
            logger.bind(request_id=rid).error(line)
        elif status["code"] >= 400:
            logger.bind(request_id=rid).warning(line)
        else:
            logger.bind(request_id=rid).debug(line)


def register_error_handling(app: FastAPI) -> None:
    """Install the request-context middleware + unified exception handlers."""
    app.add_middleware(RequestContextMiddleware)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


__all__ = ["register_error_handling", "RequestContextMiddleware"]
