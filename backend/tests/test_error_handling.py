"""
tests/test_error_handling.py
----------------------------
Covers A-2 / O-5: every non-2xx response is the unified
{code, message, detail, request_id} envelope, an X-Request-ID header is always
present, inbound ids are propagated, and unhandled exceptions become a safe
structured 500 (not an opaque "Internal Server Error").

Driven through an isolated FastAPI app wired with register_error_handling — no
model/DB load, deterministic raising routes.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

from app.core.errors import register_error_handling


class _Body(BaseModel):
    n: int


def _build_app() -> FastAPI:
    app = FastAPI()
    register_error_handling(app)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("kaboom: simulated unexpected failure")

    @app.get("/missing")
    async def missing():
        raise HTTPException(status_code=404, detail="模型不存在。")

    @app.post("/needs")
    async def needs(body: _Body):
        return {"ok": body.n}

    return app


def _client(app: FastAPI) -> AsyncClient:
    # raise_app_exceptions=False: an unhandled error must come back as our
    # structured 500 response, never propagate into the test.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_unhandled_exception_becomes_structured_500():
    async with _client(_build_app()) as ac:
        resp = await ac.get("/boom")
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "internal_error"
    assert body["message"]                      # friendly, non-empty
    assert body["request_id"]
    # header is present and matches the body id (used for log correlation)
    assert resp.headers.get("x-request-id") == body["request_id"]
    # raw internals are not leaked by default (DEBUG off)
    assert body["detail"] is None


async def test_http_exception_is_wrapped_with_code():
    async with _client(_build_app()) as ac:
        resp = await ac.get("/missing")
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "not_found"
    assert body["message"] == "模型不存在。"     # localized detail preserved
    assert body["request_id"]
    assert resp.headers.get("x-request-id") == body["request_id"]


async def test_validation_error_is_friendly_not_object_object():
    async with _client(_build_app()) as ac:
        resp = await ac.post("/needs", json={})   # missing required `n`
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "validation_error"
    assert body["message"] == "请求参数有误。"
    assert isinstance(body["detail"], list) and body["detail"]   # field list, JSON-safe
    assert body["request_id"]


async def test_inbound_request_id_is_propagated():
    async with _client(_build_app()) as ac:
        resp = await ac.get("/missing", headers={"X-Request-ID": "trace-abc-123"})
    assert resp.headers.get("x-request-id") == "trace-abc-123"
    assert resp.json()["request_id"] == "trace-abc-123"


async def test_request_id_header_on_success():
    async with _client(_build_app()) as ac:
        resp = await ac.post("/needs", json={"n": 7})
    assert resp.status_code == 200
    assert resp.json() == {"ok": 7}
    assert resp.headers.get("x-request-id")        # injected even on 2xx
