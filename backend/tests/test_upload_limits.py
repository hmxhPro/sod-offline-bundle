"""
tests/test_upload_limits.py
---------------------------
Covers P-1: the video upload aborts with 413 once the byte cap is exceeded and
the partial file is cleaned up. Driven through the real ASGI app (no network).
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import app


async def test_upload_rejects_oversize(monkeypatch):
    # Shrink the cap so the test doesn't need a 2 GB file; neutralize the disk
    # water-mark so we exercise the size path specifically.
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 1024)
    monkeypatch.setattr(settings, "MIN_FREE_DISK_BYTES", 0)

    before = set(settings.UPLOAD_DIR.glob("*"))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/upload",
            files={"file": ("clip.mp4", b"x" * 5000, "video/mp4")},
        )
    assert resp.status_code == 413
    # the partial file must have been unlinked on abort
    after = set(settings.UPLOAD_DIR.glob("*"))
    assert after == before


async def test_upload_rejects_bad_extension():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
    assert resp.status_code == 415
