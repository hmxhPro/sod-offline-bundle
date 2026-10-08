"""
app/services/download_naming.py
-------------------------------
Build the user-facing filename for the results-ZIP download.

Naming rule:
- Trained-model detection      → "<model name>_<timestamp>.zip"
- Natural-language detection   → "<classes / prompt>_<timestamp>.zip"

The detection mode is recovered from the persisted task prompt: trained-model
tasks store the display label "模型：<name>" (written by api/detect.py via
MODEL_PROMPT_PREFIX below), while natural-language tasks store the raw user
prompt — the classes being detected. TaskRecord has no dedicated mode/model
column and altering shipped tables needs Alembic (see app/db/models.py), so
the prefix is the contract.

Pure logic, no FastAPI/DB imports — unit-testable without the app stack.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional
from urllib.parse import quote

# Prefix of the display prompt persisted for trained-model tasks.
# api/detect.py both writes it (start_detection) and strips it here.
MODEL_PROMPT_PREFIX = "模型："

# Characters invalid in Windows filenames (superset of POSIX), plus controls.
_FORBIDDEN = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def sanitize_label(text: str, max_len: int = 60) -> str:
    """Reduce free text to a cross-platform-safe filename stem."""
    text = _FORBIDDEN.sub("", text)
    text = re.sub(r"\s+", "_", text.strip())
    text = text.strip("._")
    text = text[:max_len].rstrip("._")
    return text or "detection"


def build_zip_filename(
    prompt: str,
    created_at: Optional[datetime] = None,
    fallback_ts: Optional[float] = None,
) -> str:
    """
    Compose "<label>_<YYYYMMDD_HHMMSS>.zip" from the persisted task prompt.

    Timestamp preference: task created_at (converted to local time when
    tz-aware), else fallback_ts (e.g. the ZIP file's mtime, already local),
    else now.
    """
    label = (prompt or "").strip()
    if label.startswith(MODEL_PROMPT_PREFIX):
        label = label[len(MODEL_PROMPT_PREFIX):]
    label = sanitize_label(label)

    dt = created_at
    if dt is None and fallback_ts is not None:
        dt = datetime.fromtimestamp(fallback_ts)
    if dt is not None and dt.tzinfo is not None:
        dt = dt.astimezone()
    stamp = (dt or datetime.now()).strftime("%Y%m%d_%H%M%S")

    return f"{label}_{stamp}.zip"


def content_disposition(filename: str, ascii_fallback: str) -> str:
    """
    RFC 6266 attachment header carrying BOTH parameter forms for non-ASCII
    names. Starlette's FileResponse(filename=...) emits only filename* for
    non-ASCII, which curl -OJ (documented in docs/系统功能与技术.md) ignores —
    it would save the download as the URL's last path segment, an
    extension-less task UUID. Legacy clients read filename=, modern ones
    prefer filename*.
    """
    if filename.isascii():
        return f'attachment; filename="{filename}"'
    return (
        f'attachment; filename="{ascii_fallback}"; '
        f"filename*=utf-8''{quote(filename)}"
    )
