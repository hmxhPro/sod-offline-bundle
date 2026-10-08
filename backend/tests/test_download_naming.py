"""
tests/test_download_naming.py
-----------------------------
Naming of the results-ZIP download: trained-model tasks are named after the
model, natural-language tasks after the prompt/classes, both suffixed with
the task's creation timestamp. Pure logic, no app/DB imports.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.services.download_naming import (
    MODEL_PROMPT_PREFIX,
    build_zip_filename,
    content_disposition,
    sanitize_label,
)

_TS = datetime(2026, 6, 12, 9, 30, 5)


# ── Mode-specific labels ─────────────────────────────────────────────────────

def test_trained_model_prompt_uses_model_name():
    name = build_zip_filename(f"{MODEL_PROMPT_PREFIX}安全帽", created_at=_TS)
    assert name == "安全帽_20260612_093005.zip"


def test_natural_language_prompt_uses_classes():
    name = build_zip_filename("红色安全帽、白色卡车", created_at=_TS)
    assert name == "红色安全帽、白色卡车_20260612_093005.zip"


def test_prefix_only_stripped_at_start():
    # The marker mid-prompt is user text, not the trained-model contract.
    name = build_zip_filename(f"检测{MODEL_PROMPT_PREFIX}样式的物体", created_at=_TS)
    assert name.startswith("检测模型：样式的物体"[:2])
    assert MODEL_PROMPT_PREFIX not in name or not name.startswith("样式")


# ── Sanitization ─────────────────────────────────────────────────────────────

def test_forbidden_filename_chars_removed():
    assert sanitize_label('a/b\\c:d*e?f"g<h>i|j') == "abcdefghij"


def test_whitespace_collapsed_to_underscore():
    assert sanitize_label("  red   car \t truck ") == "red_car_truck"


def test_long_label_truncated():
    assert len(sanitize_label("x" * 200)) == 60


def test_empty_or_stripped_label_falls_back():
    assert sanitize_label("") == "detection"
    assert sanitize_label("???***") == "detection"
    assert build_zip_filename("", created_at=_TS) == "detection_20260612_093005.zip"


def test_no_path_traversal_in_label():
    name = build_zip_filename("../../etc/passwd", created_at=_TS)
    assert "/" not in name and ".." not in name


# ── Timestamp sources ────────────────────────────────────────────────────────

def test_naive_created_at_formatted_as_is():
    assert build_zip_filename("猫", created_at=_TS).endswith("_20260612_093005.zip")


def test_aware_created_at_converted_to_local():
    aware = _TS.replace(tzinfo=timezone.utc)
    expected = aware.astimezone().strftime("%Y%m%d_%H%M%S")
    assert build_zip_filename("猫", created_at=aware) == f"猫_{expected}.zip"


def test_fallback_ts_used_when_created_at_missing():
    fallback = datetime(2026, 1, 2, 3, 4, 5).timestamp()
    name = build_zip_filename("猫", created_at=None, fallback_ts=fallback)
    assert name == "猫_20260102_030405.zip"


def test_now_used_when_nothing_available():
    before = datetime.now() - timedelta(seconds=5)
    name = build_zip_filename("猫")
    stamp = datetime.strptime(name[len("猫_"):-len(".zip")], "%Y%m%d_%H%M%S")
    assert before <= stamp <= datetime.now() + timedelta(seconds=5)


# ── Content-Disposition header ───────────────────────────────────────────────
# Non-ASCII names must keep a plain filename= fallback: curl -OJ ignores
# filename* and would otherwise save the download as an extension-less UUID.

def test_ascii_filename_uses_plain_form_only():
    header = content_disposition("cat_20260612.zip", ascii_fallback="x.zip")
    assert header == 'attachment; filename="cat_20260612.zip"'
    assert "filename*" not in header


def test_non_ascii_filename_carries_both_forms():
    header = content_disposition(
        "安全帽_20260612_093005.zip", ascii_fallback="detection_12345678.zip"
    )
    assert 'filename="detection_12345678.zip"' in header
    assert "filename*=utf-8''%E5%AE%89%E5%85%A8%E5%B8%BD_20260612_093005.zip" in header


def test_header_value_is_latin1_safe():
    # Starlette encodes header values as latin-1; the raw Chinese must not leak.
    header = content_disposition("安全帽.zip", ascii_fallback="d.zip")
    header.encode("latin-1")
