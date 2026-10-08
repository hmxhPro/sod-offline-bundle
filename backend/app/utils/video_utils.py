"""
app/utils/video_utils.py
-------------------------
Utility functions for video file handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import cv2


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """
    Read video metadata without decoding all frames.

    :returns: Dict with keys: total_frames, fps, width, height, duration_seconds
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0
    finally:
        cap.release()

    return {
        "total_frames": max(total_frames, 0),
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": round(duration, 3),
    }


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.mmm string.

    Examples:
        5.0   → "00:00:05.000"
        65.5  → "00:01:05.500"
        3661.123 → "01:01:01.123"
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
