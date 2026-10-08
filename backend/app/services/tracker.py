"""
app/services/tracker.py
------------------------
ByteTrack wrapper for multi-object tracking.

ByteTrack is a state-of-the-art, association-based tracker that does NOT
require re-detection every frame — it propagates tracks using a Kalman filter,
dramatically reducing GPU inference calls.

Installation:
  pip install git+https://github.com/ifzhang/ByteTrack.git
  OR (simpler) pip install bytetracker

We use the `bytetracker` pip package API here for portability.
If you have the full ByteTrack repo, replace with its BYTETracker class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.services.detector import RawDetection


# ────────────────────────────────────────────────────────────────────────────
# Tracked object output
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackedObject:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str


# ────────────────────────────────────────────────────────────────────────────
# ByteTrack wrapper
# ────────────────────────────────────────────────────────────────────────────

class ByteTracker:
    """
    Wraps ByteTrack to assign persistent IDs to detections across frames.

    Internally tries to import `bytetracker` (pip package).
    Falls back to a simple passthrough tracker (no ID persistence) if
    the library is unavailable — so the rest of the pipeline still works.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self._tracker = None
        self._loaded = False  # True once _load() has run (even if fallback)
        self._next_id = 1  # fallback ID counter

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            from bytetracker import BYTETracker

            self._tracker = BYTETracker(
                track_thresh=self.track_thresh,
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
                frame_rate=self.frame_rate,
            )
            logger.info("ByteTrack initialized (bytetracker package).")
            return
        except Exception:
            pass

        try:
            # Load ByteTrack tracker directly from local repo, bypassing
            # yolox/__init__.py (which pulls in thop/other heavy deps).
            import sys as _sys, os as _os, types as _types
            _here = _os.path.dirname(__file__)
            # ByteTrack lives at backend/ByteTrack (../../ from this file);
            # also accept a repo-root checkout (../../../). First match wins.
            _tracker_dir = ""
            for _rel in ("../../ByteTrack/yolox/tracker",
                         "../../../ByteTrack/yolox/tracker"):
                _cand = _os.path.abspath(_os.path.join(_here, _rel))
                if _os.path.isdir(_cand):
                    _tracker_dir = _cand
                    break
            if _tracker_dir:
                if "yolox" not in _sys.modules:
                    _yolox_pkg = _types.ModuleType("yolox")
                    _yolox_pkg.__path__ = [_os.path.dirname(_tracker_dir)]
                    _sys.modules["yolox"] = _yolox_pkg
                if "yolox.tracker" not in _sys.modules:
                    _tracker_pkg = _types.ModuleType("yolox.tracker")
                    _tracker_pkg.__path__ = [_tracker_dir]
                    _tracker_pkg.__package__ = "yolox.tracker"
                    _sys.modules["yolox.tracker"] = _tracker_pkg
                if _tracker_dir not in _sys.path:
                    _sys.path.insert(0, _tracker_dir)
            from yolox.tracker.byte_tracker import BYTETracker as YoloxBYTETracker

            class _Args2:
                track_thresh = self.track_thresh
                track_buffer = self.track_buffer
                match_thresh = self.match_thresh
                mot20 = False

            self._tracker = YoloxBYTETracker(args=_Args2(), frame_rate=self.frame_rate)
            logger.info("ByteTrack initialized (local yolox repo).")
            return
        except Exception:
            pass

        logger.warning(
            "ByteTrack not found. Falling back to passthrough tracker "
            "(no persistent IDs across frames). "
            "Install: pip install bytetracker"
        )

    def update(
        self,
        detections: List[RawDetection],
        image_shape: tuple,  # (H, W)
    ) -> List[TrackedObject]:
        """
        Update tracker with new detections and return tracked objects.

        :param detections: List of RawDetection from the detector.
        :param image_shape: (height, width) of the frame.
        :returns: List of TrackedObject with persistent track_ids.
        """
        self._load()

        if not detections:
            return []

        # Build [x1, y1, x2, y2, score] array
        dets_np = np.array(
            [[d.x1, d.y1, d.x2, d.y2, d.score] for d in detections],
            dtype=np.float32,
        )

        if self._tracker is None:
            # Passthrough: assign sequential IDs without tracking
            result = [
                TrackedObject(
                    track_id=self._next_id + i,
                    x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                    score=d.score, label=d.label,
                )
                for i, d in enumerate(detections)
            ]
            self._next_id += len(detections)
            return result

        h, w = image_shape
        try:
            online_targets = self._tracker.update(
                dets_np,
                [h, w],
                [h, w],
            )
        except Exception as exc:
            logger.warning(f"ByteTrack update failed ({exc}), using passthrough.")
            return [
                TrackedObject(
                    track_id=self._next_id + i,
                    x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                    score=d.score, label=d.label,
                )
                for i, d in enumerate(detections)
            ]

        # Match tracker output back to original detection labels by IoU
        tracked: List[TrackedObject] = []
        for t in online_targets:
            # ByteTrack may return tlbr or tlwh depending on version
            if hasattr(t, "tlbr"):
                x1, y1, x2, y2 = t.tlbr
            elif hasattr(t, "tlwh"):
                tx, ty, tw, th = t.tlwh
                x1, y1, x2, y2 = tx, ty, tx + tw, ty + th
            else:
                continue

            # Find closest original detection to get the label
            label = _match_label(
                x1, y1, x2, y2,
                detections,
            )
            tracked.append(
                TrackedObject(
                    track_id=int(t.track_id),
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    score=float(t.score) if hasattr(t, "score") else 1.0,
                    label=label,
                )
            )
        return tracked

    def reset(self) -> None:
        """Reset tracker state (call between tasks)."""
        self._tracker = None
        self._loaded = False
        self._next_id = 1


def _match_label(
    tx1: float, ty1: float, tx2: float, ty2: float,
    detections: List[RawDetection],
) -> str:
    """
    Find the detection whose bounding box best overlaps the tracker output
    and return its label.
    """
    best_iou = -1.0
    best_label = "object"
    for d in detections:
        iou = _iou(tx1, ty1, tx2, ty2, d.x1, d.y1, d.x2, d.y2)
        if iou > best_iou:
            best_iou = iou
            best_label = d.label
    return best_label


def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / (area_a + area_b - inter_area + 1e-6)


# ────────────────────────────────────────────────────────────────────────────
# Factory function
# ────────────────────────────────────────────────────────────────────────────

def create_tracker(fps: float = 30.0) -> ByteTracker:
    """Create a fresh ByteTracker instance (one per task)."""
    return ByteTracker(
        track_thresh=settings.TRACK_THRESH,
        track_buffer=settings.TRACK_BUFFER,
        match_thresh=settings.MATCH_THRESH,
        frame_rate=int(fps),
    )
