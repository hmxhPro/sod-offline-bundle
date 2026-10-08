"""
app/services/frame_save_gate.py
--------------------------------
Per-task decision of which annotated frames get written to disk.

Why this exists: with continuous video, the same physical object stays in
view across many detection keyframes, and the frame-level save modes
("keyframes_only" / "detections_only") write a near-identical JPEG for every
one of them. ByteTrack already tells us "same track_id = same object", so
"unique_targets" mode uses that to keep one snapshot per newly seen target
plus an optional periodic re-save, instead of one per keyframe.

The gate is deliberately dependency-free (no settings/cv2 imports) so it can
be unit-tested without pulling in the detector stack.
"""

from __future__ import annotations

from typing import Dict, Sequence


class FrameSaveGate:
    """
    Stateful save decision, one instance per detection task.

    Modes (mirrors settings.SAVE_FRAMES_MODE):
      - "all":             every processed frame.
      - "keyframes_only":  every detection keyframe, boxes or not.
      - "detections_only": detection keyframes with at least one box.
      - "unique_targets":  a frame is saved only when it shows a track_id
        never saved before (checked on every frame, so brand-new targets are
        captured as soon as ByteTrack activates them), or — on detection
        keyframes only — a known track whose last save is at least
        cooldown_sec old. Tracking-only frames never trigger re-saves: their
        boxes are Kalman-propagated copies of the previous keyframe.

    cooldown_sec semantics ("unique_targets" only): minimum seconds between
    saves of the SAME track. 0 disables throttling, which degenerates to
    "detections_only". A saved frame shows every box in it, so a save
    refreshes the timestamp of ALL tracks present, not just the one that
    triggered it.
    """

    def __init__(self, mode: str, cooldown_sec: float = 10.0) -> None:
        self.mode = mode
        self.cooldown_sec = cooldown_sec
        # track_id -> timestamp_seconds of the last frame saved showing it
        self._last_saved: Dict[int, float] = {}

    def should_save(
        self,
        is_detection_frame: bool,
        detections: Sequence,  # objects exposing .track_id (schemas.Detection)
        ts_seconds: float,
    ) -> bool:
        if self.mode == "all":
            return True
        if self.mode == "keyframes_only":
            return is_detection_frame
        if self.mode == "detections_only":
            return is_detection_frame and len(detections) > 0

        # ── "unique_targets" ─────────────────────────────────────────────
        if not detections:
            return False

        has_new_track = any(
            d.track_id not in self._last_saved for d in detections
        )
        cooldown_due = is_detection_frame and any(
            ts_seconds - self._last_saved[d.track_id] >= self.cooldown_sec
            for d in detections
            if d.track_id in self._last_saved
        )
        if not has_new_track and not cooldown_due:
            return False

        for d in detections:
            self._last_saved[d.track_id] = ts_seconds
        return True
