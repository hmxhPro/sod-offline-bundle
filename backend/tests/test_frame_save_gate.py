"""
tests/test_frame_save_gate.py
------------------------------
FrameSaveGate: which annotated frames reach disk under each SAVE_FRAMES_MODE,
in particular the ByteTrack-ID-based dedup of "unique_targets".
Pure logic, no detector/cv2 imports.
"""

from __future__ import annotations

import types

from app.services.frame_save_gate import FrameSaveGate


def _dets(*track_ids):
    return [types.SimpleNamespace(track_id=tid) for tid in track_ids]


# ── Legacy modes keep their exact semantics ──────────────────────────────────

def test_all_mode_saves_every_frame():
    gate = FrameSaveGate(mode="all")
    assert gate.should_save(is_detection_frame=False, detections=[], ts_seconds=0.0)
    assert gate.should_save(is_detection_frame=True, detections=_dets(1), ts_seconds=1.0)


def test_keyframes_only_saves_detection_frames_with_or_without_boxes():
    gate = FrameSaveGate(mode="keyframes_only")
    assert gate.should_save(is_detection_frame=True, detections=[], ts_seconds=0.0)
    assert not gate.should_save(is_detection_frame=False, detections=_dets(1), ts_seconds=0.5)


def test_detections_only_requires_keyframe_and_boxes():
    gate = FrameSaveGate(mode="detections_only")
    assert not gate.should_save(is_detection_frame=True, detections=[], ts_seconds=0.0)
    assert not gate.should_save(is_detection_frame=False, detections=_dets(1), ts_seconds=0.5)
    assert gate.should_save(is_detection_frame=True, detections=_dets(1), ts_seconds=1.0)
    # No dedup: the same track saves on every keyframe
    assert gate.should_save(is_detection_frame=True, detections=_dets(1), ts_seconds=1.5)


# ── unique_targets: dedup by track_id ────────────────────────────────────────

def test_unique_first_appearance_saves_even_on_tracking_frame():
    # ByteTrack activates new tracks one frame after the keyframe, so the
    # first-appearance trigger must work on non-detection frames too.
    gate = FrameSaveGate(mode="unique_targets", cooldown_sec=10.0)
    assert gate.should_save(is_detection_frame=False, detections=_dets(1), ts_seconds=0.1)


def test_unique_same_track_suppressed_within_cooldown():
    gate = FrameSaveGate(mode="unique_targets", cooldown_sec=10.0)
    assert gate.should_save(is_detection_frame=True, detections=_dets(1), ts_seconds=0.0)
    # Same target on every subsequent keyframe: all suppressed for 10 s
    for ts in (1.0, 3.0, 5.0, 9.9):
        assert not gate.should_save(is_detection_frame=True, detections=_dets(1), ts_seconds=ts)


def test_unique_resaves_after_cooldown_on_keyframe_only():
    gate = FrameSaveGate(mode="unique_targets", cooldown_sec=10.0)
    assert gate.should_save(is_detection_frame=True, detections=_dets(1), ts_seconds=0.0)
    # Cooldown elapsed, but tracking-only frames carry stale propagated boxes
    assert not gate.should_save(is_detection_frame=False, detections=_dets(1), ts_seconds=11.0)
    assert gate.should_save(is_detection_frame=True, detections=_dets(1), ts_seconds=12.0)


def test_unique_new_track_in_existing_scene_saves_and_refreshes_all():
    gate = FrameSaveGate(mode="unique_targets", cooldown_sec=10.0)
    assert gate.should_save(is_detection_frame=True, detections=_dets(1), ts_seconds=0.0)
    # Track 2 enters at t=5 → save; the frame also shows track 1, so both
    # timestamps refresh to 5.0
    assert gate.should_save(is_detection_frame=True, detections=_dets(1, 2), ts_seconds=5.0)
    # t=12: 12-5=7 < 10 for both → suppressed
    assert not gate.should_save(is_detection_frame=True, detections=_dets(1, 2), ts_seconds=12.0)
    # t=16: 16-5=11 ≥ 10 → re-save
    assert gate.should_save(is_detection_frame=True, detections=_dets(1, 2), ts_seconds=16.0)


def test_unique_empty_detections_never_save():
    gate = FrameSaveGate(mode="unique_targets", cooldown_sec=10.0)
    assert not gate.should_save(is_detection_frame=True, detections=[], ts_seconds=0.0)
    assert not gate.should_save(is_detection_frame=False, detections=[], ts_seconds=1.0)


def test_unique_zero_cooldown_degenerates_to_detections_only():
    gate = FrameSaveGate(mode="unique_targets", cooldown_sec=0.0)
    assert gate.should_save(is_detection_frame=True, detections=_dets(1), ts_seconds=0.0)
    # Every keyframe with boxes re-saves...
    assert gate.should_save(is_detection_frame=True, detections=_dets(1), ts_seconds=0.5)
    # ...but tracking-only frames with already-seen tracks still don't
    assert not gate.should_save(is_detection_frame=False, detections=_dets(1), ts_seconds=0.6)
