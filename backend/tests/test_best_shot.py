"""
tests/test_best_shot.py
------------------------
Best-shot snapshot selection:
  - BestShotSelector  — pure per-track "keep the best" bookkeeping + the
    ref-counted JPEG cache that bounds memory to one image per live track.
  - BestShotScorer    — the cv2-backed quality score (sharpness / area /
    confidence / edge penalty), exercised on tiny synthetic frames.
"""

from __future__ import annotations

import numpy as np

from app.services.best_shot import BestShotScorer, BestShotSelector


# ── BestShotSelector: keep the highest score per track ───────────────────────

def test_first_candidate_is_retained():
    sel = BestShotSelector()
    assert sel.consider(track_id=1, score=0.5, frame_idx=10, encode=lambda: b"f10")
    assert sel.winners() == {10: b"f10"}
    assert sel.track_count == 1


def test_higher_score_replaces_lower_is_ignored():
    sel = BestShotSelector()
    sel.consider(track_id=1, score=0.5, frame_idx=10, encode=lambda: b"f10")
    # Lower score: rejected, best frame unchanged.
    assert not sel.consider(track_id=1, score=0.4, frame_idx=11, encode=lambda: b"f11")
    assert sel.winners() == {10: b"f10"}
    # Higher score: replaces, and the old frame is evicted (no refs left).
    assert sel.consider(track_id=1, score=0.6, frame_idx=12, encode=lambda: b"f12")
    assert sel.winners() == {12: b"f12"}
    assert sel.track_count == 1


def test_equal_score_keeps_earlier_frame():
    sel = BestShotSelector()
    sel.consider(track_id=1, score=0.5, frame_idx=10, encode=lambda: b"f10")
    assert not sel.consider(track_id=1, score=0.5, frame_idx=20, encode=lambda: b"f20")
    assert sel.winners() == {10: b"f10"}


def test_winners_dedup_one_frame_shared_by_two_tracks():
    sel = BestShotSelector()
    sel.consider(track_id=1, score=0.5, frame_idx=10, encode=lambda: b"f10")
    # Track 2 peaks on the SAME frame → one file, two tracks.
    sel.consider(track_id=2, score=0.7, frame_idx=10, encode=lambda: b"f10")
    assert sel.winners() == {10: b"f10"}
    assert sel.track_count == 2


def test_shared_frame_survives_until_last_track_moves_off():
    sel = BestShotSelector()
    sel.consider(track_id=1, score=0.5, frame_idx=10, encode=lambda: b"f10")
    sel.consider(track_id=2, score=0.5, frame_idx=10, encode=lambda: b"f10")
    # Track 1 finds a better frame; frame 10 is still track 2's best.
    sel.consider(track_id=1, score=0.9, frame_idx=20, encode=lambda: b"f20")
    assert sel.winners() == {10: b"f10", 20: b"f20"}
    # Track 2 also moves on → frame 10 is now unreferenced and dropped.
    sel.consider(track_id=2, score=0.9, frame_idx=30, encode=lambda: b"f30")
    assert sel.winners() == {20: b"f20", 30: b"f30"}


def test_min_score_filters_poor_shots():
    sel = BestShotSelector(min_score=0.3)
    assert not sel.consider(track_id=1, score=0.2, frame_idx=10, encode=lambda: b"f10")
    assert sel.winners() == {}
    assert sel.track_count == 0
    assert sel.consider(track_id=1, score=0.35, frame_idx=11, encode=lambda: b"f11")
    assert sel.winners() == {11: b"f11"}


def test_encode_is_lazy_and_called_once_per_retained_frame():
    calls = {"n": 0}

    def encode():
        calls["n"] += 1
        return b"frame"

    sel = BestShotSelector()
    # Two tracks pick the same frame: encoded exactly once.
    sel.consider(track_id=1, score=0.5, frame_idx=10, encode=encode)
    sel.consider(track_id=2, score=0.6, frame_idx=10, encode=encode)
    assert calls["n"] == 1
    # A rejected (lower-score) candidate never encodes.
    sel.consider(track_id=1, score=0.1, frame_idx=11, encode=encode)
    assert calls["n"] == 1


# ── BestShotScorer: quality components ───────────────────────────────────────

def _noisy(h: int, w: int, seed: int = 0) -> np.ndarray:
    """A high-frequency (sharp) BGR frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _flat(h: int, w: int, value: int = 128) -> np.ndarray:
    """A uniform (blurry / no-edge) BGR frame."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_sharper_crop_scores_higher_than_flat():
    scorer = BestShotScorer()
    box = (10, 10, 90, 90)
    sharp = scorer.score(_noisy(100, 100), *box, confidence=0.8)
    flat = scorer.score(_flat(100, 100), *box, confidence=0.8)
    assert sharp > flat


def test_larger_box_scores_higher_via_area_term():
    # Confidence-only weighting removed so the area term dominates; same
    # (flat) image so sharpness is equal for both.
    scorer = BestShotScorer(w_sharpness=0.0, w_area=1.0, w_confidence=0.0,
                            area_ref=0.5, edge_margin_px=0)
    frame = _flat(200, 200)
    small = scorer.score(frame, 80, 80, 100, 100, confidence=0.5)   # 20x20
    large = scorer.score(frame, 40, 40, 160, 160, confidence=0.5)   # 120x120
    assert large > small


def test_edge_touching_box_is_penalised():
    scorer = BestShotScorer(edge_margin_px=4, edge_penalty=0.5)
    frame = _noisy(200, 200)
    # Centered box vs one flush against the left/top edge, same size.
    centered = scorer.score(frame, 60, 60, 140, 140, confidence=0.9)
    at_edge = scorer.score(frame, 0, 0, 80, 80, confidence=0.9)
    assert at_edge < centered


def test_higher_confidence_scores_higher():
    scorer = BestShotScorer()
    frame = _noisy(100, 100)
    box = (20, 20, 80, 80)
    lo = scorer.score(frame, *box, confidence=0.2)
    hi = scorer.score(frame, *box, confidence=0.9)
    assert hi > lo


def test_degenerate_box_scores_zero():
    scorer = BestShotScorer()
    frame = _noisy(100, 100)
    # Zero-area and fully out-of-bounds boxes yield no usable crop.
    assert scorer.score(frame, 50, 50, 50, 50, confidence=0.9) == 0.0
    assert scorer.score(frame, 200, 200, 260, 260, confidence=0.9) == 0.0


def test_score_is_bounded_unit_interval():
    scorer = BestShotScorer()
    frame = _noisy(100, 100)
    s = scorer.score(frame, 10, 10, 90, 90, confidence=1.0)
    assert 0.0 <= s <= 1.0
