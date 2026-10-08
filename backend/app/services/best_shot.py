"""
app/services/best_shot.py
--------------------------
"Best-shot" snapshot selection for continuous video (安防行业的"抓拍优选").

Why this exists: the "unique_targets" save mode keeps a target's FIRST frame
(plus periodic re-saves). But the first frame a target is seen in is often the
worst one — small, far away, motion-blurred, or half-entered at the frame edge.
Best-shot instead scores EVERY frame a target appears in across its whole track
lifetime and, when the video ends, writes only the single highest-scoring frame
per target. One snapshot per object, and it is the clearest/largest/most
confident moment, at the cost of the result only being known once the track is
done (so writes are deferred to packaging time rather than streamed inline).

Two pieces, split so the bookkeeping stays unit-testable without a GPU/cv2:

  - BestShotScorer  — turns (frame, bbox, confidence) into a single quality
    score in [0, 1]. Uses cv2 (Laplacian sharpness), so its tests use tiny
    synthetic images.
  - BestShotSelector — pure bookkeeping: per-track running best, plus a
    ref-counted cache of the encoded JPEG bytes for each currently-winning
    frame so memory stays bounded to one image per live track (a frame chosen
    by several tracks is stored once). No cv2/settings imports.

The selector is fed a precomputed score and a lazy `encode` thunk; it only
materialises JPEG bytes when a frame actually becomes some track's best, and
drops them as soon as no track points at that frame anymore.
"""

from __future__ import annotations

from typing import Callable, Dict

import cv2
import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Scoring (cv2-backed)
# ────────────────────────────────────────────────────────────────────────────

class BestShotScorer:
    """
    Score how good a single detection crop is as a snapshot of its target.

    score = (w_sharp·sharpness + w_area·area + w_conf·confidence) · edge_factor

    Components, each normalised to [0, 1]:
      - sharpness:  variance of the Laplacian of the GRAYSCALE crop, taken from
        the RAW (un-annotated) frame so the drawn box edges don't inflate it.
        Saturates at `sharpness_ref` (a typical "in-focus" variance).
      - area:       bbox area as a fraction of the frame, saturating at
        `area_ref`. More pixels on target ⇒ more usable detail.
      - confidence: the detector/track score, clamped to [0, 1].

    edge_factor multiplies the whole score by `edge_penalty` (<1) when the box
    touches the frame border within `edge_margin_px` — such targets are usually
    truncated (entering/leaving frame) and make poor snapshots.

    Weights need not sum to 1; they are normalised internally.
    """

    def __init__(
        self,
        w_sharpness: float = 0.4,
        w_area: float = 0.3,
        w_confidence: float = 0.3,
        sharpness_ref: float = 200.0,
        area_ref: float = 0.1,
        edge_margin_px: int = 4,
        edge_penalty: float = 0.6,
    ) -> None:
        total = w_sharpness + w_area + w_confidence
        if total <= 0:
            raise ValueError("best-shot weights must sum to a positive value")
        self.w_sharpness = w_sharpness / total
        self.w_area = w_area / total
        self.w_confidence = w_confidence / total
        self.sharpness_ref = max(sharpness_ref, 1e-6)
        self.area_ref = min(max(area_ref, 1e-6), 1.0)
        self.edge_margin_px = max(edge_margin_px, 0)
        self.edge_penalty = min(max(edge_penalty, 0.0), 1.0)

    def score(
        self,
        frame: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        confidence: float,
    ) -> float:
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        if frame_area <= 0:
            return 0.0

        # Clamp box to frame bounds before cropping. Clamping to [0, w]/[0, h]
        # (not w-1/h-1) lets a fully-off-screen box collapse to cx2 <= cx1 and
        # score 0, rather than leaving a fake 1-px sliver.
        cx1 = int(max(0, min(x1, w)))
        cy1 = int(max(0, min(y1, h)))
        cx2 = int(max(0, min(x2, w)))
        cy2 = int(max(0, min(y2, h)))
        if cx2 <= cx1 or cy2 <= cy1:
            return 0.0

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return 0.0

        # Sharpness: Laplacian variance on the grayscale crop.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharp_norm = min(sharpness / self.sharpness_ref, 1.0)

        # Area: fraction of the frame the VISIBLE (clamped) box covers, so a
        # target truncated by the frame edge counts only its on-screen pixels.
        box_area = float((cx2 - cx1) * (cy2 - cy1))
        area_norm = min((box_area / frame_area) / self.area_ref, 1.0)

        conf_norm = min(max(confidence, 0.0), 1.0)

        raw = (
            self.w_sharpness * sharp_norm
            + self.w_area * area_norm
            + self.w_confidence * conf_norm
        )

        # Edge / truncation penalty.
        m = self.edge_margin_px
        touches_edge = (
            x1 <= m or y1 <= m or x2 >= (w - m) or y2 >= (h - m)
        )
        if touches_edge:
            raw *= self.edge_penalty

        return float(min(max(raw, 0.0), 1.0))


# ────────────────────────────────────────────────────────────────────────────
# Selection (pure bookkeeping)
# ────────────────────────────────────────────────────────────────────────────

class BestShotSelector:
    """
    Keep the single best-scoring frame per track across a whole video.

    Stateful, one instance per detection task. Call `consider(...)` for every
    detection on every processed frame; at the end call `winners()` to get the
    frame_idx → JPEG-bytes map to write to disk (deduped: a frame that is the
    best shot of several tracks appears once).

    Memory is bounded to one encoded frame per live track: the JPEG bytes of a
    frame are kept only while at least one track still points at it (ref count),
    and are encoded lazily — `encode` is invoked only when a frame first becomes
    some track's best.
    """

    def __init__(self, min_score: float = 0.0) -> None:
        self.min_score = min_score
        self._best_score: Dict[int, float] = {}   # track_id -> best score so far
        self._best_frame: Dict[int, int] = {}      # track_id -> winning frame_idx
        self._refs: Dict[int, int] = {}            # frame_idx -> # tracks pointing here
        self._frames: Dict[int, bytes] = {}        # frame_idx -> encoded JPEG bytes

    def consider(
        self,
        track_id: int,
        score: float,
        frame_idx: int,
        encode: Callable[[], bytes],
    ) -> bool:
        """
        Offer one detection as a snapshot candidate for `track_id`.

        Returns True if it became this track's new best (and thus this frame is
        now retained). `encode()` is called at most once per retained frame.
        """
        if score < self.min_score:
            return False

        prev = self._best_score.get(track_id)
        if prev is not None and score <= prev:
            return False

        # New best for this track: release the frame it used to point at...
        old_frame = self._best_frame.get(track_id)
        if old_frame is not None:
            self._release(old_frame)

        # ...and retain the new one (encoding it once if not already cached).
        self._best_score[track_id] = score
        self._best_frame[track_id] = frame_idx
        self._refs[frame_idx] = self._refs.get(frame_idx, 0) + 1
        if frame_idx not in self._frames:
            self._frames[frame_idx] = encode()
        return True

    def _release(self, frame_idx: int) -> None:
        remaining = self._refs.get(frame_idx, 0) - 1
        if remaining <= 0:
            self._refs.pop(frame_idx, None)
            self._frames.pop(frame_idx, None)
        else:
            self._refs[frame_idx] = remaining

    def winners(self) -> Dict[int, bytes]:
        """frame_idx → JPEG bytes for every frame chosen as some track's best."""
        return dict(self._frames)

    @property
    def track_count(self) -> int:
        """Number of distinct tracks that have a retained best shot."""
        return len(self._best_frame)
