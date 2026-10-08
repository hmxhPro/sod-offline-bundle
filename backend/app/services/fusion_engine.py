"""
app/services/fusion_engine.py
-------------------------------
Track state machine and score fusion for VLM-enhanced detection.

Each track maintains a state: tentative → confirmed / rejected / lost.
The fusion engine decides when to trigger VLM verification and
combines DINO detection scores with VLM semantic scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.services.tracker import TrackedObject
from app.services.vlm_service import VLMResult


@dataclass
class TrackState:
    track_id: int
    label: str
    status: str = "tentative"  # tentative | confirmed | rejected | lost
    dino_scores: List[float] = field(default_factory=list)
    vlm_scores: List[float] = field(default_factory=list)
    vlm_verified: bool = False
    vlm_is_target: Optional[bool] = None
    last_vlm_time: float = 0.0
    last_seen_time: float = 0.0
    first_seen_time: float = 0.0
    hit_count: int = 0
    miss_count: int = 0

    @property
    def dino_score_avg(self) -> float:
        return float(np.mean(self.dino_scores[-10:])) if self.dino_scores else 0.0

    @property
    def vlm_score_avg(self) -> float:
        return float(np.mean(self.vlm_scores[-5:])) if self.vlm_scores else 0.0

    @property
    def final_score(self) -> float:
        if self.vlm_scores:
            return (
                settings.VLM_WEIGHT * self.vlm_score_avg
                + settings.DINO_WEIGHT * self.dino_score_avg
            )
        return self.dino_score_avg


class FusionEngine:
    def __init__(self) -> None:
        self._tracks: Dict[int, TrackState] = {}

    def get_track(self, track_id: int) -> Optional[TrackState]:
        return self._tracks.get(track_id)

    def update_tracks(
        self,
        tracked_objects: List[TrackedObject],
        current_time: float,
    ) -> None:
        seen_ids = set()
        for obj in tracked_objects:
            seen_ids.add(obj.track_id)
            ts = self._tracks.get(obj.track_id)
            if ts is None:
                ts = TrackState(
                    track_id=obj.track_id,
                    label=obj.label,
                    first_seen_time=current_time,
                )
                self._tracks[obj.track_id] = ts
            ts.dino_scores.append(obj.score)
            ts.last_seen_time = current_time
            ts.hit_count += 1
            if ts.status == "lost":
                ts.status = "tentative"

        for tid, ts in self._tracks.items():
            if tid not in seen_ids and ts.status not in ("rejected", "lost"):
                ts.miss_count += 1
                if ts.miss_count > 10:
                    ts.status = "lost"

    def should_verify(self, track_id: int, current_time: float) -> bool:
        ts = self._tracks.get(track_id)
        if ts is None:
            return False
        if ts.status == "rejected":
            return False

        # New track: always verify
        is_new = ts.hit_count <= 1
        if is_new:
            return True

        # Low confidence candidate: verify if not yet verified
        low_confidence = ts.dino_score_avg < settings.DINO_CANDIDATE_THRESHOLD
        if low_confidence and not ts.vlm_verified:
            return True

        # Mid confidence candidate: verify if not yet verified
        mid_confidence = ts.dino_score_avg < settings.DINO_DIRECT_CONFIRM_THRESHOLD
        if mid_confidence and not ts.vlm_verified:
            return True

        # Periodic re-verification based on status
        if ts.status == "confirmed":
            interval = settings.VLM_INTERVAL_CONFIRMED
        else:
            interval = settings.VLM_INTERVAL_TENTATIVE

        time_since_last = current_time - ts.last_vlm_time
        return time_since_last >= interval

    def apply_vlm_result(
        self,
        track_id: int,
        vlm_result: VLMResult,
        current_time: float,
    ) -> None:
        ts = self._tracks.get(track_id)
        if ts is None:
            return

        ts.vlm_verified = True
        ts.last_vlm_time = current_time
        ts.vlm_scores.append(vlm_result.confidence)
        ts.vlm_is_target = vlm_result.is_target

        # Multi-dimensional judgment logic
        dino_avg = ts.dino_score_avg
        vlm_conf = vlm_result.confidence
        hit_count = ts.hit_count

        # Priority 1: VLM high confidence confirmation
        if vlm_result.is_target and vlm_conf >= settings.VLM_CONFIRM_THRESHOLD:
            ts.status = "confirmed"
        # Priority 2: VLM rejection
        elif not vlm_result.is_target and vlm_conf >= 0.60:
            ts.status = "rejected"
        # Priority 3: High DINO score + multiple hits
        elif dino_avg >= settings.DINO_DIRECT_CONFIRM_THRESHOLD and hit_count >= settings.MIN_HIT_COUNT_HIGH_CONF:
            if vlm_result.is_target:
                ts.status = "confirmed"
            else:
                ts.status = "candidate"
        # Priority 4: Mid DINO score + more hits
        elif dino_avg >= settings.DINO_CANDIDATE_THRESHOLD and hit_count >= settings.MIN_HIT_COUNT_MID_CONF:
            if vlm_result.is_target:
                ts.status = "candidate"
            else:
                ts.status = "rejected"
        # Priority 5: Low score or insufficient hits
        elif dino_avg < settings.DINO_CANDIDATE_THRESHOLD:
            if vlm_result.is_target and vlm_conf >= 0.50:
                ts.status = "candidate"
            else:
                ts.status = "rejected"
        else:
            ts.status = "candidate"

        final = ts.final_score
        logger.debug(
            f"Track {track_id} VLM: is_target={vlm_result.is_target} "
            f"vlm_conf={vlm_conf:.2f} dino_avg={dino_avg:.2f} "
            f"hit_count={hit_count} final={final:.2f} "
            f"status={ts.status} reason={vlm_result.reason}"
        )

    def get_tracks_needing_verification(
        self,
        tracked_objects: List[TrackedObject],
        current_time: float,
    ) -> List[TrackedObject]:
        """
        Get tracks that need VLM verification, prioritized by:
        1. Stable tracks (confirmed status, consistent position)
        2. Low confidence tracks (need semantic validation)
        3. New tracks (first-time verification)
        """
        candidates = [
            obj for obj in tracked_objects
            if self.should_verify(obj.track_id, current_time)
        ]

        # Prioritize stable confirmed tracks and low-confidence tentative tracks
        def priority_key(obj: TrackedObject) -> tuple:
            ts = self._tracks.get(obj.track_id)
            if ts is None:
                return (2, 0.0)  # New track: medium priority

            # Priority 1: Stable confirmed tracks (periodic re-verification)
            if ts.status == "confirmed" and ts.hit_count >= 5:
                return (0, -ts.hit_count)  # Higher hit count = higher priority

            # Priority 2: Low confidence tentative tracks (need validation)
            if ts.status == "tentative" and ts.dino_score_avg < settings.DINO_CANDIDATE_THRESHOLD:
                return (1, ts.dino_score_avg)  # Lower score = higher priority

            # Priority 3: Other tracks
            return (2, -ts.hit_count)

        candidates.sort(key=priority_key)
        return candidates

    def reset(self) -> None:
        self._tracks.clear()
