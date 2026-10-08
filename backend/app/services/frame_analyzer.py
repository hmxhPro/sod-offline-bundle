"""
app/services/frame_analyzer.py
-------------------------------
Adaptive keyframe detection for optimizing video processing.

Analyzes frame-to-frame changes using multiple methods:
  1. Frame difference (pixel-level change detection)
  2. Perceptual hash (structural similarity)
  3. Histogram comparison (color distribution changes)

Determines whether a frame is a "keyframe" requiring full detection,
or can be skipped with tracker-only propagation.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple

from app.core.logging import logger


class FrameAnalyzer:
    """
    Analyzes frame changes to determine adaptive detection interval.

    Uses multiple metrics to detect scene changes and adjust the interval
    between detections dynamically.
    """

    def __init__(
        self,
        diff_threshold: float = 0.05,
        hash_threshold: int = 8,
        hist_threshold: float = 0.85,
        min_interval: int = 10,
        max_interval: int = 30,
    ):
        """
        Args:
            diff_threshold: Fraction of pixels that must change (0.0-1.0)
            hash_threshold: Max Hamming distance for perceptual hash
            hist_threshold: Min histogram correlation (0.0-1.0)
            min_interval: Minimum frames between detections
            max_interval: Maximum frames between detections
        """
        self.diff_threshold = diff_threshold
        self.hash_threshold = hash_threshold
        self.hist_threshold = hist_threshold
        self.min_interval = min_interval
        self.max_interval = max_interval

        self.last_keyframe: Optional[np.ndarray] = None
        self.last_keyframe_hash: Optional[int] = None
        self.last_keyframe_hist: Optional[np.ndarray] = None
        self.frames_since_detection: int = 0
        self.current_interval: int = min_interval

    def should_detect(self, frame: np.ndarray) -> Tuple[bool, str, int]:
        """
        Determine if this frame requires detection based on frame count.

        Args:
            frame: Current frame (BGR format)

        Returns:
            (should_detect, reason, next_interval) tuple
        """
        self.frames_since_detection += 1

        # First frame always requires detection
        if self.last_keyframe is None:
            self._update_keyframe(frame)
            next_interval = self._calculate_next_interval(frame)
            return True, "first_frame", next_interval

        # Check if we've reached the current interval
        if self.frames_since_detection >= self.current_interval:
            # Analyze scene change to determine next interval
            next_interval = self._calculate_next_interval(frame)
            self._update_keyframe(frame)
            return True, f"interval_reached({self.current_interval})", next_interval

        # Not time to detect yet
        return False, f"waiting({self.frames_since_detection}/{self.current_interval})", self.current_interval

    def _calculate_next_interval(self, frame: np.ndarray) -> int:
        """
        Analyze frame changes and calculate the next detection interval.

        Returns:
            Next interval (clamped between min_interval and max_interval)
        """
        if self.last_keyframe is None:
            return self.min_interval

        # Analyze frame changes
        diff_ratio = self._compute_frame_diff(frame, self.last_keyframe)
        current_hash = self._compute_phash(frame)
        hamming_dist = self._hamming_distance(current_hash, self.last_keyframe_hash)
        current_hist = self._compute_histogram(frame)
        hist_corr = cv2.compareHist(
            self.last_keyframe_hist,
            current_hist,
            cv2.HISTCMP_CORREL,
        )

        # Calculate change score (0.0 = no change, 1.0 = maximum change)
        change_score = 0.0

        # Weight different metrics
        if diff_ratio > self.diff_threshold:
            change_score += (diff_ratio / self.diff_threshold) * 0.4

        if hamming_dist > self.hash_threshold:
            change_score += (hamming_dist / 64.0) * 0.3

        if hist_corr < self.hist_threshold:
            change_score += (1.0 - hist_corr) * 0.3

        change_score = min(change_score, 1.0)

        # Map change score to interval:
        # High change (>0.5) -> min_interval (detect more frequently)
        # Low change (<0.2) -> max_interval (detect less frequently)
        # Medium change -> interpolate
        if change_score > 0.5:
            next_interval = self.min_interval
        elif change_score < 0.2:
            next_interval = self.max_interval
        else:
            # Linear interpolation between min and max
            ratio = (0.5 - change_score) / 0.3  # 0.2-0.5 -> 1.0-0.0
            next_interval = int(self.min_interval + ratio * (self.max_interval - self.min_interval))

        return max(self.min_interval, min(next_interval, self.max_interval))

    def _compute_frame_diff(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> float:
        """
        Compute fraction of pixels that changed between frames.

        Uses downsampled grayscale for speed.
        """
        # Downsample for speed (320x240 is sufficient for change detection)
        h, w = frame1.shape[:2]
        target_w = min(320, w)
        target_h = int(h * target_w / w)

        small1 = cv2.resize(frame1, (target_w, target_h))
        small2 = cv2.resize(frame2, (target_w, target_h))

        # Convert to grayscale
        gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Threshold to binary (changed/unchanged)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Compute ratio of changed pixels
        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size

        return changed_pixels / total_pixels

    def _compute_phash(self, frame: np.ndarray, hash_size: int = 8) -> int:
        """
        Compute perceptual hash (pHash) of frame.

        pHash is robust to minor changes but sensitive to structural changes.
        """
        # Resize to hash_size + 1 for DCT
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))

        # Compute DCT (Discrete Cosine Transform)
        dct = cv2.dct(np.float32(resized))

        # Keep only top-left 8x8 (low frequencies)
        dct_low = dct[:hash_size, :hash_size]

        # Compute median
        median = np.median(dct_low)

        # Generate hash: 1 if above median, 0 otherwise
        hash_bits = (dct_low > median).flatten()

        # Convert to integer
        hash_value = 0
        for i, bit in enumerate(hash_bits):
            if bit:
                hash_value |= (1 << i)

        return hash_value

    def _hamming_distance(self, hash1: int, hash2: int) -> int:
        """Compute Hamming distance between two hashes."""
        xor = hash1 ^ hash2
        return bin(xor).count('1')

    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute color histogram for frame.

        Uses HSV color space for better perceptual matching.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Compute histogram for H and S channels
        hist = cv2.calcHist(
            [hsv],
            [0, 1],  # H and S channels
            None,
            [50, 60],  # bins
            [0, 180, 0, 256],  # ranges
        )

        # Normalize
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        return hist

    def _update_keyframe(self, frame: np.ndarray) -> None:
        """Update reference keyframe and reset counter."""
        self.last_keyframe = frame.copy()
        self.last_keyframe_hash = self._compute_phash(frame)
        self.last_keyframe_hist = self._compute_histogram(frame)
        self.frames_since_detection = 0

    def update_interval(self, new_interval: int) -> None:
        """Update the current detection interval."""
        self.current_interval = max(self.min_interval, min(new_interval, self.max_interval))

    def reset(self) -> None:
        """Reset analyzer state (call between tasks)."""
        self.last_keyframe = None
        self.last_keyframe_hash = None
        self.last_keyframe_hist = None
        self.frames_since_detection = 0
        self.current_interval = self.min_interval

    def get_stats(self) -> dict:
        """Get current analyzer statistics."""
        return {
            "frames_since_detection": self.frames_since_detection,
            "current_interval": self.current_interval,
            "has_keyframe": self.last_keyframe is not None,
        }


def create_frame_analyzer(
    diff_threshold: float = 0.05,
    hash_threshold: int = 8,
    hist_threshold: float = 0.85,
    min_interval: int = 10,
    max_interval: int = 30,
) -> FrameAnalyzer:
    """
    Factory function to create a FrameAnalyzer instance.

    Args:
        diff_threshold: Pixel change threshold (0.05 = 5% of pixels)
        hash_threshold: Perceptual hash Hamming distance threshold
        hist_threshold: Histogram correlation threshold (0.85 = 85% similar)
        min_interval: Minimum frames between detections
        max_interval: Maximum frames between detections
    """
    return FrameAnalyzer(
        diff_threshold=diff_threshold,
        hash_threshold=hash_threshold,
        hist_threshold=hist_threshold,
        min_interval=min_interval,
        max_interval=max_interval,
    )
