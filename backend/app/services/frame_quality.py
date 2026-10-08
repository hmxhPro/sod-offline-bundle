"""
app/services/frame_quality.py
------------------------------
Frame quality assessment to skip low-quality frames (black screens,
blurred frames, etc.) before running expensive detection.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple

from app.core.logging import logger


class FrameQualityChecker:
    """
    Checks frame quality to avoid wasting GPU inference on bad frames.

    Detects:
    - Black/near-black frames (video end, transitions)
    - Extremely dark frames (underexposed)
    - Extremely bright frames (overexposed/white screens)
    - Very blurred frames (out of focus)
    """

    def __init__(
        self,
        black_threshold: float = 15.0,  # Mean brightness below this = black
        dark_threshold: float = 30.0,   # Mean brightness below this = too dark
        bright_threshold: float = 240.0, # Mean brightness above this = overexposed
        blur_threshold: float = 100.0,   # Laplacian variance below this = blurred
        min_std_dev: float = 10.0,       # Std dev below this = flat/uniform
    ):
        self.black_threshold = black_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.blur_threshold = blur_threshold
        self.min_std_dev = min_std_dev

    def check_frame(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Check if frame is good quality for detection.

        Returns:
            (is_good, reason) - True if frame is good, False with reason if bad
        """
        if frame is None or frame.size == 0:
            return False, "empty_frame"

        # Convert to grayscale for analysis
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # 1. Check for black/near-black frames
        mean_brightness = np.mean(gray)
        if mean_brightness < self.black_threshold:
            return False, f"black_frame (brightness={mean_brightness:.1f})"

        # 2. Check for extremely dark frames
        if mean_brightness < self.dark_threshold:
            return False, f"too_dark (brightness={mean_brightness:.1f})"

        # 3. Check for overexposed/white frames
        if mean_brightness > self.bright_threshold:
            return False, f"overexposed (brightness={mean_brightness:.1f})"

        # 4. Check for flat/uniform frames (no content variation)
        std_dev = np.std(gray)
        if std_dev < self.min_std_dev:
            return False, f"flat_frame (std_dev={std_dev:.1f})"

        # 5. Check for blur (optional, can be expensive)
        # Laplacian variance measures edge sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < self.blur_threshold:
            return False, f"blurred (variance={laplacian_var:.1f})"

        return True, "good"

    def check_frame_fast(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Fast quality check (skips blur detection).
        Use this for real-time processing.
        """
        if frame is None or frame.size == 0:
            return False, "empty_frame"

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Quick brightness and variance checks
        mean_brightness = np.mean(gray)

        if mean_brightness < self.black_threshold:
            return False, f"black_frame (brightness={mean_brightness:.1f})"

        if mean_brightness < self.dark_threshold:
            return False, f"too_dark (brightness={mean_brightness:.1f})"

        if mean_brightness > self.bright_threshold:
            return False, f"overexposed (brightness={mean_brightness:.1f})"

        std_dev = np.std(gray)
        if std_dev < self.min_std_dev:
            return False, f"flat_frame (std_dev={std_dev:.1f})"

        return True, "good"


def create_quality_checker(
    black_threshold: float = 15.0,
    dark_threshold: float = 30.0,
    bright_threshold: float = 240.0,
    blur_threshold: float = 100.0,
    min_std_dev: float = 10.0,
) -> FrameQualityChecker:
    """Factory function to create a quality checker."""
    return FrameQualityChecker(
        black_threshold=black_threshold,
        dark_threshold=dark_threshold,
        bright_threshold=bright_threshold,
        blur_threshold=blur_threshold,
        min_std_dev=min_std_dev,
    )
