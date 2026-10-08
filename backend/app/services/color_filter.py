"""
app/services/color_filter.py
------------------------------
Color-based post-processing filter for detection candidates.

Uses HSV color space statistics to validate whether a detected bounding box
matches the expected color characteristics of the target object.

This is more reliable than expecting Grounding DINO to understand color
descriptions in text prompts.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.logging import logger


class ColorFilter:
    """
    HSV-based color filter for detection post-processing.

    Analyzes the color distribution within a cropped detection region
    and validates against expected color ranges.
    """

    def __init__(self, color_rules: List[dict]):
        """
        Args:
            color_rules: List of color filter rules, each containing:
                - name: Color name (for logging)
                - h_range: [h_min, h_max] in OpenCV HSV (0-180)
                - s_range: [s_min, s_max] (0-255)
                - v_range: [v_min, v_max] (0-255)
                - min_ratio: Minimum ratio of pixels matching this color (0.0-1.0)
        """
        self.color_rules = color_rules
        self.enabled = len(color_rules) > 0

    def validate(
        self,
        crop: np.ndarray,
        detection_label: str = "",
    ) -> Tuple[bool, str, dict, float]:
        """
        Validate if the cropped region matches expected color characteristics.

        Args:
            crop: BGR image crop of the detection
            detection_label: Label of the detection (for logging)

        Returns:
            (is_valid, reason, stats, penalty_factor) tuple
            - is_valid: True if color matches well, False if mismatch
            - reason: Human-readable reason
            - stats: Color statistics dict
            - penalty_factor: Score multiplier (1.0 = no penalty, 0.3 = strong penalty)
        """
        if not self.enabled:
            return True, "no_color_filter", {}, 1.0

        if crop is None or crop.size == 0:
            return False, "empty_crop", {}, 0.3

        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        total_pixels = h * w

        # Check each color rule (OR logic: any rule passing means valid)
        stats = {}
        best_ratio = 0.0
        best_rule_name = ""

        for rule in self.color_rules:
            color_name = rule.get("name", "unknown")
            h_range = rule.get("h_range", [0, 180])
            s_range = rule.get("s_range", [0, 255])
            v_range = rule.get("v_range", [0, 255])
            min_ratio = rule.get("min_ratio", 0.3)

            # Create mask for this color range
            lower = np.array([h_range[0], s_range[0], v_range[0]], dtype=np.uint8)
            upper = np.array([h_range[1], s_range[1], v_range[1]], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)

            # Calculate ratio of matching pixels
            matching_pixels = np.count_nonzero(mask)
            ratio = matching_pixels / total_pixels

            stats[color_name] = {
                "ratio": round(ratio, 3),
                "min_required": min_ratio,
                "passed": ratio >= min_ratio,
            }

            if ratio > best_ratio:
                best_ratio = ratio
                best_rule_name = color_name

            # If this rule passes strongly, full confidence
            if ratio >= min_ratio:
                return True, f"color_match_{color_name}({ratio:.2f}>={min_ratio})", stats, 1.0

        # No rule passed - apply penalty based on best match
        # Calculate penalty factor based on how close we got
        if best_ratio > 0:
            # Partial match: penalty proportional to shortfall
            # best_ratio / min_required gives us a ratio of how close we got
            best_min_required = stats[best_rule_name]["min_required"]
            match_ratio = best_ratio / best_min_required if best_min_required > 0 else 0

            if match_ratio >= 0.7:  # 70-99% of required
                penalty_factor = 0.7
                severity = "minor"
            elif match_ratio >= 0.5:  # 50-69% of required
                penalty_factor = 0.5
                severity = "moderate"
            else:  # <50% of required
                penalty_factor = 0.3
                severity = "major"
        else:
            # No color match at all
            penalty_factor = 0.3
            severity = "major"

        failed_reasons = [
            f"{name}:{s['ratio']:.2f}<{s['min_required']}"
            for name, s in stats.items()
        ]
        reason = f"color_mismatch_{severity}({', '.join(failed_reasons)})"
        return False, reason, stats, penalty_factor

    def validate_batch(
        self,
        frame: np.ndarray,
        detections: List[dict],
    ) -> List[Tuple[bool, float]]:
        """
        Validate a batch of detections from the same frame.

        Args:
            frame: Full BGR frame
            detections: List of detection dicts with keys: x1, y1, x2, y2, label

        Returns:
            List of (is_valid, penalty_factor) tuples
        """
        if not self.enabled:
            return [(True, 1.0)] * len(detections)

        results = []
        for det in detections:
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            label = det.get("label", "")

            # Crop detection region
            crop = frame[y1:y2, x1:x2]

            # Validate
            is_valid, reason, stats, penalty_factor = self.validate(crop, label)
            results.append((is_valid, penalty_factor))

            if not is_valid:
                logger.debug(
                    f"Color penalty applied: {label} at ({x1},{y1},{x2},{y2}) - "
                    f"{reason}, penalty={penalty_factor:.2f}"
                )

        return results


def create_color_filter(color_rules: Optional[List[dict]] = None) -> ColorFilter:
    """
    Factory function to create a ColorFilter instance.

    Args:
        color_rules: List of color filter rules (see ColorFilter.__init__)

    Returns:
        ColorFilter instance
    """
    if color_rules is None:
        color_rules = []
    return ColorFilter(color_rules)


# ────────────────────────────────────────────────────────────────────────────
# Preset color filters for common targets
# ────────────────────────────────────────────────────────────────────────────

PRESET_COLOR_FILTERS = {
    "菜地": [
        {
            "name": "green",
            "h_range": [35, 85],
            "s_range": [40, 255],
            "v_range": [40, 255],
            "min_ratio": 0.25,
        },
        {
            "name": "brown",
            "h_range": [10, 25],
            "s_range": [30, 200],
            "v_range": [20, 150],
            "min_ratio": 0.20,
        },
    ],
    "菜园": [
        {
            "name": "green",
            "h_range": [35, 85],
            "s_range": [40, 255],
            "v_range": [40, 255],
            "min_ratio": 0.30,
        },
    ],
    "水塘": [
        {
            "name": "blue",
            "h_range": [100, 130],
            "s_range": [30, 255],
            "v_range": [30, 255],
            "min_ratio": 0.40,
        },
        {
            "name": "dark_blue",
            "h_range": [90, 110],
            "s_range": [20, 150],
            "v_range": [20, 100],
            "min_ratio": 0.35,
        },
    ],
    "鱼塘": [
        {
            "name": "blue",
            "h_range": [100, 130],
            "s_range": [30, 255],
            "v_range": [30, 255],
            "min_ratio": 0.35,
        },
        {
            "name": "dark_water",
            "h_range": [90, 110],
            "s_range": [20, 150],
            "v_range": [20, 100],
            "min_ratio": 0.30,
        },
    ],
    "大棚": [
        {
            "name": "white",
            "h_range": [0, 180],
            "s_range": [0, 50],
            "v_range": [150, 255],
            "min_ratio": 0.40,
        },
        {
            "name": "light_gray",
            "h_range": [0, 180],
            "s_range": [0, 80],
            "v_range": [100, 200],
            "min_ratio": 0.35,
        },
    ],
    "太阳能板": [
        {
            "name": "dark_blue",
            "h_range": [100, 130],
            "s_range": [30, 200],
            "v_range": [20, 120],
            "min_ratio": 0.40,
        },
        {
            "name": "black",
            "h_range": [0, 180],
            "s_range": [0, 100],
            "v_range": [0, 80],
            "min_ratio": 0.35,
        },
    ],
}


def get_preset_color_filter(target_name: str) -> Optional[List[dict]]:
    """
    Get preset color filter rules for a known target.

    Args:
        target_name: Chinese name of the target

    Returns:
        List of color filter rules, or None if no preset exists
    """
    return PRESET_COLOR_FILTERS.get(target_name)
