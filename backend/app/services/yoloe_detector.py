"""
app/services/yoloe_detector.py
------------------------------
First-class YOLOE open-vocabulary detector for the VIDEO pipeline, with optional
SAHI sliced inference for small objects.

YOLOE (Ultralytics) is an open-vocabulary detector: ``set_classes(text labels)``
lets it detect arbitrary classes without fine-tuning. SAHI (Slicing Aided Hyper
Inference) slices large frames into tiles to boost small-object recall — which is
the whole point of this project.

Weights come from ``settings.yoloe_base_model`` (``YOLOE_BASE_MODEL`` in .env),
the same base used by still-image detection and custom training, so the entire
system shares one YOLOE base.

Selected as the default detector when ``DETECTION_MODEL == "yoloe"`` (see
``app/services/detector.py:get_detector``).

Installation:
    pip install ultralytics>=8.3.0 sahi>=0.11.18
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.services.detector import BaseDetector, RawDetection


class YOLOEDetector(BaseDetector):
    """
    YOLOE open-vocabulary detector with optional SAHI slicing.

    Loaded once as the video pipeline's default detector. Open-vocab classes are
    supplied per request via ``set_classes()`` parsed from the (LLM-normalized)
    prompt. On large frames it falls back to SAHI sliced inference.
    """

    def __init__(self, device: str) -> None:
        super().__init__(device)
        self._model = None
        self._sahi_enabled = True
        # set_classes + predict must be atomic: this singleton can be shared by
        # concurrent detection tasks carrying different class lists.
        self._predict_lock = threading.Lock()

    def load(self) -> None:
        """Load the YOLOE base weights (idempotent)."""
        if self._model is not None:
            return  # Already loaded

        # Force offline mode; never hit the network at inference time.
        os.environ.setdefault("YOLO_OFFLINE", "1")
        os.environ.setdefault("ULTRALYTICS_OFFLINE", "1")

        weights = settings.yoloe_base_model
        if not weights:
            raise RuntimeError(
                "未配置 YOLOE 权重（请在 .env 设置 YOLOE_BASE_MODEL）。"
            )

        wp = Path(weights)
        weights_arg = str(wp) if (wp.exists() and wp.is_file()) else weights
        if wp.exists() and wp.is_file():
            logger.info(f"Using local YOLOE weights: {wp}")
        else:
            logger.info(f"YOLOE weights resolved by name (cache/download): {weights}")

        # Prefer the dedicated YOLOE class (required for zero-shot set_classes);
        # fall back to the generic YOLO loader. Mirrors image_detector._load.
        try:
            from ultralytics import YOLOE  # type: ignore

            self._model = YOLOE(weights_arg)
            logger.info(f"Loaded YOLOE via ultralytics.YOLOE: {weights_arg}")
        except ImportError as e:
            # ultralytics itself is missing → unrecoverable.
            raise ImportError(
                "Required packages not installed. Run:\n"
                "  pip install ultralytics>=8.3.0 sahi>=0.11.18"
            ) from e
        except Exception as exc:
            from ultralytics import YOLO

            self._model = YOLO(weights_arg)
            logger.info(
                f"Loaded YOLOE weights via YOLO fallback (no YOLOE: {exc}): {weights_arg}"
            )

        if self.device and self.device != "cpu":
            try:
                self._model.to(self.device)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning(f"model.to({self.device}) failed: {exc}")

        # SAHI configuration
        self._slice_height = getattr(settings, "SAHI_SLICE_HEIGHT", 640)
        self._slice_width = getattr(settings, "SAHI_SLICE_WIDTH", 640)
        self._overlap_height_ratio = getattr(settings, "SAHI_OVERLAP_HEIGHT_RATIO", 0.2)
        self._overlap_width_ratio = getattr(settings, "SAHI_OVERLAP_WIDTH_RATIO", 0.2)

        logger.info(f"YOLOE loaded successfully on {self.device}")
        logger.info(
            f"SAHI config: slice={self._slice_height}x{self._slice_width}, "
            f"overlap={self._overlap_height_ratio}x{self._overlap_width_ratio}"
        )

    def _parse_classes_from_prompt(self, prompt: str) -> List[str]:
        """Parse class names from a prompt like 'person . car . dog'."""
        import re

        parts = re.split(r"[.。,，;；]", prompt)
        return [p.strip() for p in parts if p.strip()]

    def predict(
        self,
        image: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,  # unused — YOLOE has no separate text head
    ) -> List[RawDetection]:
        """Run YOLOE detection, using SAHI slicing on large frames."""
        if image is None or image.size == 0:
            logger.warning("predict() received empty image, skipping.")
            return []
        if image.ndim != 3 or image.shape[2] != 3:
            logger.warning(f"predict() unexpected image shape {image.shape}, skipping.")
            return []
        if image.shape[0] < 32 or image.shape[1] < 32:
            logger.warning(f"predict() image too small {image.shape}, skipping.")
            return []

        classes = self._parse_classes_from_prompt(prompt)
        if not classes:
            logger.warning("No valid classes found in prompt, skipping.")
            return []

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        use_sahi = self._sahi_enabled and (
            h > self._slice_height * 1.5 or w > self._slice_width * 1.5
        )

        # Hold set_classes + inference under one lock so a concurrent task with a
        # different prompt cannot swap classes mid-predict.
        with self._predict_lock:
            self._model.set_classes(classes)
            if use_sahi:
                return self._predict_with_sahi(image_rgb, box_threshold, classes)
            return self._predict_direct(image_rgb, box_threshold, classes)

    def _predict_direct(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float,
        classes: List[str],
    ) -> List[RawDetection]:
        """Direct whole-frame inference (no slicing)."""
        try:
            results = self._model.predict(
                image_rgb, conf=conf_threshold, verbose=False, device=self.device
            )

            detections: List[RawDetection] = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    for box, score, cls_id in zip(boxes, scores, class_ids):
                        x1, y1, x2, y2 = box
                        label = (
                            classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}"
                        )
                        detections.append(
                            RawDetection(
                                x1=float(x1),
                                y1=float(y1),
                                x2=float(x2),
                                y2=float(y2),
                                score=float(score),
                                label=label,
                            )
                        )
            return detections

        except Exception as e:
            logger.error(f"YOLOE direct inference error: {e}")
            return []

    def _predict_with_sahi(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float,
        classes: List[str],
    ) -> List[RawDetection]:
        """Sliced inference via SAHI for better small-object recall."""
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction

            # Wrap the already-`set_classes`'d model for SAHI.
            detection_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model=self._model,
                confidence_threshold=conf_threshold,
                device=self.device,
            )

            result = get_sliced_prediction(
                image_rgb,
                detection_model,
                slice_height=self._slice_height,
                slice_width=self._slice_width,
                overlap_height_ratio=self._overlap_height_ratio,
                overlap_width_ratio=self._overlap_width_ratio,
                verbose=0,
            )

            detections: List[RawDetection] = []
            for obj in result.object_prediction_list:
                bbox = obj.bbox
                x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
                score = obj.score.value
                cls_id = obj.category.id
                label = classes[cls_id] if cls_id < len(classes) else obj.category.name
                detections.append(
                    RawDetection(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        score=float(score),
                        label=label,
                    )
                )

            logger.debug(f"SAHI detected {len(detections)} objects")
            return detections

        except ImportError:
            logger.warning("SAHI not available, falling back to direct inference")
            return self._predict_direct(image_rgb, conf_threshold, classes)
        except Exception as e:
            logger.error(f"SAHI inference error: {e}, falling back to direct inference")
            return self._predict_direct(image_rgb, conf_threshold, classes)
