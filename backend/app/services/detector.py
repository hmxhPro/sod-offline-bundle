"""
app/services/detector.py
-------------------------
Detection model abstraction.

Supports two backends:
  1. Grounding DINO  – via groundingdino library (preferred for open-vocab detection)
  2. Florence-2      – via HuggingFace transformers

The singleton `get_detector()` lazily loads the model once and reuses it
across all tasks (thread/process safe for single-process inference).

Multi-GPU support note:
  To scale to multiple GPUs, instantiate one Detector per GPU in a pool
  and distribute tasks via `task_manager.semaphore` per GPU.
  Example using torch.multiprocessing or Ray is described in README.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import BoundingBox, Detection


# ────────────────────────────────────────────────────────────────────────────
# Data class for raw detector output
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class RawDetection:
    """Single raw detection before tracking."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str
    color_penalty: float = 1.0  # Color filter penalty factor (1.0 = no penalty)

    def to_xyxys(self) -> np.ndarray:
        """Return [x1, y1, x2, y2, score] array for ByteTrack."""
        # Apply color penalty to score for tracking
        adjusted_score = self.score * self.color_penalty
        return np.array([self.x1, self.y1, self.x2, self.y2, adjusted_score], dtype=np.float32)

    def to_schema(self) -> Detection:
        return Detection(
            label=self.label,
            score=round(float(self.score * self.color_penalty), 4),
            bbox=BoundingBox(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2),
        )


# ────────────────────────────────────────────────────────────────────────────
# Abstract base
# ────────────────────────────────────────────────────────────────────────────

_UNK_PATTERN = re.compile(r'\[?unk\]?', re.IGNORECASE)


def _parse_prompt_labels(prompt: str) -> List[str]:
    """Split a prompt like 'person . car . dog' into individual class names."""
    parts = re.split(r'[.。,，;；]', prompt)
    return [p.strip() for p in parts if p.strip()]


def _clean_phrase(phrase: str, prompt_labels: List[str]) -> str:
    """Replace garbled 'unk' phrases with the original prompt label."""
    cleaned = phrase.strip()
    if not cleaned or _UNK_PATTERN.search(cleaned):
        return prompt_labels[0] if prompt_labels else "object"
    return cleaned


class BaseDetector(ABC):
    def __init__(self, device: str) -> None:
        self.device = device
        self._model = None

    @property
    def model(self):
        return self._model

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory (idempotent)."""

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> List[RawDetection]:
        """
        Run inference on a single BGR image (OpenCV format).

        Returns a list of RawDetection objects.
        """


# ────────────────────────────────────────────────────────────────────────────
# Grounding DINO detector
# ────────────────────────────────────────────────────────────────────────────

class GroundingDINODetector(BaseDetector):
    """
    Uses the official groundingdino package.
    Install: pip install groundingdino-py  (or from source)

    Config & checkpoint must exist at paths specified in settings.
    """

    def __init__(self, device: str) -> None:
        super().__init__(device)
        self._predict_lock = __import__("threading").Lock()

    def load(self) -> None:
        if self._model is not None:
            return  # Already loaded
        try:
            # Force offline mode to use local cache
            import os
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'

            from groundingdino.util.inference import load_model as gdino_load_model
            import groundingdino.datasets.transforms as T

            logger.info(
                f"Loading Grounding DINO from {settings.GDINO_CHECKPOINT_PATH}"
            )
            self._model = gdino_load_model(
                str(settings.GDINO_CONFIG_PATH),
                str(settings.GDINO_CHECKPOINT_PATH),
                device=self.device,
            )
            self._model.eval()
            # Workaround: some GroundingDINO versions omit `poss` from __init__.
            if not hasattr(self._model, "poss"):
                self._model.poss = []

            # Cache the image transform — creating it per-frame is wasteful.
            self._transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            logger.info("Grounding DINO loaded successfully.")
        except ImportError:
            raise ImportError(
                "groundingdino-py is not installed. "
                "Run: pip install groundingdino-py "
                "or: pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
            )

    def predict(
        self,
        image: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> List[RawDetection]:
        from groundingdino.util.inference import predict as gdino_predict

        if image is None or image.size == 0:
            logger.warning("predict() received empty image, skipping.")
            return []
        if image.ndim != 3 or image.shape[2] != 3:
            logger.warning(f"predict() unexpected image shape {image.shape}, skipping.")
            return []
        if image.shape[0] < 32 or image.shape[1] < 32:
            logger.warning(f"predict() image too small {image.shape}, skipping.")
            return []

        # Convert BGR (OpenCV) -> RGB PIL
        pil_image = Image.fromarray(image[:, :, ::-1])
        h, w = image.shape[:2]

        # Use the cached transform (created once in load())
        image_tensor, _ = self._transform(pil_image, None)

        # Sanitize prompt: Grounding DINO expects lowercase phrases ending with '.'
        clean_prompt = prompt.strip().lower()
        if not clean_prompt.endswith("."):
            clean_prompt += "."

        def _run_predict():
            return gdino_predict(
                model=self._model,
                image=image_tensor,
                caption=clean_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device,
            )

        # Thread-safe inference; inference_mode is faster than no_grad
        with self._predict_lock:
            with torch.inference_mode():
                try:
                    boxes, logits, phrases = _run_predict()
                except AttributeError as e:
                    err_msg = str(e)
                    if "poss" in err_msg:
                        self._model.poss = []
                        boxes, logits, phrases = _run_predict()
                    elif "features" in err_msg:
                        # GroundingDINO version mismatch: model structure changed
                        logger.warning(f"GroundingDINO AttributeError (features): {e}, skipping frame.")
                        return []
                    else:
                        raise
                except (IndexError, RuntimeError) as e:
                    # backbone produced no feature maps (e.g. degenerate image after resize)
                    logger.warning(f"GroundingDINO inference error: {e}, skipping frame.")
                    return []

        # boxes are [cx, cy, w, h] normalized → convert to absolute xyxy
        prompt_labels = _parse_prompt_labels(prompt)
        results: List[RawDetection] = []
        for box, score, phrase in zip(boxes.cpu().numpy(), logits.cpu().numpy(), phrases):
            cx, cy, bw, bh = box
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            label = _clean_phrase(phrase, prompt_labels)
            results.append(
                RawDetection(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    score=float(score), label=label,
                )
            )
        return results


# ────────────────────────────────────────────────────────────────────────────
# Florence-2 detector
# ────────────────────────────────────────────────────────────────────────────

class Florence2Detector(BaseDetector):
    """
    Uses Microsoft Florence-2 via HuggingFace transformers.
    Task: <OPEN_VOCABULARY_DETECTION>

    Model ID: microsoft/Florence-2-large (or -base)
    """

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            model_id = settings.FLORENCE2_MODEL_ID
            logger.info(f"Loading Florence-2 model: {model_id}")

            self._processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            self._model.eval()
            logger.info("Florence-2 loaded successfully.")
        except ImportError:
            raise ImportError(
                "transformers is not installed. Run: pip install transformers"
            )

    def predict(
        self,
        image: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,  # Not directly used by Florence-2
    ) -> List[RawDetection]:
        from transformers import AutoModelForCausalLM, AutoProcessor

        pil_image = Image.fromarray(image[:, :, ::-1])
        h, w = image.shape[:2]

        task_prompt = "<OPEN_VOCABULARY_DETECTION>"
        text_input = f"{task_prompt}{prompt}"

        inputs = self._processor(
            text=text_input, images=pil_image, return_tensors="pt"
        ).to(self.device, torch.float16 if "cuda" in self.device else torch.float32)

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self._processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(w, h),
        )
        od_result = parsed.get(task_prompt, {})
        bboxes = od_result.get("bboxes", [])
        labels = od_result.get("bboxes_labels", [])

        results: List[RawDetection] = []
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            # Florence-2 doesn't return per-box scores; use a fixed confidence
            score = 0.90
            if score >= box_threshold:
                results.append(
                    RawDetection(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        score=score, label=label,
                    )
                )
        return results


# ────────────────────────────────────────────────────────────────────────────
# Trained-model detector (uses a fine-tuned model's baked-in classes)
# ────────────────────────────────────────────────────────────────────────────

class TrainedModelDetector(BaseDetector):
    """
    Adapter that runs a user-trained model (best.pt) inside the video pipeline.

    Unlike the open-vocabulary detectors, this ignores the text `prompt`: the
    model's classes are baked into its weights (`model.names`). It delegates to
    the still-image detection registry (`image_detector.detect_with_model`) so
    model loading, the per-model LRU cache, the YOLOE→YOLO fallback and the
    per-model predict lock are all reused — no duplicate model-loading code.
    """

    def __init__(self, device: str, weights_path: str) -> None:
        super().__init__(device)
        self.weights_path = weights_path

    def load(self) -> None:
        # No-op: the underlying ultralytics model is loaded lazily (and cached)
        # by the image-detection registry on the first predict() call. The API
        # layer already fails fast if the weights file is missing.
        return

    def predict(
        self,
        image: np.ndarray,
        prompt: str,            # ignored — classes come from the weights
        box_threshold: float,
        text_threshold: float,  # ignored — no text head
    ) -> List[RawDetection]:
        # Lazy import to avoid a circular import (image_detector imports nothing
        # from this module at import time, but keep it lazy to be safe).
        from app.services.image_detector import detect_with_model

        detections = detect_with_model(self.weights_path, image, conf=box_threshold)
        return [
            RawDetection(
                x1=float(d.bbox.x1), y1=float(d.bbox.y1),
                x2=float(d.bbox.x2), y2=float(d.bbox.y2),
                score=float(d.score), label=str(d.label),
            )
            for d in detections
        ]


# ────────────────────────────────────────────────────────────────────────────
# Factory / Singleton
# ────────────────────────────────────────────────────────────────────────────

_detector_instance: BaseDetector | None = None
_detector_lock = __import__("threading").Lock()

# Per-weights-path cache of trained-model detectors. These are nearly stateless
# (the heavy model lives in the shared image-detection registry), so caching is
# just to avoid re-creating the thin adapter for each task on the same model.
_trained_detectors: dict[str, "TrainedModelDetector"] = {}
_trained_detectors_lock = __import__("threading").Lock()


def get_trained_detector(weights_path: str) -> TrainedModelDetector:
    """Return a cached TrainedModelDetector for the given weights file."""
    with _trained_detectors_lock:
        det = _trained_detectors.get(weights_path)
        if det is None:
            det = TrainedModelDetector(device=settings.DEVICE, weights_path=weights_path)
            det.load()
            _trained_detectors[weights_path] = det
            logger.info(f"Trained-model detector ready: {weights_path}")
        return det



def get_detector() -> BaseDetector:
    """Return the singleton detector, loading it on first call."""
    global _detector_instance
    if _detector_instance is not None:
        return _detector_instance
    with _detector_lock:
        if _detector_instance is not None:
            return _detector_instance
        model_name = settings.DETECTION_MODEL.lower()
        device = settings.DEVICE

        if model_name == "grounding_dino":
            detector = GroundingDINODetector(device=device)
        elif model_name == "florence2":
            detector = Florence2Detector(device=device)
        elif model_name == "yoloe":
            from app.services.yoloe_detector import YOLOEDetector
            detector = YOLOEDetector(device=device)
        else:
            raise ValueError(f"Unknown DETECTION_MODEL: {model_name}")

        detector.load()  # Only assign after successful load
        logger.info(f"Detector ready: {model_name} on {device}")
        _detector_instance = detector

    return _detector_instance
