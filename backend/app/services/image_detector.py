"""
app/services/image_detector.py
------------------------------
Still-image YOLOE detection with a per-model registry (REQ1 + REQ3).

Deliberately separate from the video `get_detector()` singleton: REQ3 routes
each image-detect request to a chosen trained model (best.pt) OR to the
zero-shot YOLOE base model. A small LRU cache keeps loaded models warm and
frees GPU memory (shared with the warm video detector) on eviction.

Two modes:
  - zero-shot : load YOLOE base weights, `set_classes(user class names)`.
  - trained   : load best.pt; classes are baked into `model.names`
                (do NOT call set_classes — that would clobber them).

All `.predict()` calls hold a per-model lock and are intended to be invoked
from the API via `await asyncio.to_thread(...)` so they never block the event
loop or the live video SSE pipeline.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import BoundingBox, Detection


# BGR palette for annotated outputs.
_PALETTE = [
    (0, 255, 255), (0, 140, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0),
    (0, 0, 255), (255, 0, 0), (180, 0, 255), (0, 255, 100), (255, 100, 0),
]


class _LoadedModel:
    """A loaded ultralytics model + a lock (predict/set_classes aren't safe to
    interleave across threads on one model instance)."""

    def __init__(self, model, is_yoloe: bool) -> None:
        self.model = model
        self.is_yoloe = is_yoloe
        self.lock = threading.Lock()


class ImageModelRegistry:
    """Process-wide LRU cache of loaded image-detection models, keyed by the
    absolute weights path."""

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, capacity)
        self._cache: "OrderedDict[str, _LoadedModel]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, weights_path: str) -> _LoadedModel:
        key = str(Path(weights_path).resolve())
        with self._lock:
            lm = self._cache.get(key)
            if lm is not None:
                self._cache.move_to_end(key)
                return lm
        # Load outside the registry lock (loading is slow), re-check after.
        lm = self._load(weights_path)
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                self._cache.move_to_end(key)
                return existing
            self._cache[key] = lm
            self._cache.move_to_end(key)
            self._evict_if_needed()
        return lm

    def evict(self, weights_path: str) -> None:
        key = str(Path(weights_path).resolve())
        with self._lock:
            old = self._cache.pop(key, None)
        if old is not None:
            self._free(old)

    def _evict_if_needed(self) -> None:
        # caller holds self._lock
        while len(self._cache) > self._capacity:
            _, old = self._cache.popitem(last=False)
            self._free(old)

    @staticmethod
    def _free(lm: "_LoadedModel") -> None:
        try:
            del lm.model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(f"Error freeing image model: {exc}")

    def _load(self, weights_path: str) -> _LoadedModel:
        os.environ.setdefault("YOLO_OFFLINE", "1")
        os.environ.setdefault("ULTRALYTICS_OFFLINE", "1")

        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(f"模型权重不存在: {weights_path}")

        # Prefer YOLOE (required for zero-shot set_classes); fall back to YOLO.
        try:
            from ultralytics import YOLOE  # type: ignore

            model = YOLOE(str(p))
            is_yoloe = True
            logger.info(f"Loaded image model via YOLOE: {p}")
        except Exception as exc:
            from ultralytics import YOLO

            model = YOLO(str(p))
            is_yoloe = False
            logger.info(f"Loaded image model via YOLO (no YOLOE: {exc}): {p}")

        if settings.DEVICE and settings.DEVICE != "cpu":
            try:
                model.to(settings.DEVICE)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"model.to({settings.DEVICE}) failed: {exc}")
        return _LoadedModel(model, is_yoloe)


_registry: Optional[ImageModelRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> ImageModelRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ImageModelRegistry(settings.IMAGE_MODEL_CACHE_SIZE)
    return _registry


def _predict_locked(
    lm: _LoadedModel,
    image_bgr: np.ndarray,
    conf: float,
    set_classes: Optional[List[str]],
) -> List[Detection]:
    """Run one prediction. When `set_classes` is given (zero-shot), set the
    open-vocab classes and predict atomically under the model lock so
    concurrent requests with different class lists don't race."""
    if image_bgr is None or image_bgr.size == 0:
        return []
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with lm.lock:
        names_override: Optional[dict] = None
        if set_classes is not None:
            if not hasattr(lm.model, "set_classes"):
                raise RuntimeError(
                    "当前模型不支持零样本文本提示，需要 YOLOE/YOLO-World 权重 "
                    "及 mobileclip_blt.ts 文本编码器。"
                )
            lm.model.set_classes(set_classes)
            names_override = {i: n for i, n in enumerate(set_classes)}
        results = lm.model.predict(
            image_rgb, conf=conf, verbose=False, device=settings.DEVICE
        )

    return _results_to_detections(results, lm.model, names_override)


def _results_to_detections(results, model, names_override: Optional[dict]) -> List[Detection]:
    dets: List[Detection] = []
    if not results:
        return dets
    r = results[0]
    names = names_override or getattr(r, "names", None) or getattr(model, "names", {}) or {}
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return dets

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), score, cid in zip(xyxy, confs, clss):
        cid = int(cid)
        if isinstance(names, dict):
            label = names.get(cid, names.get(str(cid), f"class_{cid}"))
        elif names is not None and cid < len(names):
            label = names[cid]
        else:
            label = f"class_{cid}"
        dets.append(
            Detection(
                label=str(label),
                score=round(float(score), 4),
                bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
            )
        )
    return dets


def detect_with_model(
    weights_path: str, image_bgr: np.ndarray, conf: Optional[float] = None
) -> List[Detection]:
    """Detect using a trained model's baked-in classes (no set_classes)."""
    lm = get_registry().get(weights_path)
    c = conf if conf is not None else settings.IMAGE_DETECT_CONF
    return _predict_locked(lm, image_bgr, c, set_classes=None)


def detect_zeroshot(
    class_names: List[str], image_bgr: np.ndarray, conf: Optional[float] = None
) -> List[Detection]:
    """Zero-shot detect with user-supplied open-vocabulary class names."""
    base = settings.yoloe_base_model
    if not base:
        raise RuntimeError(
            "未配置 YOLOE 基础权重（请在 .env 设置 YOLOE_BASE_MODEL）。"
        )
    cls = [c.strip() for c in (class_names or []) if c and c.strip()]
    if not cls:
        raise ValueError("零样本检测需要至少一个类别名。")
    lm = get_registry().get(base)
    c = conf if conf is not None else settings.IMAGE_DETECT_CONF
    return _predict_locked(lm, image_bgr, c, set_classes=cls)


def annotate_image(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw boxes + (CJK-safe) labels onto a copy of the image."""
    from PIL import Image, ImageDraw  # local import
    from app.services.visualizer import _get_font  # reuse CJK font lookup

    annotated = image_bgr.copy()
    for i, d in enumerate(detections):
        color = _PALETTE[i % len(_PALETTE)]
        x1, y1, x2, y2 = int(d.bbox.x1), int(d.bbox.y1), int(d.bbox.x2), int(d.bbox.y2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _get_font(22)
    for i, d in enumerate(detections):
        color = _PALETTE[i % len(_PALETTE)]
        x1, y1 = int(d.bbox.x1), int(d.bbox.y1)
        text = f"{d.label} {d.score:.2f}"
        tb = font.getbbox(text) if hasattr(font, "getbbox") else (0, 0, 0, 16)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(y1 - th - 6, 0)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=(color[2], color[1], color[0]))
        draw.text((x1 + 4, ty + 1), text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
