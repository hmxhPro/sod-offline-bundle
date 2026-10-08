"""
app/api/image_detect.py
-----------------------
Still-image detection (REQ1): zero-shot YOLOE OR a chosen trained model OR LLM vision model.

POST /api/image-detect                 -> detect on one/many uploaded images
GET  /api/image-detect/{batch}/{file}  -> serve an annotated result image

Supports three detection modes:
- YOLO: use a trained model (model_id)
- Zero-shot: open-vocabulary detection (class_names)
- LLM: multimodal LLM detection (llm_prompt, llm_model)
"""

from __future__ import annotations

import uuid
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.logging import logger
from app.db.models import TrainedModelRecord
from app.db.session import AsyncSessionLocal
from app.models.schemas import ImageDetectResponse, ImageDetectResultItem, Detection

router = APIRouter()

# Decoded-pixel guard for the cv2 path (P-1). MAX_IMAGE_BYTES bounds the encoded
# upload, but a small file can still decode to a huge bitmap; reject those.
_MAX_DECODED_PIXELS = 100_000_000  # ~100 megapixels


def _reject_unsafe(component: str) -> None:
    if "/" in component or "\\" in component or ".." in component:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "非法的标识符。")


def _imgdet_dir(batch_id: str):
    return settings.RESULTS_DIR / settings.IMGDET_SUBDIR / batch_id


def _convert_llm_boxes_to_detections(boxes: List, class_name: str) -> List[Detection]:
    """将 LLM 标注器返回的 BoundingBox 列表转换为 Detection 格式

    Args:
        boxes: List[BoundingBox] from llm_annotation.py
        class_name: 类别名称（通常是用户输入的提示词）

    Returns:
        List[Detection] 用于图片标注
    """
    from app.services.llm_annotation import BoundingBox

    detections = []
    for box in boxes:
        if not isinstance(box, BoundingBox):
            continue

        # BoundingBox: (class_id, cx, cy, w, h, confidence)
        # Detection: (class_id, class_name, confidence, bbox=[x1, y1, w, h])
        x1 = box.cx - box.w / 2
        y1 = box.cy - box.h / 2

        detections.append(Detection(
            class_id=box.class_id,
            class_name=class_name,
            confidence=box.confidence,
            bbox=[x1, y1, box.w, box.h]
        ))

    return detections


@router.post(
    "/image-detect",
    response_model=ImageDetectResponse,
    summary="图片检测（零样本 YOLOE 或选定的已训练模型）",
)
async def image_detect(
    files: List[UploadFile] = File(..., description="一张或多张图片"),
    model_id: Optional[str] = Form(None),
    class_names: Optional[str] = Form(None),
    conf: Optional[float] = Form(None),
    detection_mode: Optional[str] = Form(None, description="检测方式: yolo | zeroshot | llm"),
    llm_model: Optional[str] = Form(None, description="LLM 模型名称"),
    llm_prompt: Optional[str] = Form(None, description="LLM 检测提示词"),
) -> ImageDetectResponse:
    import asyncio
    from app.services.image_detector import annotate_image, detect_with_model, detect_zeroshot

    # 参数清理
    model_id = (model_id or "").strip() or None
    class_names = (class_names or "").strip() or None
    detection_mode = (detection_mode or "").strip() or None
    llm_model = (llm_model or "").strip() or None
    llm_prompt = (llm_prompt or "").strip() or None

    # 检测模式验证
    if detection_mode == "llm":
        # LLM 模式：需要 llm_prompt
        if not llm_prompt:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "LLM 检测模式需要提供 llm_prompt（检测目标描述）。",
            )
    elif detection_mode == "zeroshot":
        # 零样本模式：需要 class_names
        if not class_names:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "零样本检测模式需要提供 class_names。",
            )
    else:
        # YOLO 模式或向后兼容旧调用
        if not model_id and not class_names:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "请提供 model_id（使用已训练模型）或 class_names（零样本）。",
            )
        # 向后兼容：根据参数推断模式
        if not detection_mode:
            detection_mode = "yolo" if model_id else "zeroshot"

    if conf is not None:
        conf = max(0.0, min(1.0, float(conf)))

    if len(files) > settings.MAX_UPLOAD_FILES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"单次最多检测 {settings.MAX_UPLOAD_FILES} 张图片。",
        )

    # Resolve detection mode.
    weights_path: Optional[str] = None
    classes: List[str] = []
    mode: str
    resp_class_names: List[str] = []

    if detection_mode == "llm":
        mode = "llm"
        # LLM 检测时使用提示词作为类别名
        resp_class_names = [llm_prompt]
    elif detection_mode == "yolo" or model_id:
        mode = "model"
        try:
            async with AsyncSessionLocal() as session:
                rec = await session.get(TrainedModelRecord, model_id)
        except Exception as exc:
            logger.warning(f"image_detect model lookup failed: {exc}")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc
        if rec is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "所选模型不存在。")
        weights_path = rec.weights_path
        if isinstance(rec.class_names, dict):
            resp_class_names = [str(v) for v in rec.class_names.values()]
    else:
        mode = "zeroshot"
        classes = [c.strip() for c in class_names.replace("，", ",").split(",") if c.strip()]
        if not classes:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "零样本检测至少需要一个类别名。")
        resp_class_names = classes

    batch_id = uuid.uuid4().hex
    out_dir = _imgdet_dir(batch_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[ImageDetectResultItem] = []
    for idx, f in enumerate(files):
        raw = await f.read()
        if not raw:
            continue
        if len(raw) > settings.MAX_IMAGE_BYTES:
            logger.info(
                f"image_detect: skipping oversize file {f.filename} "
                f"(> {settings.MAX_IMAGE_BYTES // (1024 * 1024)} MB)"
            )
            continue
        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            logger.info(f"image_detect: skipping undecodable file {f.filename}")
            continue
        h, w = image.shape[:2]
        if h * w > _MAX_DECODED_PIXELS:
            logger.info(f"image_detect: skipping oversize-decode {f.filename} ({w}x{h})")
            continue

        try:
            if mode == "llm":
                # LLM 检测：保存临时图片文件供 LLM 标注器使用
                temp_img_path = out_dir / f"temp_{idx}.jpg"
                cv2.imwrite(str(temp_img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])

                # 导入并初始化 LLM 标注器
                from app.services.llm_annotation import QwenAnnotator
                from app.core.config import settings as cfg

                # 检查 LLM 配置
                if not cfg.LLM_API_KEY:
                    raise HTTPException(
                        status.HTTP_503_SERVICE_UNAVAILABLE,
                        "LLM 检测未配置：缺少 LLM_API_KEY。"
                    )

                # 使用配置的模型或用户指定的模型
                model_to_use = llm_model or cfg.LLM_MODEL or "qwen-vl-max"
                annotator = QwenAnnotator(
                    api_key=cfg.LLM_API_KEY,
                    model_name=model_to_use,
                    api_base=cfg.LLM_API_BASE or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    max_size=1024,
                    timeout=120.0,
                )

                # 调用 LLM 标注
                boxes = await annotator.annotate_image(
                    str(temp_img_path),
                    llm_prompt,
                    w,
                    h,
                )

                # 转换为 Detection 格式
                detections = _convert_llm_boxes_to_detections(boxes, llm_prompt)

                # 清理临时文件
                temp_img_path.unlink(missing_ok=True)

            elif mode == "model":
                detections = await asyncio.to_thread(detect_with_model, weights_path, image, conf)
            else:
                detections = await asyncio.to_thread(detect_zeroshot, classes, image, conf)
        except FileNotFoundError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

        annotated = await asyncio.to_thread(annotate_image, image, detections)
        out_name = f"{idx}.jpg"
        cv2.imwrite(str(out_dir / out_name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        results.append(
            ImageDetectResultItem(
                image_index=idx,
                filename=f.filename or out_name,
                width=w,
                height=h,
                detections=detections,
                annotated_url=f"/api/image-detect/{batch_id}/{out_name}",
            )
        )

    if not results:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "没有可解码的图片。")

    logger.info(
        f"image_detect batch={batch_id} mode={mode} images={len(results)} "
        f"classes={resp_class_names}"
    )
    return ImageDetectResponse(
        batch_id=batch_id,
        mode=mode,
        model_id=model_id,
        class_names=resp_class_names,
        results=results,
    )


@router.get(
    "/image-detect/{batch_id}/{filename}",
    response_class=FileResponse,
    summary="获取图片检测的标注结果图",
)
async def get_imgdet_result(batch_id: str, filename: str):
    _reject_unsafe(batch_id)
    _reject_unsafe(filename)
    p = _imgdet_dir(batch_id) / filename
    if not p.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "结果图不存在。")
    return FileResponse(path=str(p), media_type="image/jpeg")
