"""
app/services/vlm_service.py
-----------------------------
MiniCPM-V API client for semantic verification of detection crops.

Calls a vLLM-served MiniCPM-V model via OpenAI-compatible API.
Sends cropped detection regions and asks the VLM to judge whether
the crop matches the user's target description.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import httpx
import numpy as np

from app.core.config import settings
from app.core.logging import logger


@dataclass
class VLMResult:
    is_target: bool
    matched_label: str
    confidence: float
    reason: str


_VLM_PROMPT_TEMPLATE = (
    "这是一张无人机俯视视角的视频裁剪图。\n\n"
    "**重要提示**：\n"
    "1. 目标可能较小、清晰度有限，这是正常的\n"
    "2. 如果图像过暗、过亮、模糊或几乎全黑/全白，请直接判定为 false\n"
    "3. 只有当你能清晰识别出目标的主要特征时，才判定为 true\n\n"
    "用户想识别的目标是：{target_text}\n\n"
    "请判断图中主要目标是否符合用户目标。\n"
    "请只输出 JSON，不要输出其他内容：\n\n"
    '{{"is_target": true/false, "matched_label": "目标名称", '
    '"confidence": 0.0-1.0, "reason": "一句话说明原因"}}\n\n'
    "注意：confidence 应该反映你的确信程度，如果图像质量差或目标不清晰，confidence 应该很低。"
)


class VLMService:
    def __init__(self) -> None:
        self._client: Optional[httpx.Client] = None

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=settings.VLM_API_BASE,
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        return self._client

    def crop_detection(
        self,
        frame: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        frame_area = h * w

        if frame_area > 0 and (box_area / frame_area) < settings.SMALL_OBJECT_AREA_RATIO:
            padding = settings.CROP_PADDING_SMALL
        else:
            padding = settings.CROP_PADDING_NORMAL

        pad_x = box_w * padding
        pad_y = box_h * padding
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))

        return frame[cy1:cy2, cx1:cx2]

    def _encode_image_b64(self, image: np.ndarray) -> str:
        success, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise RuntimeError("Failed to encode crop as JPEG")
        return base64.b64encode(buf).decode("utf-8")

    def verify_crop(self, crop: np.ndarray, target_text: str) -> Optional[VLMResult]:
        client = self._ensure_client()
        img_b64 = self._encode_image_b64(crop)
        prompt = _VLM_PROMPT_TEMPLATE.format(target_text=target_text)

        payload = {
            "model": settings.VLM_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 256,
            "temperature": 0.1,
        }

        try:
            resp = client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return self._parse_response(content)
        except httpx.HTTPStatusError as e:
            logger.warning(f"VLM API error {e.response.status_code}: {e.response.text[:200]}")
            return None
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.warning(f"VLM API connection error: {e}")
            return None
        except Exception as e:
            logger.warning(f"VLM verify_crop unexpected error: {e}")
            return None

    def _parse_response(self, content: str) -> Optional[VLMResult]:
        content = content.strip()
        json_match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
        if not json_match:
            logger.warning(f"VLM response has no JSON: {content[:200]}")
            return None
        try:
            obj = json.loads(json_match.group())
            return VLMResult(
                is_target=bool(obj.get("is_target", False)),
                matched_label=str(obj.get("matched_label", "")),
                confidence=float(obj.get("confidence", 0.0)),
                reason=str(obj.get("reason", "")),
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"VLM JSON parse error: {e}, content: {content[:200]}")
            return None

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


_vlm_instance: Optional[VLMService] = None


def get_vlm_service() -> VLMService:
    global _vlm_instance
    if _vlm_instance is None:
        _vlm_instance = VLMService()
    return _vlm_instance
