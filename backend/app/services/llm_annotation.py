"""
app/services/llm_annotation.py
-------------------------------
大模型自动标注服务 —— 使用通义千问（Qwen）多模态模型自动标注数据集图片。

设计要点：
- 提供商、接口地址等全部在后端硬编码，固定为 Qwen 系列模型（见 app/core/config.py）。
  使用者只需在 .env 中配置 ``LLM_API_KEY``，并按需修改 ``LLM_MODEL`` 即可，
  前端不感知也无法选择。
- 通过阿里云 DashScope 的 **OpenAI 兼容端点** 直接发起 HTTP 请求，使用环境内已自带的
  ``httpx``，**不依赖** openai / anthropic / aiohttp 等第三方 SDK，适配离线整包部署。

标注流程：
1. 用户上传图片到某个类别（annotation_status='pending'）。
2. 创建标注任务，指定提示词（要检测的目标描述）。
3. 异步批量调用 Qwen API，获取目标的像素边界框。
4. 将检测结果转换为 YOLO 归一化格式（class_id cx cy w h）。
5. 保存标注，更新图片状态。
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from PIL import Image

from app.core.config import settings
from app.core.logging import logger

# 后端固定的提供商标识（写入任务记录，仅用于展示/排查）。
PROVIDER = "qwen"

# 已知的多模态模型白名单（支持图片输入）——作为命名规则识别的补充。
KNOWN_MULTIMODAL_MODELS = {
    # Qwen-VL 系列（主力视觉-语言模型）
    "qwen-vl-plus",
    "qwen-vl-plus-latest",
    "qwen-vl-max",
    "qwen-vl-max-latest",
    "qwen-vl-max-0809",
    "qwen-vl-ocr",

    # Qwen2-VL 系列（第二代）
    "qwen2-vl-7b-instruct",
    "qwen2-vl-72b-instruct",

    # Qwen2.5-VL 系列
    "qwen2.5-vl-7b-instruct",
    "qwen2.5-vl-32b-instruct",
    "qwen2.5-vl-72b-instruct",

    # QVQ 系列（视觉问答）
    "qvq-72b-preview",
}

# 命名规则识别视觉模型：qwen*-vl-*、qvq*、*omni*（omni 系列同样接受图片输入）。
# 阿里云上新频繁，白名单枚举跟不上账号实际清单，以规则匹配为主、白名单兜底。
_VL_NAME_PATTERN = re.compile(r"(^|[-_.])(vl|qvq|omni)([-_.\d]|$)", re.IGNORECASE)


def normalize_model_name(model_name: Optional[str] = None) -> str:
    """Return the requested model or the backend default, stripped."""
    return (model_name or settings.LLM_MODEL or "").strip()


def is_multimodal_model(model_name: Optional[str]) -> bool:
    """Check if a model looks image-capable (whitelist or naming pattern)."""
    name = normalize_model_name(model_name).lower()
    return name in KNOWN_MULTIMODAL_MODELS or bool(_VL_NAME_PATTERN.search(name))


def validate_multimodal_model(model_name: Optional[str] = None) -> str:
    """Validate and return the model name (allows any model, runtime will fail if not multimodal)."""
    model = normalize_model_name(model_name)
    if not model:
        raise ValueError("模型名称不能为空")
    return model


# 视觉模型优先级（探测可用性时的尝试顺序；不在账号清单里的不会被尝试）。
_PREFERRED_VL_MODELS = [
    "qwen3-vl-plus",
    "qwen-vl-max-latest",
    "qwen-vl-max",
    "qwen3-vl-flash",
    "qwen-vl-plus-latest",
    "qwen-vl-plus",
    "qwen2.5-vl-72b-instruct",
    "qwen2.5-vl-32b-instruct",
    "qwen2.5-vl-7b-instruct",
]

# 探测成功缓存。API Key/Base 改动必须重启进程才生效，故按进程生命周期缓存即可。
_probe_ok_models: set[str] = set()


async def probe_model(model_name: str) -> Tuple[bool, str]:
    """用一次 1-token 对话请求探测当前 Key 是否真的能调用该模型。

    模型出现在 /models 清单里 ≠ 有调用权限（实测过清单 200 但推理 403
    access_denied 的账号），以一次近乎零成本的真实调用为准。
    """
    url = f"{settings.LLM_API_BASE.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.LLM_API_KEY.strip()}"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
    except Exception as exc:
        return False, f"网络错误: {exc}"
    if resp.status_code == 200:
        _probe_ok_models.add(model_name)
        return True, ""
    detail = resp.text[:200]
    try:
        err = resp.json().get("error", {})
        detail = err.get("code") or err.get("message") or detail
    except Exception:
        pass
    return False, f"HTTP {resp.status_code}: {detail}"


async def resolve_usable_model(requested: Optional[str] = None, max_probes: int = 6) -> str:
    """解析出一个当前 Key 实际可调用的视觉模型。

    顺序：指定/默认模型优先，失败后从账号清单里的视觉模型中按
    ``_PREFERRED_VL_MODELS`` 优先级逐个探测；全部失败抛 ``RuntimeError``，
    错误消息包含逐个失败原因，可直接作为任务 error 展示给用户。
    """
    candidates: List[str] = []
    req = normalize_model_name(requested)
    if req:
        candidates.append(req)
    try:
        account_models = await list_available_models()
    except Exception as exc:
        logger.warning(f"拉取账号模型清单失败，仅尝试指定模型：{exc}")
        account_models = []
    vl_models = [m for m in account_models if is_multimodal_model(m)]
    for name in _PREFERRED_VL_MODELS:
        if name in vl_models and name not in candidates:
            candidates.append(name)
    for name in vl_models:
        if name not in candidates:
            candidates.append(name)
    if not candidates:
        raise RuntimeError("账号模型清单中没有视觉(VL)模型，且未指定模型名。")

    failures: List[str] = []
    for name in candidates[:max_probes]:
        if name in _probe_ok_models:
            return name
        ok, detail = await probe_model(name)
        if ok:
            _probe_ok_models.add(name)
            if req and name != req:
                logger.warning(f"模型 {req} 不可用，自动切换为可用视觉模型 {name}")
            return name
        logger.warning(f"视觉模型探测失败 {name}: {detail}")
        failures.append(f"{name} → {detail}")
    raise RuntimeError(
        "未探测到可调用的视觉模型（Key 有效但对模型无调用权限，请到百炼控制台"
        "确认已开通模型服务且账户有额度）。已尝试: " + "；".join(failures)
    )


@dataclass
class BoundingBox:
    """检测到的目标边界框（归一化坐标）"""
    class_id: int  # 固定为0（单类别训练）
    cx: float      # 中心点x (0-1)
    cy: float      # 中心点y (0-1)
    w: float       # 宽度 (0-1)
    h: float       # 高度 (0-1)
    confidence: float = 1.0

    def to_yolo_line(self) -> str:
        """转换为YOLO格式的标注行"""
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class AnnotationResult:
    """单张图片的标注结果"""
    image_id: str
    boxes: List[BoundingBox]
    success: bool
    error: Optional[str] = None
    raw_response: Optional[str] = None


class LLMAnnotator:
    """大模型标注器基类（统一接口，便于未来扩展）"""

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    async def annotate_image(
        self,
        image_path: str,
        prompt: str,
        image_width: int,
        image_height: int,
    ) -> List[BoundingBox]:
        """标注单张图片 - 子类实现"""
        raise NotImplementedError

    @staticmethod
    def _encode_image(image_path: str, max_size: int = 1024) -> Tuple[str, int, int]:
        """编码图片为base64，并按最长边缩放以节省token。

        返回 (base64字符串, 缩放后宽, 缩放后高)。模型基于缩放后的图片返回像素坐标，
        因此后续按缩放后的尺寸归一化即可。
        """
        with Image.open(image_path) as img:
            # 保持宽高比缩放（thumbnail 只缩不放）
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            # 转换为RGB（去除alpha通道）
            if img.mode != "RGB":
                img = img.convert("RGB")

            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)

            b64 = base64.b64encode(buffer.read()).decode("utf-8")
            return b64, img.width, img.height


def _extract_json_object(text: str) -> Optional[dict]:
    """从模型输出中尽力提取一个 JSON 对象。

    依次尝试：直接解析 → 去除 ```json 围栏后解析 → 截取首个 '{' 到末个 '}' 的子串解析。
    全部失败返回 None（与「解析成功但为空」区分开）。
    """
    candidates: List[str] = []
    raw = (text or "").strip()
    if raw:
        candidates.append(raw)

    # 去除可能的 ```json ... ``` 围栏
    if "```" in raw:
        for seg in raw.split("```"):
            seg = seg.strip()
            if seg.lower().startswith("json"):
                seg = seg[len("json"):].strip()
            if seg:
                candidates.append(seg)

    # 截取首个 '{' 到末个 '}'（容忍前后散落的说明文字）
    lo, hi = raw.find("{"), raw.rfind("}")
    if lo != -1 and hi != -1 and hi > lo:
        json_str = raw[lo : hi + 1]
        # 尝试修复缺少字段名的格式：{"x1": 580, 346, 612, 410} -> {"x1": 580, "y1": 346, "x2": 612, "y2": 410}
        json_str = _fix_incomplete_bbox_json(json_str)
        candidates.append(json_str)

    for cand in candidates:
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _fix_incomplete_bbox_json(json_str: str) -> str:
    """尝试修复模型输出的不完整边界框 JSON。

    修复多种格式错误：
    1. {"x1": 580, 346, 612, 410}
    2. {"x1": 450, 362, "y1": 387, "x2": 540, "y2": 402}
    3. {"x1": 452, 926, "y1": 487, "x2": 488, 950}  <- y2 缺少引号
    4. 跨行的混合格式
    """
    import re

    # 先移除多余的换行和空格，便于匹配
    normalized = re.sub(r'\s+', ' ', json_str)

    # 修复缺少引号的字段名（如 "x2": 488, 950 应该是 "x2": 488, "y2": 950）
    # 在任何字段名后如果有 <数字>, <数字> 且第二个数字后不是逗号或}，则第二个数字前应该加字段名
    # 更简单的方法：在对象内部，如果看到 : <数字>, <数字> 且第二个数字不在引号内，补充字段名

    # 模式：匹配整个对象，然后逐个修复
    # {"x1": A, B, "y1": C, "x2": D, E} -> {"x1": A, "y1": B, "x2": C, "y2": D}
    # 但这种模式太复杂，改用更通用的方法

    # 策略：找到所有 "字段": 数字, 数字 模式，补充下一个字段名
    field_order = ["x1", "y1", "x2", "y2"]

    # 先处理 "x1": A, B, "y1": ... 的情况（B应该成为y1）
    pattern_x1 = r'("x1"\s*:\s*)(\d+)\s*,\s*(\d+)\s*,\s*("y1")'
    normalized = re.sub(pattern_x1, r'\1\2, "y1": \3, \4', normalized)

    # 处理 "y1": A, "x2": B, C 的情况（C应该成为y2）
    pattern_x2 = r'("x2"\s*:\s*)(\d+)\s*,\s*(\d+)(\s*[,}])'
    normalized = re.sub(pattern_x2, r'\1\2, "y2": \3\4', normalized)

    # 处理完全缺少字段名的情况：{"x1": A, B, C, D}
    pattern_all_missing = r'("x1"\s*:\s*)(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(\s*[,}])'
    normalized = re.sub(pattern_all_missing, r'\1\2, "y1": \3, "x2": \4, "y2": \5\6', normalized)

    if normalized != json_str:
        # 再次规范化空格
        normalized = re.sub(r'\s+', ' ', normalized)
        logger.info(f"修复了不完整的边界框 JSON 格式")

    return normalized


def _parse_objects_response(
    response_text: str,
    resized_w: int,
    resized_h: int,
) -> List[BoundingBox]:
    """解析模型返回的JSON，转换为YOLO格式边界框。

    期望格式：{"objects": [{"x1": int, "y1": int, "x2": int, "y2": int}, ...]}
    坐标为相对于送入模型图片（缩放后 resized_w×resized_h）的像素值。
    对被 markdown 代码块包裹、或前后夹带说明文字的 JSON 做了容错处理。
    """
    data = _extract_json_object(response_text)
    if data is None:
        # JSON 解析失败：检查响应中是否包含坐标数字，判断是"没找到目标"还是"格式错误"
        has_coords = bool(re.search(r'\d{2,}', response_text))  # 至少2位数字（像素坐标）

        if has_coords:
            # 响应中有数字，可能是格式错误导致解析失败
            logger.error(f"Qwen响应包含坐标但JSON解析失败（格式错误）\n响应内容: {response_text}")
            raise RuntimeError(
                f"模型返回了坐标数据但格式无法解析。请检查模型输出格式是否正确。\n"
                f"响应片段: {response_text[:200]}"
            )
        else:
            # 响应中没有明显的坐标数字，可能是模型说"没找到目标"
            logger.warning(f"Qwen响应JSON解析失败，可能未找到目标\n响应内容: {response_text}")
            return []

    try:
        objects = data.get("objects", [])

        boxes: List[BoundingBox] = []
        for obj in objects:
            try:
                # 标准格式：所有字段都存在
                x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
            except (KeyError, TypeError):
                # 容错：尝试从对象的所有值中提取数值
                try:
                    values = []
                    # 先尝试按标准字段名提取
                    if "x1" in obj:
                        values.append(obj["x1"])
                    # 收集所有数值
                    for v in obj.values():
                        if isinstance(v, (int, float)):
                            values.append(v)

                    # 去重并保持顺序
                    seen = set()
                    unique_values = []
                    for v in values:
                        if v not in seen:
                            seen.add(v)
                            unique_values.append(v)

                    # 如果恰好有4个数值，假设是 x1, y1, x2, y2
                    if len(unique_values) == 4:
                        x1, y1, x2, y2 = unique_values
                        logger.warning(f"容错解析边界框（模型可能省略了字段名）: {obj} -> [{x1}, {y1}, {x2}, {y2}]")
                    else:
                        continue
                except Exception:
                    continue

            # 确保左上 < 右下
            x_lo, x_hi = sorted((float(x1), float(x2)))
            y_lo, y_hi = sorted((float(y1), float(y2)))

            # 归一化到0-1（相对于缩放后的图片）
            x1_norm = max(0.0, min(1.0, x_lo / resized_w))
            y1_norm = max(0.0, min(1.0, y_lo / resized_h))
            x2_norm = max(0.0, min(1.0, x_hi / resized_w))
            y2_norm = max(0.0, min(1.0, y_hi / resized_h))

            # 转换为YOLO格式（cx, cy, w, h）
            cx = (x1_norm + x2_norm) / 2.0
            cy = (y1_norm + y2_norm) / 2.0
            w = x2_norm - x1_norm
            h = y2_norm - y1_norm

            if w > 0 and h > 0:  # 有效框
                boxes.append(BoundingBox(class_id=0, cx=cx, cy=cy, w=w, h=h))

        return boxes

    except Exception as e:
        logger.error(f"解析Qwen响应时出错: {e}")
        return []


class QwenAnnotator(LLMAnnotator):
    """通义千问（Qwen）多模态标注器。

    通过 DashScope 的 OpenAI 兼容端点 ``/chat/completions`` 发起请求，
    payload 与 OpenAI Vision 一致（image_url 使用 base64 data URL）。
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        api_base: str,
        max_size: int = 1024,
        timeout: float = 120.0,
    ):
        super().__init__(api_key, model_name)
        self.api_base = api_base.rstrip("/")
        self.max_size = max_size
        self.timeout = timeout

    async def annotate_image(
        self,
        image_path: str,
        prompt: str,
        image_width: int,
        image_height: int,
    ) -> List[BoundingBox]:
        b64_image, resized_w, resized_h = self._encode_image(image_path, self.max_size)

        system_prompt = f"""你是一个专业的目标检测标注助手。用户会给你一张图片和目标描述，你需要：

1. 仔细观察图片中是否存在用户描述的目标
2. 如果存在，标注所有匹配目标的边界框
3. 严格以JSON格式返回结果，不要输出任何额外文字

返回格式（JSON）：
{{
  "objects": [
    {{"x1": <左上角x像素>, "y1": <左上角y像素>, "x2": <右下角x像素>, "y2": <右下角y像素>}}
  ]
}}

如果图片中没有目标，返回：{{"objects": []}}

**关键要求**：
- 每个坐标必须带字段名：x1, y1, x2, y2（不能省略字段名）
- 坐标为像素值：x 在 0-{resized_w} 之间，y 在 0-{resized_h} 之间
- 边界框要紧贴目标，不要留太多空白
- 只标注清晰可见的完整目标

正确示例：{{"x1": 100, "y1": 50, "x2": 200, "y2": 150}}
错误示例：{{"x1": 100, 50, 200, 150}} ← 缺少字段名"""

        user_message = f"请在图片中标注所有的：{prompt}"

        # 某些模型（如 z-image-turbo）可能不支持 system 消息或有特殊格式要求
        # 尝试构建兼容的 payload
        has_system_support = not any(
            marker in self.model_name.lower()
            for marker in ["z-image", "turbo", "mini"]
        )

        if has_system_support:
            # 标准格式：system + user
            payload: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                },
                            },
                            {"type": "text", "text": user_message},
                        ],
                    },
                ],
                "max_tokens": 2048,
                "temperature": 0.1,
            }
        else:
            # 简化格式：仅 user，将 system 合并到 user 消息中
            combined_message = f"{system_prompt}\n\n{user_message}"
            payload: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                },
                            },
                            {"type": "text", "text": combined_message},
                        ],
                    },
                ],
                "max_tokens": 2048,
                "temperature": 0.1,
            }

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 记录请求信息（不含 base64 图片）
        logger.debug(
            f"调用 Qwen API: model={self.model_name}, "
            f"prompt='{prompt[:50]}...', image_size={image_width}x{image_height}"
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    error_text = resp.text[:500]
                    error_msg = f"Qwen API错误 [{resp.status_code}]: {error_text}"

                    # 针对常见错误给出具体建议
                    if resp.status_code == 400:
                        if "invalid" in error_text.lower() or "unexpected" in error_text.lower():
                            error_msg += (
                                f"\n提示：模型 {self.model_name} 可能不支持图片输入（非多模态模型），"
                                "请尝试切换为 qwen-vl-plus、qwen-vl-max 或其他 VL/Vision 系列模型。"
                            )

                    raise RuntimeError(error_msg)
                data = resp.json()

            text_content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            # 个别兼容实现会把 content 拆成分块列表
            if isinstance(text_content, list):
                text_content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in text_content
                )

            logger.debug(f"Qwen响应: {text_content}")
            return _parse_objects_response(text_content, resized_w, resized_h)

        except Exception as e:
            logger.error(f"Qwen标注失败 [{image_path}]: {e}")
            raise


def create_annotator(model_name: Optional[str] = None) -> LLMAnnotator:
    """工厂函数：依据后端配置创建 Qwen 标注器。

    API Base / Key 仍来自后端配置；调用方可传入经校验的多模态模型名，用于
    单次标注任务切换模型。未传入时使用 settings.LLM_MODEL。
    """
    if not settings.llm_configured:
        raise RuntimeError(
            "未配置 LLM_API_KEY，无法使用大模型标注。请在后端 .env 中设置 LLM_API_KEY。"
        )
    selected_model = validate_multimodal_model(model_name)
    return QwenAnnotator(
        api_key=settings.LLM_API_KEY.strip(),
        model_name=selected_model,
        api_base=settings.LLM_API_BASE,
        max_size=settings.LLM_IMAGE_MAX_SIZE,
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )


async def iter_annotate_images_batch(
    annotator: LLMAnnotator,
    images: List[Dict[str, Any]],
    prompt: str,
    category_id: str,
    max_concurrent: int = 3,
) -> AsyncIterator[AnnotationResult]:
    """Yield each image's annotation result as soon as it finishes."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _annotate_one(img: Dict[str, Any]) -> AnnotationResult:
        async with semaphore:
            max_retries = 3
            base_delay = 2.0  # 基础延迟（秒）

            for attempt in range(max_retries):
                try:
                    boxes = await annotator.annotate_image(
                        image_path=img["stored_path"],
                        prompt=prompt,
                        image_width=img["width"],
                        image_height=img["height"],
                    )
                    return AnnotationResult(
                        image_id=img["id"],
                        boxes=boxes,
                        success=True,
                    )
                except Exception as e:
                    error_str = str(e)
                    # 检查是否为 429 限流错误
                    is_rate_limit = "429" in error_str or "limit_requests" in error_str

                    if is_rate_limit and attempt < max_retries - 1:
                        # 指数退避：2s, 4s, 8s
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"图片 {img['id']} 遇到限流（尝试 {attempt + 1}/{max_retries}），"
                            f"等待 {delay:.1f}s 后重试..."
                        )
                        await asyncio.sleep(delay)
                        continue

                    # 非限流错误或已达最大重试次数
                    logger.error(f"标注图片失败 {img['id']}: {e}")
                    return AnnotationResult(
                        image_id=img["id"],
                        boxes=[],
                        success=False,
                        error=error_str,
                    )

    tasks = [asyncio.create_task(_annotate_one(img)) for img in images]
    try:
        for done in asyncio.as_completed(tasks):
            yield await done
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()


async def annotate_images_batch(
    annotator: LLMAnnotator,
    images: List[Dict[str, Any]],
    prompt: str,
    category_id: str,
    max_concurrent: int = 3,
) -> List[AnnotationResult]:
    """
    批量标注图片

    Args:
        annotator: LLM标注器
        images: 图片列表 [{"id": str, "stored_path": str, "width": int, "height": int}, ...]
        prompt: 用户提示词
        category_id: 类别ID
        max_concurrent: 最大并发请求数

    Returns:
        标注结果列表
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _annotate_one(img: Dict[str, Any]) -> AnnotationResult:
        async with semaphore:
            max_retries = 3
            base_delay = 2.0  # 基础延迟（秒）

            for attempt in range(max_retries):
                try:
                    boxes = await annotator.annotate_image(
                        image_path=img["stored_path"],
                        prompt=prompt,
                        image_width=img["width"],
                        image_height=img["height"],
                    )
                    return AnnotationResult(
                        image_id=img["id"],
                        boxes=boxes,
                        success=True,
                    )
                except Exception as e:
                    error_str = str(e)
                    # 检查是否为 429 限流错误
                    is_rate_limit = "429" in error_str or "limit_requests" in error_str

                    if is_rate_limit and attempt < max_retries - 1:
                        # 指数退避：2s, 4s, 8s
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"图片 {img['id']} 遇到限流（尝试 {attempt + 1}/{max_retries}），"
                            f"等待 {delay:.1f}s 后重试..."
                        )
                        await asyncio.sleep(delay)
                        continue

                    # 非限流错误或已达最大重试次数
                    logger.error(f"标注图片失败 {img['id']}: {e}")
                    return AnnotationResult(
                        image_id=img["id"],
                        boxes=[],
                        success=False,
                        error=error_str,
                    )

    results = []
    async for result in iter_annotate_images_batch(
        annotator=annotator,
        images=images,
        prompt=prompt,
        category_id=category_id,
        max_concurrent=max_concurrent,
    ):
        results.append(result)
    return results


def save_annotation_to_disk(
    category_id: str,
    image_id: str,
    boxes: List[BoundingBox],
) -> Path:
    """将标注结果保存为YOLO格式txt文件"""
    ann_path = settings.ANNOTATIONS_DIR / category_id / f"{image_id}.txt"
    ann_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [box.to_yolo_line() for box in boxes]
    ann_path.write_text("\n".join(lines), encoding="utf-8")

    return ann_path


# ── 目标名词提取（用于一站式自动标注自动命名数据集类别）──────────────────────────
# 用户可输入自然语言（"帮我标注出河流中的船只"）或直接名词（"船只"）。这里把它
# 归一成一个简短的类别名（"船只"）。优先用本地启发式（零成本）；自然语言短语再调
# Qwen 纯文本接口提取，避免依赖可能未启动的本地 VLM。

# 常见指令性/修饰性词，启发式剥离后若剩单个短名词即可直接用作类别名。
_INSTRUCTION_WORDS = [
    "帮我", "帮忙", "请", "麻烦", "给我", "我想", "我要", "需要",
    "标注", "标记", "检测", "识别", "框出", "框选", "找出", "找到", "圈出",
    "所有", "全部", "这些", "那些", "里面", "图片", "图像", "视频", "画面",
    "图中", "画面中", "视频中", "里的", "中的", "出来", "出", "把", "一下", "目标", "的",
]
# 出现这些词即认为是"句子/指令"而非直接名词，需要走模型提取。
# 含结构助词"的"（如"低空的小型无人机"）也按短语处理，交给模型提取核心名词。
_PHRASE_MARKERS = ("帮", "请", "标注", "标记", "检测", "识别", "框", "找", "圈", "把", "中的", "里的", "我想", "我要", "的")
# 类别名最大长度（留足余量给 _时间戳后缀，避免超出 CategoryRecord.name 的 128 限制）。
_MAX_NOUN_LEN = 32


def _looks_like_simple_noun(raw: str) -> bool:
    """短、无指令词、无分隔符 → 视为用户直接给的目标名词，免去一次模型调用。"""
    s = raw.strip()
    if not s or len(s) > 8:
        return False
    if any(m in s for m in _PHRASE_MARKERS):
        return False
    if re.search(r"[,，、;；\s]", s):
        return False
    return True


def _heuristic_noun(raw: str) -> str:
    """启发式：剥离指令/修饰词，返回剩余文本（兜底用）。"""
    s = raw.strip()
    for w in _INSTRUCTION_WORDS:
        s = s.replace(w, "")
    return s.strip(" \t，,。.、;；:：\"'“”")


def _clean_noun(text: str) -> str:
    """清洗模型返回：取首行、去首尾标点/引号；过长则视为无效。"""
    if not text:
        return ""
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    first = re.sub(r'^[\s"\'“”‘’`：:，,。.\-]+|[\s"\'“”‘’`：:，,。.\-]+$', "", first)
    return "" if len(first) > 20 else first


async def _qwen_extract_noun(raw: str) -> str:
    """调用 Qwen 纯文本接口，从一句话里提取要检测的核心目标名词。"""
    if not settings.llm_configured:
        return ""
    payload: Dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你从用户的一句话里提取出他要检测/标注的核心目标名词。"
                    "只返回这个名词本身，要尽量简短，不要任何解释、标点或多余文字。"
                    "例如输入“帮我标注出河流中的船只”，输出“船只”；输入“低空的小型无人机”，输出“无人机”。"
                ),
            },
            {"role": "user", "content": raw},
        ],
        "max_tokens": 32,
        "temperature": 0.0,
    }
    url = f"{settings.LLM_API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.LLM_API_KEY.strip()}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Qwen 文本接口错误 [{resp.status_code}]: {resp.text[:200]}")
        data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return _clean_noun(content)


async def list_available_models() -> List[str]:
    """Fetch the model IDs available to the configured DashScope API key.

    DashScope exposes an OpenAI-compatible base URL, so model discovery uses the
    standard ``GET {base_url}/models`` shape and returns only model IDs to the
    frontend. API keys and provider-specific raw payloads are never exposed.
    """
    if not settings.llm_configured:
        raise RuntimeError("后端未配置 LLM_API_KEY。")

    url = f"{settings.LLM_API_BASE.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {settings.LLM_API_KEY.strip()}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Qwen 模型列表接口错误 [{resp.status_code}]: {resp.text[:300]}")
        data = resp.json()

    models: List[str] = []
    for item in data.get("data", []) if isinstance(data, dict) else []:
        if isinstance(item, dict) and item.get("id"):
            models.append(str(item["id"]))
    return sorted(set(models), key=str.lower)


def filter_multimodal_models(models: List[str]) -> List[str]:
    """Keep only model IDs that look image-capable for annotation UI choices."""
    return [model for model in models if is_multimodal_model(model)]


async def extract_target_noun(raw: str) -> str:
    """把用户输入归一成简短的类别名。

    流程：直接名词 → 原样返回；自然语言短语 → Qwen 提取；失败 → 启发式剥词 → 原文。
    返回空字符串表示无法提取（调用方应据此报错）。
    """
    raw = (raw or "").strip()
    if not raw:
        return ""
    if _looks_like_simple_noun(raw):
        return raw[:_MAX_NOUN_LEN]
    try:
        noun = await _qwen_extract_noun(raw)
        if noun:
            return noun[:_MAX_NOUN_LEN]
    except Exception as exc:
        logger.warning(f"Qwen 提取目标名词失败，回退启发式：{exc}")
    return (_heuristic_noun(raw) or raw)[:_MAX_NOUN_LEN]

