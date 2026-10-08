"""
app/services/prompt_normalizer.py
-----------------------------------
Converts user's natural language input (often Chinese) into
Grounding DINO-friendly English phrases via the local LLM API.

Also preserves the original input for MiniCPM-V verification prompts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

from app.core.config import settings
from app.core.logging import logger


@dataclass
class NormalizedPrompt:
    raw_query: str
    dino_prompt: str
    vlm_query: str
    targets: List[dict] = field(default_factory=list)
    target_type: str = "common"  # common | small | rare
    suggested_candidate_threshold: float = 0.30
    suggested_confirm_threshold: float = 0.55
    need_vlm_verification: bool = True
    visual_features: List[str] = field(default_factory=list)
    color_filters: List[dict] = field(default_factory=list)  # Color filtering rules


_NORMALIZE_SYSTEM = (
    "你是一个目标检测辅助工具。用户会输入想要检测的目标描述（可能是中文），"
    "你需要将其转换为 Grounding DINO 模型能理解的英文短语，并提供目标理解信息。\n\n"
    "**核心规则**：\n"
    "1. 提取所有具体目标名词，忽略模糊或无关的描述词\n"
    "2. 每个目标翻译为英文主名称，并添加2-3个同义词或相似概念\n"
    "3. 添加外观特征描述（形状、颜色、纹理、位置等）\n"
    "4. **关键**：添加负面排除短语，明确说明「不是什么」，避免误检\n"
    "   - 例如：菜地 → 'not building roofs, not concrete ground, not residential area'\n"
    "   - 例如：水塘 → 'not swimming pool, not roof reflection, not road surface'\n"
    "   - 例如：太阳能板 → 'not blue roof tiles, not water surface, not glass windows'\n"
    "5. 判断目标类型：common（常见）、small（小目标<5%画面）、rare（小众/罕见）\n"
    "6. **重要**：dino_prompt 只包含每个目标的主要英文名称，用句号分隔\n"
    "   - 格式：'target1. target2. target3.'\n"
    "   - 不要在 dino_prompt 中包含同义词、描述词或负面短语\n"
    "   - 同义词和描述放在 targets[].en 数组中供参考\n"
    "7. 如果是无人机/俯视场景，考虑俯视角度的视觉特征\n"
    "8. **颜色特征**：为每个目标提供主要颜色范围（HSV），用于后处理降权\n"
    "   - 格式：{\"name\": \"颜色名\", \"h_range\": [h_min, h_max], \"s_range\": [s_min, s_max], \"v_range\": [v_min, v_max], \"min_ratio\": 最小占比}\n"
    "   - H范围: 0-180 (OpenCV), S/V范围: 0-255\n"
    "   - 可以提供多个颜色范围（如绿色+棕色的菜地）\n"
    "   - min_ratio: 该颜色在裁剪区域中的最小占比（0.0-1.0）\n\n"
    "**用户输入修正**：\n"
    "- 如果用户输入过于模糊（如\"那个东西\"），要求提供更具体的描述\n"
    "- 如果用户输入包含多个不相关目标，拆分为独立目标\n"
    "- 如果用户输入包含场景描述（如\"水面上的\"），提取核心目标并添加场景约束\n"
    "- 如果用户输入是动作描述（如\"在飞的鸟\"），转换为目标名词（\"bird\"）\n\n"
    "颜色参考（OpenCV HSV）：\n"
    "- 绿色（植被）: H=[35-85], S=[40-255], V=[40-255]\n"
    "- 棕色/土色: H=[10-25], S=[30-200], V=[20-150]\n"
    "- 蓝色（水体）: H=[100-130], S=[50-255], V=[50-255]\n"
    "- 灰色/白色（建筑）: S=[0-50], V=[100-255]\n"
    "- 黄色: H=[20-35], S=[50-255], V=[100-255]\n"
    "- 深蓝/黑色（太阳能板）: H=[100-130], S=[30-255], V=[20-100]\n\n"
    "提示词示例：\n"
    "- 输入：菜地\n"
    "  输出：dino_prompt='vegetable field.', targets=[{\"zh\":\"菜地\", \"en\":[\"vegetable field\",\"farmland\",\"cultivated land\"], \"color_filters\":[{\"name\":\"green\",\"h_range\":[35,85],\"s_range\":[40,255],\"v_range\":[40,255],\"min_ratio\":0.25}]}]\n"
    "- 输入：菜地,水塘\n"
    "  输出：dino_prompt='vegetable field. water pond.', targets=[{\"zh\":\"菜地\", \"en\":[...]}, {\"zh\":\"水塘\", \"en\":[...]}]\n"
    "- 输入：太阳能板\n"
    "  输出：dino_prompt='solar panel.', targets=[{\"zh\":\"太阳能板\", \"en\":[\"solar panel\",\"photovoltaic panel\"], \"color_filters\":[{\"name\":\"dark_blue\",\"h_range\":[100,130],\"s_range\":[30,255],\"v_range\":[20,100],\"min_ratio\":0.4}]}]\n\n"
    "只输出 JSON，格式如下：\n"
    '{"targets": [{"zh": "中文名", "en": ["main_name", "synonym1", "synonym2"], '
    '"color_filters": [{"name": "color_name", "h_range": [h_min, h_max], "s_range": [s_min, s_max], "v_range": [v_min, v_max], "min_ratio": 0.0-1.0}]}], '
    '"dino_prompt": "target1. target2.", '
    '"target_type": "common|small|rare", '
    '"visual_features": ["feature1", "feature2"], '
    '"vlm_description": "用于 VLM 复核的中文描述（更详细、包含场景和特征）"}'
)


def normalize_prompt(raw_input: str) -> NormalizedPrompt:
    raw_input = raw_input.strip()
    if not raw_input:
        return NormalizedPrompt(raw_query=raw_input, dino_prompt="", vlm_query=raw_input)

    if _is_already_dino_friendly(raw_input):
        dino_prompt = raw_input.lower()
        if not dino_prompt.endswith("."):
            dino_prompt += "."
        return NormalizedPrompt(
            raw_query=raw_input,
            dino_prompt=dino_prompt,
            vlm_query=raw_input,
        )

    llm_result = _call_llm_normalize(raw_input)
    if llm_result is not None:
        return llm_result

    return _fallback_normalize(raw_input)


def _is_already_dino_friendly(text: str) -> bool:
    has_cjk = bool(re.search(r'[一-鿿㐀-䶿]', text))
    if has_cjk:
        return False
    words = text.split()
    return len(words) <= 20 and all(w.isascii() for w in words)


def _call_llm_normalize(raw_input: str) -> Optional[NormalizedPrompt]:
    try:
        client = httpx.Client(
            base_url=settings.VLM_API_BASE,
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        resp = client.post(
            "/chat/completions",
            json={
                "model": settings.VLM_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": _NORMALIZE_SYSTEM},
                    {"role": "user", "content": raw_input},
                ],
                "max_tokens": 512,
                "temperature": 0.1,
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        client.close()

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            return None
        obj = json.loads(json_match.group())
        dino_prompt = obj.get("dino_prompt", "")
        targets = obj.get("targets", [])
        target_type = obj.get("target_type", "common")
        visual_features = obj.get("visual_features", [])
        vlm_description = obj.get("vlm_description", raw_input)

        # Extract color filters from targets
        color_filters = []
        for target in targets:
            target_filters = target.get("color_filters", [])
            if target_filters:
                color_filters.extend(target_filters)

        if not dino_prompt:
            return None

        # Set thresholds based on target type
        if target_type == "small":
            candidate_thr = 0.25
            confirm_thr = 0.55
        elif target_type == "rare":
            candidate_thr = 0.25
            confirm_thr = 0.60
        else:  # common
            candidate_thr = 0.30
            confirm_thr = 0.55

        return NormalizedPrompt(
            raw_query=raw_input,
            dino_prompt=dino_prompt.lower(),
            vlm_query=vlm_description,
            targets=targets,
            target_type=target_type,
            suggested_candidate_threshold=candidate_thr,
            suggested_confirm_threshold=confirm_thr,
            need_vlm_verification=target_type in ("small", "rare"),
            visual_features=visual_features,
            color_filters=color_filters,
        )
    except Exception as e:
        logger.warning(f"LLM prompt normalization failed: {e}, using fallback")
        return None


_ZH_EN_MAP = {
    "菜地": ("vegetable field", "common"),
    "菜园": ("vegetable garden", "common"),
    "钓鱼台": ("fishing platform", "rare"),
    "小船": ("small boat", "small"),
    "车辆": ("vehicle", "common"),
    "房屋": ("house", "common"),
    "人": ("person", "common"),
    "树": ("tree", "common"),
    "道路": ("road", "common"),
    "水塘": ("water pond", "common"),
    "鱼塘": ("fish pond", "common"),
    "大棚": ("greenhouse", "common"),
    "太阳能板": ("solar panel", "common"),
    "违建": ("illegal building", "common"),
    "虾塘增氧机": ("pond aerator", "rare"),
    "增氧机": ("aerator", "small"),
    "鱼排": ("fish cage", "rare"),
    "泵房": ("pump house", "rare"),
    "地笼": ("fish trap", "rare"),
    "小型浮台": ("floating platform", "small"),
    "围网": ("fishing net", "small"),
}


def _fallback_normalize(raw_input: str) -> NormalizedPrompt:
    parts = re.split(r'[,，、;；\s]+', raw_input)
    en_phrases = []
    targets = []
    target_type = "common"

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in _ZH_EN_MAP:
            phrase, ttype = _ZH_EN_MAP[part]
            en_phrases.append(phrase + ".")
            targets.append({"zh": part, "en": [phrase]})
            if ttype in ("rare", "small"):
                target_type = ttype
        else:
            en_phrases.append(part.lower() + ".")
            targets.append({"zh": part, "en": [part.lower()]})

    dino_prompt = " ".join(en_phrases) if en_phrases else raw_input.lower() + "."

    # Set thresholds based on target type
    if target_type == "small":
        candidate_thr = 0.25
        confirm_thr = 0.55
    elif target_type == "rare":
        candidate_thr = 0.25
        confirm_thr = 0.60
    else:
        candidate_thr = 0.30
        confirm_thr = 0.55

    return NormalizedPrompt(
        raw_query=raw_input,
        dino_prompt=dino_prompt,
        vlm_query=raw_input,
        targets=targets,
        target_type=target_type,
        suggested_candidate_threshold=candidate_thr,
        suggested_confirm_threshold=confirm_thr,
        need_vlm_verification=target_type in ("small", "rare"),
    )
