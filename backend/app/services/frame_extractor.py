"""
app/services/frame_extractor.py
--------------------------------
从上传的视频里抽帧，作为某个类别数据集的待标注图片。

设计：
- 三档密度（sparse / medium / dense）对应不同抽帧间隔；每档有总帧数上限
  （见 settings.VIDEO_FRAME_MAX_*），按视频时长在上限内**均匀降采样**——不是只截
  视频前段，长视频也能覆盖全程。
- 用 ``cap.grab()`` 快速跳帧、命中步长才 ``cap.retrieve()`` 解码，避免逐帧 decode。
- 纯磁盘 IO、不写数据库；返回每帧的落盘信息，由调用方批量建 DatasetImageRecord。
- 这是 CPU/IO 密集的同步函数，调用方应放进 ``asyncio.to_thread`` 执行，避免阻塞事件循环。
"""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Dict, List

import cv2

from app.core.config import settings
from app.core.logging import logger
from app.utils.video_utils import get_video_info

# 密度档 → 抽帧间隔（秒）。
_DENSITY_INTERVAL_SEC: Dict[str, float] = {
    "sparse": 3.0,
    "medium": 1.0,
    "dense": 1.0 / 3.0,
}


def density_max_frames(density: str) -> int:
    """该密度档允许的最大抽帧数（控制付费标注调用总量）。"""
    return {
        "sparse": settings.VIDEO_FRAME_MAX_SPARSE,
        "medium": settings.VIDEO_FRAME_MAX_MEDIUM,
        "dense": settings.VIDEO_FRAME_MAX_DENSE,
    }.get(density, settings.VIDEO_FRAME_MAX_MEDIUM)


def extract_frames(
    video_path: Path,
    out_dir: Path,
    density: str = "medium",
    max_frames: int | None = None,
) -> List[Dict]:
    """从视频抽帧并保存为 JPEG。

    :param video_path: 视频文件路径
    :param out_dir: 帧图片保存目录（调用方负责确保可写）
    :param density: sparse | medium | dense
    :param max_frames: 总帧数上限；None 时取该密度档默认上限
    :returns: ``[{"image_id","stored_path","filename","width","height"}, ...]``
    :raises RuntimeError: 视频无法打开或未抽到任何帧
    """
    interval_sec = _DENSITY_INTERVAL_SEC.get(density, _DENSITY_INTERVAL_SEC["medium"])
    cap_limit = max(1, max_frames if max_frames is not None else density_max_frames(density))

    info = get_video_info(video_path)
    fps = info.get("fps") or 30.0
    # 部分容器（可变帧率 / webm）会报告异常 fps（如 90000），导致步长爆炸、整段几乎只抽
    # 到 1 帧；钳制到合理范围，异常时回退 30。
    if not (0 < fps <= 240):
        logger.warning(f"视频 fps 异常（{fps}），回退为 30 计算抽帧间隔：{video_path.name}")
        fps = 30.0
    total_frames = info.get("total_frames") or 0
    duration = info.get("duration_seconds") or 0.0
    # 元数据缺失（total=0，流式/部分 webm 常见）时用时长估算总帧数，保证"上限均匀降采样"
    # 仍能覆盖全程，而不是只抽视频前段。
    if total_frames <= 0 and duration > 0:
        total_frames = int(duration * fps)

    # 基础步长：按密度间隔换算帧数；再叠加"上限均匀降采样"步长，取较大者，
    # 保证最终抽帧数 ≤ cap_limit 且覆盖整段视频。
    step = max(1, round(fps * interval_sec))
    if total_frames > 0:
        step = max(step, math.ceil(total_frames / cap_limit))

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    saved: List[Dict] = []
    frame_idx = 0
    quality = [cv2.IMWRITE_JPEG_QUALITY, int(settings.JPEG_QUALITY)]
    try:
        while len(saved) < cap_limit:
            # grab() 只取帧不解码；命中步长时才 retrieve() 解码，省 CPU。
            if not cap.grab():
                break  # 到视频末尾
            if frame_idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    frame_idx += 1
                    continue
                image_id = uuid.uuid4().hex
                filename = f"{image_id}.jpg"
                dest = out_dir / filename
                try:
                    if not cv2.imwrite(str(dest), frame, quality):
                        logger.warning(f"抽帧写入失败（跳过）：{dest}")
                        dest.unlink(missing_ok=True)  # 清掉可能的半截/空文件
                        frame_idx += 1
                        continue
                except Exception as exc:  # 坏帧/编码异常，跳过不中断
                    logger.warning(f"抽帧编码异常（跳过）frame={frame_idx}: {exc}")
                    frame_idx += 1
                    continue
                h, w = frame.shape[:2]
                saved.append({
                    "image_id": image_id,
                    "stored_path": str(dest.resolve()),
                    "filename": filename,
                    "width": int(w),
                    "height": int(h),
                })
            frame_idx += 1
    finally:
        cap.release()

    if not saved:
        raise RuntimeError("未能从视频中抽取到任何有效帧。")

    logger.info(
        f"视频抽帧完成：{video_path.name} → {len(saved)} 帧"
        f"（密度={density}, 步长={step}, 源帧数={total_frames}, fps={fps:.1f}）"
    )
    return saved
