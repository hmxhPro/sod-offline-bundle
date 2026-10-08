"""
app/services/dataset_builder.py
-------------------------------
Materialize a frozen YOLO dataset for a training job from a category's
uploaded images + their working annotations.

Layout produced (REQ2 — the annotated dataset is stored SEPARATELY from the
raw uploads, frozen per training job):

    datasets/<category_id>/raw/<image_id>.<ext>        # raw uploads (input)
    annotations/<category_id>/<image_id>.txt           # working labels (input)
    datasets/<category_id>/yolo/<job_id>/              # frozen output
        images/{train,val}/<image_id>.<ext>
        labels/{train,val}/<image_id>.txt
        dataset.yaml

Phase 1 is single-class per category: every box is class 0 and
`names: {0: <category name>}`.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import List, Optional

import yaml  # provided by ultralytics' dependency set

from app.core.config import settings
from app.core.logging import logger
from app.utils.naming import strip_timestamp_suffix


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def _polygon_to_box(coords: list[float]) -> list[float]:
    """Enclosing axis-aligned YOLO box (normalized cx,cy,w,h) of a flat polygon
    [x1,y1, x2,y2, ...]; coords clamped to [0,1]."""
    xs = [_clamp01(coords[i]) for i in range(0, len(coords), 2)]
    ys = [_clamp01(coords[i]) for i in range(1, len(coords), 2)]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]


def _read_label(category_id: str, image_id: str) -> str:
    """Return YOLO **detection** label text for one image, class forced to 0
    (Phase 1 single-class).

    The training base is a plain YOLOv11 detection model, so labels are plain
    boxes ``0 cx cy w h`` (normalized). Each working annotation line is
    normalized to a box:

      * box line     `c cx cy w h`          → kept as-is (class → 0)
      * polygon line `c x1 y1 x2 y2 ...`    → enclosing axis-aligned box

    (The detection backend only ever reads boxes — image_detector reads r.boxes.)
    Missing/empty file → "" (a valid background sample)."""
    p = settings.ANNOTATIONS_DIR / category_id / f"{image_id}.txt"
    if not p.exists():
        return ""
    out: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        coords = parts[1:]
        try:
            if len(coords) == 4:  # already a box (cx, cy, w, h)
                box = [_clamp01(float(v)) for v in coords]
            elif len(coords) >= 6 and len(coords) % 2 == 0:  # polygon → box
                box = _polygon_to_box([float(v) for v in coords])
            else:
                continue
        except ValueError:
            continue
        out.append("0 " + " ".join(f"{v:.6f}" for v in box))
    return "\n".join(out)


def finalize(
    category_id: str,
    category_name: str,
    job_id: str,
    images: List[dict],
    val_split: Optional[float] = None,
    min_val: Optional[int] = None,
) -> dict:
    """Build the frozen YOLO dataset + dataset.yaml.

    `images` is a list of {"id", "stored_path"} for ANNOTATED images only.
    Returns {dataset_yaml, train_count, val_count, num_images, val_is_train}.
    """
    val_split = settings.TRAIN_VAL_SPLIT if val_split is None else val_split
    min_val = settings.MIN_VAL_IMAGES if min_val is None else min_val

    if not images:
        raise ValueError("没有已标注的图片，无法训练。请先标注至少一张图片。")

    root = settings.DATASETS_DIR / category_id / "yolo" / job_id
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    items = list(images)
    # Seeded shuffle so the same annotations reproduce the same train/val split
    # (and thus the same best.pt / mAP) across reruns (M-3). A local Random
    # instance avoids perturbing global RNG state in the worker thread.
    rng = random.Random(settings.TRAIN_SPLIT_SEED)
    rng.shuffle(items)
    n = len(items)
    n_train = max(1, int(round(n * val_split)))
    train_items = items[:n_train]
    val_items = items[n_train:]

    val_is_train = False
    if len(val_items) < min_val:
        # Too few images for a real holdout: train on all and mirror them into
        # val so Ultralytics can still compute mAP (metrics will be optimistic).
        train_items = items
        val_items = items
        val_is_train = True

    def _materialize(split: str, group: List[dict]) -> int:
        count = 0
        for it in group:
            src = Path(it["stored_path"])
            ext = src.suffix or ".jpg"
            dst_img = root / "images" / split / f"{it['id']}{ext}"
            try:
                shutil.copy2(src, dst_img)
            except FileNotFoundError:
                logger.warning(f"finalize: missing raw image {src}, skipping")
                continue
            (root / "labels" / split / f"{it['id']}.txt").write_text(
                _read_label(category_id, it["id"]), encoding="utf-8"
            )
            count += 1
        return count

    train_count = _materialize("train", train_items)
    val_count = _materialize("val", val_items)

    dataset_yaml = root / "dataset.yaml"
    data = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: strip_timestamp_suffix(category_name)},
    }
    dataset_yaml.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    # Freeze the exact split (image ids + seed) alongside the dataset so each
    # best.pt is traceable to the data partition it was trained on (M-3).
    (root / "split.json").write_text(
        json.dumps(
            {
                "seed": settings.TRAIN_SPLIT_SEED,
                "val_split": val_split,
                "val_is_train": val_is_train,
                "train": [it["id"] for it in train_items],
                "val": [it["id"] for it in val_items],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        f"finalize: '{category_name}' job={job_id} train={train_count} "
        f"val={val_count} val_is_train={val_is_train} → {dataset_yaml}"
    )
    return {
        "dataset_yaml": str(dataset_yaml.resolve()),
        "train_count": train_count,
        "val_count": val_count,
        "num_images": n,
        "val_is_train": val_is_train,
    }
