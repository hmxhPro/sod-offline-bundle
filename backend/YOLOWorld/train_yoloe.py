"""
Custom-model training CLI.
--------------------------
Fine-tune a plain Ultralytics YOLOv11 (detection) model on a custom dataset.

The base checkpoint is loaded via ``ultralytics.YOLO`` so ``.train()`` uses the
standard DetectionTrainer and the classification head learns normally. (Loading
a YOLOE-family checkpoint here via ``YOLOE`` would instead route ``.train()`` to
the multi-modal YOLOETrainer, whose text/cls branch is never supervised on a
fixed-class dataset — cls loss never drops, mAP stays flat.) Open-vocabulary
detection elsewhere still uses YOLOE; this script is only the custom-training
path. The base defaults to ``TRAIN_BASE_MODEL`` in backend/.env (e.g. yolo11l.pt).

Usage:
    python train_yoloe.py --data dataset.yaml [options]

The trained weights land at:
    <project>/<name>/weights/best.pt
The backend registers that best.pt as a selectable trained model (chosen
per-request by model_id); no .env change is needed to use it. (The filename
train_yoloe.py is kept for compatibility with the backend training runner.)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Force offline mode for ultralytics (never hit the network during training)
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("ULTRALYTICS_OFFLINE", "1")

# Make app.* importable so we can reuse logger / settings
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.core.config import settings  # noqa: E402
from app.core.logging import logger  # noqa: E402


def _load_train_model(weights: str):
    """Instantiate a plain Ultralytics YOLO model for standard fine-tuning.

    We deliberately use ``YOLO`` (not ``YOLOE``): the training base is a normal
    YOLOv11 detection checkpoint, and ``YOLO(...).train()`` routes to the
    standard DetectionTrainer so the classification head trains normally.
    Loading the same weights via ``YOLOE`` would route ``.train()`` to the
    multi-modal YOLOETrainer, whose text/cls branch is never supervised on a
    fixed-class dataset (cls loss never drops, mAP stays flat) — the failure
    mode this replaces. Open-vocabulary detection still uses YOLOE elsewhere
    (see app/services/yoloe_detector.py).
    """
    from ultralytics import YOLO

    logger.info(f"Loading YOLO training base weights: {weights}")
    return YOLO(weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train / fine-tune a YOLOE model on a custom dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML (see dataset.yaml.example).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.train_base_model,
        help="Pretrained weights to start from (e.g. yolo11l.pt). "
             "Defaults to TRAIN_BASE_MODEL in .env.",
    )
    parser.add_argument("--epochs", type=int, default=settings.TRAIN_DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=settings.TRAIN_DEFAULT_IMGSZ, help="Input image size.")
    parser.add_argument(
        "--batch",
        type=int,
        default=settings.TRAIN_DEFAULT_BATCH,
        help="Batch size; pass -1 to let Ultralytics auto-fit GPU memory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=settings.DEVICE,
        help="Compute device, e.g. 'cuda:0', 'cpu', or '0,1' for multi-GPU.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(_BACKEND_ROOT / "runs" / "train"),
        help="Root output directory for training runs.",
    )
    parser.add_argument("--name", type=str, default="yoloe_exp", help="Experiment subdirectory name.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the most recent checkpoint in <project>/<name>.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader worker processes per loader. Low default (Ultralytics "
             "uses 8) for memory-constrained hosts: train+val+final-val loaders "
             "each fork this many, and on a 15 GB WSL2 box 8 each deadlocked the "
             "worker IPC at the final epoch. Use 0 to load in the main process.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early-stopping patience (epochs without val improvement).",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Freeze the first N layers (useful for light fine-tuning).",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=None,
        help="Initial learning rate (Ultralytics default if unset).",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=settings.TRAIN_MOSAIC,
        help="Mosaic augmentation probability (0-1). Kept low for small objects "
             "(mosaic tiles 4 images and shrinks each ~2×).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=settings.TRAIN_SCALE,
        help="Scale-jitter gain. Kept tight for small objects.",
    )
    parser.add_argument(
        "--close-mosaic",
        dest="close_mosaic",
        type=int,
        default=settings.TRAIN_CLOSE_MOSAIC,
        help="Disable mosaic for the final N epochs (finish on clean targets).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="On success, print a machine-readable result line "
             "'__YOLOE_TRAIN_JSON__ {...}' to stdout (best_pt + metrics). "
             "Used by the backend training runner.",
    )
    return parser.parse_args()


# Sentinel prefix the backend training runner scans stdout for.
TRAIN_JSON_PREFIX = "__YOLOE_TRAIN_JSON__"


def _extract_metrics(results) -> dict:
    """Best-effort pull of float metrics (mAP/precision/recall/...) from an
    Ultralytics training results object. Returns {} if unavailable."""
    out: dict = {}
    rd = getattr(results, "results_dict", None)
    if isinstance(rd, dict):
        for k, v in rd.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def run_training(
    data: str,
    model: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    project: str | None = None,
    name: str = "yoloe_exp",
    resume: bool = False,
    workers: int = 8,
    patience: int = 50,
    freeze: int | None = None,
    lr0: float | None = None,
    mosaic: float | None = None,
    scale: float | None = None,
    close_mosaic: int | None = None,
) -> dict:
    """Train a YOLOE model and return a result dict.

    Returns: {save_dir, best_pt, last_pt, metrics, elapsed}.
    Raises FileNotFoundError if the dataset YAML is missing.

    This is the API-callable core (the FastAPI training runner spawns this
    module as a subprocess via `main()`); the CLI `main()` is a thin wrapper.
    """
    device = device or settings.DEVICE
    project = project or str(_BACKEND_ROOT / "runs" / "train")
    # Small-object augmentation defaults (overridable by caller/CLI).
    mosaic = settings.TRAIN_MOSAIC if mosaic is None else mosaic
    scale = settings.TRAIN_SCALE if scale is None else scale
    close_mosaic = settings.TRAIN_CLOSE_MOSAIC if close_mosaic is None else close_mosaic

    data_path = Path(data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    logger.info("─" * 60)
    logger.info("YOLOE training run")
    logger.info(f"  data    : {data_path}")
    logger.info(f"  model   : {model}")
    logger.info(f"  epochs  : {epochs}")
    logger.info(f"  imgsz   : {imgsz}")
    logger.info(f"  batch   : {batch}")
    logger.info(f"  device  : {device}")
    logger.info(f"  project : {project}")
    logger.info(f"  name    : {name}")
    logger.info(f"  aug     : mosaic={mosaic} scale={scale} close_mosaic={close_mosaic}")
    logger.info("─" * 60)

    model_obj = _load_train_model(model)

    train_kwargs = dict(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        resume=resume,
        workers=workers,
        patience=patience,
        mosaic=mosaic,
        scale=scale,
        close_mosaic=close_mosaic,
    )
    if freeze is not None:
        train_kwargs["freeze"] = freeze
    if lr0 is not None:
        train_kwargs["lr0"] = lr0

    start = time.time()
    results = model_obj.train(**train_kwargs)
    elapsed = time.time() - start

    # Resolve best.pt path. Ultralytics exposes save_dir on the results object.
    save_dir = Path(getattr(results, "save_dir", Path(project) / name))
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    logger.info("─" * 60)
    logger.info(f"Training finished in {elapsed / 60:.1f} min")
    logger.info(f"  best weights : {best_pt}")
    logger.info(f"  last weights : {last_pt}")
    logger.info("─" * 60)

    return {
        "save_dir": str(save_dir),
        "best_pt": str(best_pt),
        "last_pt": str(last_pt),
        "metrics": _extract_metrics(results),
        "elapsed": elapsed,
    }


def main() -> int:
    args = parse_args()
    try:
        info = run_training(
            data=args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume,
            workers=args.workers,
            patience=args.patience,
            freeze=args.freeze,
            lr0=args.lr0,
            mosaic=args.mosaic,
            scale=args.scale,
            close_mosaic=args.close_mosaic,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    logger.info(
        "Training done. The backend registers best.pt as a selectable trained "
        "model (chosen per-request by model_id); no .env change needed."
    )

    if args.json:
        import json
        # Authoritative completion signal for the backend training runner.
        print(f"{TRAIN_JSON_PREFIX} {json.dumps(info)}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
