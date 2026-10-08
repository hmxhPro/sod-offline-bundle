# YOLOE Training Component

This directory holds the **custom-training** tooling for the project's YOLOE detector.

> The runtime **video detector** itself now lives at
> `backend/app/services/yoloe_detector.py` (`YOLOEDetector`), selected by
> `DETECTION_MODEL=yoloe`. This directory is no longer a detector implementation —
> it is where you fine-tune YOLOE on your own data and where the offline
> benchmark/utility scripts live.

## 📁 Directory Structure

```
YOLOWorld/
├── README.md               # This file
├── TRAINING.md             # Full custom-training guide (中文)
├── train_yoloe.py          # YOLOE training CLI (invoked by the in-app training workflow)
├── dataset.yaml.example    # Example Ultralytics dataset YAML
├── benchmark_detectors.py  # Offline detector speed/quality comparison
└── requirements.txt        # Training/inference extras (ultralytics, sahi)
```

## 🚀 Training

The in-app training workflow (the "训练" panel / `training_manager.py`) shells out to
`train_yoloe.py`. To run it manually:

```bash
python backend/YOLOWorld/train_yoloe.py \
    --data backend/YOLOWorld/dataset.yaml \
    --model /abs/path/to/backend/models/yolo/yoloe-11l-seg.pt   # defaults to YOLOE_BASE_MODEL in .env
```

The trained weights land at `<project>/<name>/weights/best.pt`. See **TRAINING.md** for the
full guide (dataset layout, all flags, swapping the runtime weights, WSL2 RAM notes).

## ⚙️ Configuration (`backend/.env`)

```bash
# Default video detector
DETECTION_MODEL=yoloe
# Shared YOLOE base weights: runtime detector + zero-shot image detection + training start point
YOLOE_BASE_MODEL=/abs/path/to/backend/models/yolo/yoloe-11l-seg.pt

# SAHI sliced inference for small objects (used by the runtime YOLOEDetector on large frames)
SAHI_SLICE_HEIGHT=640
SAHI_SLICE_WIDTH=640
SAHI_OVERLAP_HEIGHT_RATIO=0.2
SAHI_OVERLAP_WIDTH_RATIO=0.2

DEVICE=cuda:0
```

## 📦 YOLOE base models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yoloe-11s-seg.pt | small | ⚡⚡⚡ | ⭐⭐ |
| yoloe-11m-seg.pt | medium | ⚡⚡ | ⭐⭐⭐ |
| yoloe-11l-seg.pt | large | ⚡ | ⭐⭐⭐⭐ |

Place the weight under `backend/models/yolo/` and point `YOLOE_BASE_MODEL` at it.

## 🔧 Benchmark

Compare the YOLOE detector against Grounding DINO offline:

```bash
python benchmark_detectors.py \
    --image path/to/image.jpg \
    --prompt "person . car" \
    --detectors grounding_dino yoloe \
    --iterations 5
```

## 🔗 References

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [SAHI GitHub](https://github.com/obss/sahi)

## 📝 License

- Ultralytics YOLO (AGPL-3.0)
- SAHI (MIT)
