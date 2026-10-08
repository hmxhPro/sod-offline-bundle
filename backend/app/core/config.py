"""
app/core/config.py
------------------
Central application configuration via Pydantic Settings.
Reads from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import Literal


def _gdino_config_default() -> Path:
    """Use the config bundled with the installed groundingdino package."""
    try:
        import groundingdino
        pkg_config = Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py"
        if pkg_config.exists():
            return pkg_config
    except ImportError:
        pass
    return Path("./models/groundingdino/config/GroundingDINO_SwinT_OGC.py")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ──────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # ── Logging (O-5 / O-6) ───────────────────────────────────────────────────
    # LOG_DIR is resolved to an absolute path at logging setup, so the file sink
    # no longer depends on the process CWD (O-6). LOG_LEVEL controls the console
    # sink; the file sink always keeps DEBUG for post-mortem detail.
    LOG_DIR: Path = Path("./logs")
    LOG_LEVEL: str = "INFO"

    # ── Storage ─────────────────────────────────────────────────────────────
    UPLOAD_DIR: Path = Path("./uploads")
    RESULTS_DIR: Path = Path("./results")

    # ── Resource guards (P-1 / R-4 / R-5) ────────────────────────────────────
    # Upload limits: an unbounded upload + no cleanup fills the disk (P-1). These
    # cap per-file bytes, batch file count, and refuse new uploads below a disk
    # water mark. Tune for the deployment's disk; the in-loop byte cap is the
    # real guard (disk_usage pre-check is a TOCTOU estimate).
    MAX_UPLOAD_BYTES: int = 2 * 1024 ** 3        # video single-file cap (2 GiB)
    MAX_IMAGE_BYTES: int = 50 * 1024 ** 2        # image single-file cap (50 MiB)
    MAX_UPLOAD_FILES: int = 200                  # per-request file-count cap
    MIN_FREE_DISK_BYTES: int = 2 * 1024 ** 3     # refuse uploads below this free space
    # SSE per-task frame queue cap — a disconnected client otherwise lets the
    # queue accumulate full base64 frames until OOM (R-4). Oldest frame is
    # dropped when full; terminal sentinels are never dropped.
    TASK_QUEUE_MAXSIZE: int = 120
    # Max terminal tasks kept in memory; oldest are evicted (their results are
    # on disk + in the DB archive) to bound long-run memory growth (R-5).
    MAX_RETAINED_TASKS: int = Field(default=50, ge=1)

    # ── Detection Model ──────────────────────────────────────────────────────
    DETECTION_MODEL: Literal["florence2", "grounding_dino", "yoloe"] = "yoloe"
    DEVICE: str = "cuda:0"

    # Florence-2
    FLORENCE2_MODEL_ID: str = "microsoft/Florence-2-large"

    # Grounding DINO
    GDINO_CONFIG_PATH: Path = Field(default_factory=_gdino_config_default)
    GDINO_CHECKPOINT_PATH: Path = Path(
        "./models/groundingdino/weights/groundingdino_swint_ogc.pth"
    )

    # YOLOE (default video detector) — weights come from YOLOE_BASE_MODEL below.
    # SAHI sliced inference for small objects (auto-used on large frames).
    SAHI_SLICE_HEIGHT: int = 640
    SAHI_SLICE_WIDTH: int = 640
    SAHI_OVERLAP_HEIGHT_RATIO: float = 0.2
    SAHI_OVERLAP_WIDTH_RATIO: float = 0.2

    # ── Detection Thresholds ─────────────────────────────────────────────────
    # Three-tier confidence strategy
    DINO_DISCARD_THRESHOLD: float = Field(default=0.25, ge=0.0, le=1.0)
    DINO_CANDIDATE_THRESHOLD: float = Field(default=0.30, ge=0.0, le=1.0)
    DINO_DIRECT_CONFIRM_THRESHOLD: float = Field(default=0.60, ge=0.0, le=1.0)

    # Legacy thresholds (for backward compatibility)
    BOX_THRESHOLD: float = Field(default=0.25, ge=0.0, le=1.0)
    TEXT_THRESHOLD: float = Field(default=0.25, ge=0.0, le=1.0)

    # ── ByteTrack ────────────────────────────────────────────────────────────
    # Aligned with BOX_THRESHOLD: ByteTrack would otherwise filter detector
    # outputs a second time at this threshold and drop low-score small-object
    # boxes (0.25–0.5) before any track can be created.
    TRACK_THRESH: float = Field(default=0.25, ge=0.0, le=1.0)
    TRACK_BUFFER: int = 30
    MATCH_THRESH: float = Field(default=0.8, ge=0.0, le=1.0)

    # ── Processing ───────────────────────────────────────────────────────────
    # Run full detection every N frames; track in between for speed
    DETECTION_INTERVAL: int = Field(default=5, ge=1)
    MAX_CONCURRENT_TASKS: int = 2

    # ── Adaptive Keyframe Detection ──────────────────────────────────────────
    ENABLE_ADAPTIVE_KEYFRAME: bool = True
    FRAME_DIFF_THRESHOLD: float = Field(default=0.05, ge=0.0, le=1.0)
    PHASH_THRESHOLD: int = Field(default=8, ge=0)
    HIST_THRESHOLD: float = Field(default=0.85, ge=0.0, le=1.0)
    MIN_DETECTION_INTERVAL: int = Field(default=10, ge=1)
    MAX_DETECTION_INTERVAL: int = Field(default=30, ge=1)

    # ── Frame Saving Strategy ────────────────────────────────────────────────
    # Options: "all" | "keyframes_only" | "detections_only" | "unique_targets"
    #          | "best_shot"
    # - "all": Save every frame (original behavior, large ZIP files)
    # - "keyframes_only": Only save frames where detection was run (70-90% reduction)
    # - "detections_only": Only save detection keyframes with at least one box
    # - "unique_targets": Save a frame only when it shows a track_id not saved
    #   before, then re-save the same target at most every
    #   TRACK_SAVE_COOLDOWN_SEC. Uses ByteTrack IDs for cross-frame dedup, so
    #   a target parked in view for minutes yields a few snapshots instead of
    #   one per keyframe. See app/services/frame_save_gate.py.
    # - "best_shot": Exactly ONE snapshot per target — the single highest-quality
    #   frame across the target's whole track lifetime (scored on sharpness,
    #   on-target size, confidence, and an edge-truncation penalty). Writes are
    #   deferred to the end of the video, so the streamed thumbnails are only
    #   the live previews; the optimal snapshots land in the result ZIP. See
    #   app/services/best_shot.py.
    SAVE_FRAMES_MODE: Literal[
        "all", "keyframes_only", "detections_only", "unique_targets", "best_shot"
    ] = "unique_targets"
    # "unique_targets" only: minimum seconds between saves of the SAME track.
    # 0 disables throttling (every detection keyframe with boxes, i.e.
    # equivalent to "detections_only").
    TRACK_SAVE_COOLDOWN_SEC: float = Field(default=10.0, ge=0.0)

    # ── Best-shot scoring ("best_shot" mode only) ────────────────────────────
    # Relative weights of the three quality components (normalised internally,
    # need not sum to 1). Sharpness rewards in-focus crops, area rewards targets
    # that are large/close, confidence rewards strong detections.
    BEST_SHOT_WEIGHT_SHARPNESS: float = Field(default=0.4, ge=0.0)
    BEST_SHOT_WEIGHT_AREA: float = Field(default=0.3, ge=0.0)
    BEST_SHOT_WEIGHT_CONFIDENCE: float = Field(default=0.3, ge=0.0)
    # Laplacian-variance value treated as "fully sharp" (scores saturate here).
    BEST_SHOT_SHARPNESS_REF: float = Field(default=200.0, gt=0.0)
    # Bbox area fraction of the frame treated as "fully large" (scores saturate).
    # Small for a small-object project — 0.05 ⇒ a box covering 5% of the frame
    # already maxes the area term.
    BEST_SHOT_AREA_REF: float = Field(default=0.05, gt=0.0, le=1.0)
    # A box within this many px of any frame edge is likely truncated.
    BEST_SHOT_EDGE_MARGIN_PX: int = Field(default=4, ge=0)
    # Multiplier applied to an edge-touching box's score (1.0 = no penalty).
    BEST_SHOT_EDGE_PENALTY: float = Field(default=0.6, ge=0.0, le=1.0)
    # Snapshots scoring below this are never written (0 = keep every target's
    # best, however poor).
    BEST_SHOT_MIN_SCORE: float = Field(default=0.0, ge=0.0, le=1.0)

    # JPEG quality for saved frames (1-100, lower = smaller files)
    JPEG_QUALITY: int = Field(default=85, ge=1, le=100)

    # ── Frame Quality Filtering ──────────────────────────────────────────────
    ENABLE_FRAME_QUALITY_CHECK: bool = True
    BLACK_FRAME_THRESHOLD: float = Field(default=15.0, ge=0.0, le=255.0)
    DARK_FRAME_THRESHOLD: float = Field(default=30.0, ge=0.0, le=255.0)
    BRIGHT_FRAME_THRESHOLD: float = Field(default=240.0, ge=0.0, le=255.0)
    MIN_FRAME_STD_DEV: float = Field(default=10.0, ge=0.0)

    # ── Visualization ────────────────────────────────────────────────────────
    CORNER_STYLE_BOX: bool = True
    CORNER_LENGTH: int = Field(default=30, ge=10)
    BOX_THICKNESS: int = Field(default=4, ge=1)

    # ── Color Filtering ──────────────────────────────────────────────────────
    ENABLE_COLOR_FILTER: bool = True
    COLOR_FILTER_LOG_LEVEL: str = "DEBUG"  # DEBUG | INFO | WARNING

    # ── VLM (MiniCPM-V) ─────────────────────────────────────────────────────
    VLM_ENABLED: bool = True
    VLM_API_BASE: str = "http://localhost:8010/v1"
    VLM_MODEL_NAME: str = "MiniCPM-V-4_5"
    VLM_MAX_CONCURRENT: int = 1
    VLM_INTERVAL_TENTATIVE: float = 3.0
    VLM_INTERVAL_CONFIRMED: float = 8.0
    VLM_SCORE_THRESHOLD: float = 0.45
    CROP_PADDING_NORMAL: float = 0.15
    CROP_PADDING_SMALL: float = 0.30
    SMALL_OBJECT_AREA_RATIO: float = 0.01

    # ── Fusion ───────────────────────────────────────────────────────────────
    VLM_WEIGHT: float = 0.6
    DINO_WEIGHT: float = 0.4
    VLM_CONFIRM_THRESHOLD: float = 0.65
    FINAL_CONFIRM_THRESHOLD: float = 0.65
    CONFIRM_THRESHOLD: float = 0.65
    REJECT_THRESHOLD: float = 0.35

    # Track confirmation requirements
    MIN_HIT_COUNT_HIGH_CONF: int = 2
    MIN_HIT_COUNT_MID_CONF: int = 3

    # ── LLM 自动标注（Qwen 多模态 · OpenAI 兼容接口）─────────────────────────
    # 数据标注调用大模型的全部配置都集中在后端，硬编码为通义千问（Qwen）系列，
    # 通过阿里云 DashScope 的 OpenAI 兼容端点访问。前端不感知、也无法选择提供商
    # 或填写密钥 —— 使用者只需在 .env 中设置 LLM_API_KEY，并按需修改 LLM_MODEL。
    #
    #   LLM_API_KEY  : DashScope（百炼）API Key，形如 sk-xxxxxxxx；为空则该功能禁用。
    #   LLM_MODEL    : 通义千问“视觉理解(VL)”模型型号——必须能读图，不能是纯文本模型
    #                  （如 qwen-max / qwen3-max）。推荐 qwen3-vl-plus；备选
    #                  qwen-vl-max-latest、qwen3-vl-flash。
    #   LLM_API_BASE : OpenAI 兼容端点，一般无需改动。
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "qwen3-vl-plus"
    LLM_API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # 同时进行的标注 API 请求数（控制速率，避免触发限流）。
    LLM_MAX_CONCURRENT: int = Field(default=3, ge=1, le=10)
    # 送入模型前图片的最长边（像素）。缩放以节省 token，并让返回的像素坐标可控。
    LLM_IMAGE_MAX_SIZE: int = Field(default=1024, ge=256, le=4096)
    # 单张图片的请求超时（秒）。
    LLM_TIMEOUT_SECONDS: float = Field(default=120.0, gt=0)

    # ── 一站式自动标注：视频抽帧密度上限 ──────────────────────────────────────
    # 上传视频做自动标注时，按密度档抽帧再逐帧调用大模型标注。每帧 = 一次付费
    # API 调用，故每档设总帧数上限以控制成本；超过上限时在视频时长内均匀降采样。
    #   稀疏 ≈ 每 3 秒 1 帧 · 适中 ≈ 每 1 秒 1 帧 · 密集 ≈ 每秒 3 帧
    VIDEO_FRAME_MAX_SPARSE: int = Field(default=200, ge=1)
    VIDEO_FRAME_MAX_MEDIUM: int = Field(default=400, ge=1)
    VIDEO_FRAME_MAX_DENSE: int = Field(default=800, ge=1)

    @property
    def llm_configured(self) -> bool:
        """是否已在 .env 中配置 LLM_API_KEY（决定标注功能是否可用）。"""
        return bool(self.LLM_API_KEY.strip())

    # ── Optional Redis ───────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── PostgreSQL (task history persistence) ───────────────────────────────
    # Override in .env for non-default credentials. Falls back gracefully if
    # the DB is unreachable — detection still works, just without history.
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/sod"
    DATABASE_ECHO: bool = False  # set True to log every SQL statement

    # ── YOLOE custom training + image detection ─────────────────────────────
    # Separate storage for the annotation→train→model-list workflow (REQ2/REQ3).
    # Raw per-category uploads, the frozen annotated YOLO dataset, mutable
    # working labels, and Ultralytics training runs each live in their own root.
    DATASETS_DIR: Path = Path("./datasets")        # datasets/<cat_id>/{raw,yolo/<job_id>}
    ANNOTATIONS_DIR: Path = Path("./annotations")  # annotations/<cat_id>/<image_id>.txt (working labels)
    TRAIN_RUNS_DIR: Path = Path("./runs/train")    # Ultralytics project dir; best.pt under <name>/weights
    # Subdir under RESULTS_DIR for image-detection annotated outputs.
    IMGDET_SUBDIR: str = "imgdet"

    # Base YOLOE weights — used by the default video detector and zero-shot
    # image detection (open-vocabulary `set_classes`). Absolute path recommended.
    YOLOE_BASE_MODEL: str = "yoloe-11l-seg.pt"

    # Custom-training base weights — a PLAIN Ultralytics YOLOv11 detection model,
    # loaded via `YOLO` so `.train()` uses the standard DetectionTrainer and the
    # classification head actually learns. Deliberately decoupled from
    # YOLOE_BASE_MODEL: open-vocabulary detection stays on YOLOE; custom training
    # fine-tunes a normal YOLO. Absolute path recommended (offline training needs
    # the file present locally).
    TRAIN_BASE_MODEL: str = "yolo11l.pt"

    # In-memory cache of loaded image-detection models (LRU). Each loaded
    # YOLOE model shares cuda:0 with the warm video detector — keep this small.
    IMAGE_MODEL_CACHE_SIZE: int = Field(default=2, ge=1)
    IMAGE_DETECT_CONF: float = Field(default=0.25, ge=0.0, le=1.0)

    # Training defaults (overridable per request). Train/val split is applied
    # at dataset finalize; if fewer than MIN_VAL_IMAGES are annotated the val
    # set mirrors train (Ultralytics needs a non-empty val set to compute mAP).
    TRAIN_VAL_SPLIT: float = Field(default=0.8, ge=0.1, le=0.95)
    MIN_VAL_IMAGES: int = Field(default=5, ge=1)
    # Fixed seed for the train/val split so the same annotations reproduce the
    # same split (and thus the same best.pt / mAP) across reruns (M-3).
    TRAIN_SPLIT_SEED: int = 42
    # A finished run whose mAP50 is None or <= this floor is marked
    # `needs_review` and NOT registered as a selectable model — keeps 0-metric /
    # broken models out of production detection (M-2). Raise to enforce a real
    # quality bar; 0.0 only excludes all-zero / metric-less runs.
    MIN_DEPLOYABLE_MAP50: float = Field(default=0.0, ge=0.0, le=1.0)
    TRAIN_DEFAULT_EPOCHS: int = Field(default=100, ge=1)
    # Small objects are the whole point of this project: at imgsz=640 the median
    # labeled object is ~9 px (below the stride-8 detectable floor), which caps
    # mAP regardless of the model. Train at higher resolution so tiny objects
    # survive. imgsz=1280 ≈ 4× the activation memory of 640.
    # Measured on this 16 GB GPU (shared with the ~4 GB warm YOLOE detector):
    # the per-epoch VALIDATION pass at imgsz=1280 peaks at ~15.9 GB (~0.4 GB
    # free) and is ~batch-INDEPENDENT — the train batch only drives the lighter
    # ~10 GB training phase (Ultralytics also grad-accumulates to nbs=64, so a
    # small batch is fine). So the peak is bound by imgsz, not batch: a solo
    # training run survives, but there is little headroom for CONCURRENT
    # detection during the run. The memory lever is imgsz, not batch — to free
    # real headroom (~6 GB) drop TRAIN_DEFAULT_IMGSZ to 1024, at a small
    # small-object accuracy cost.
    TRAIN_DEFAULT_IMGSZ: int = Field(default=1280, ge=64)
    TRAIN_DEFAULT_BATCH: int = Field(default=3)
    # Small-object augmentation: mosaic (4-image tiling) and large scale jitter
    # both shrink already-tiny objects below the detectable floor. Keep mosaic
    # light and scale tight; close_mosaic disables mosaic for the final epochs so
    # training finishes on clean, full-size targets. All overridable per request.
    TRAIN_MOSAIC: float = Field(default=0.3, ge=0.0, le=1.0)
    TRAIN_SCALE: float = Field(default=0.2, ge=0.0, le=0.9)
    TRAIN_CLOSE_MOSAIC: int = Field(default=15, ge=0)
    # DataLoader worker processes. Kept low (Ultralytics default is 8) because
    # this box is a 15 GB-RAM WSL2 VM: train + val + final-validation loaders
    # each fork `workers` children, and near the last epoch all three sets are
    # alive at once. 8×3 large-seg workers exhausted RAM and deadlocked the
    # worker↔main IPC sockets (training hung forever on the final epoch). 0 =
    # load batches in the main process (slowest but immune to that deadlock).
    TRAIN_DEFAULT_WORKERS: int = Field(default=2, ge=0)

    @property
    def yoloe_base_model(self) -> str:
        """Resolved YOLOE base weights for the video detector and zero-shot
        image detection (open-vocabulary)."""
        return self.YOLOE_BASE_MODEL

    @property
    def train_base_model(self) -> str:
        """Resolved base weights for custom training (plain YOLOv11 detection)."""
        return self.TRAIN_BASE_MODEL

    def ensure_dirs(self) -> None:
        """Create storage directories if they don't exist."""
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        self.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.TRAIN_RUNS_DIR.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
settings.ensure_dirs()
