"""
app/models/schemas.py
----------------------
Pydantic data models for API request / response payloads.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ────────────────────────────────────────────────────────────────────────────
# Error envelope (A-2)
# ────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Unified body for every non-2xx response.

    `code` is machine-readable (frontend may branch on it), `message` is the
    human-facing Chinese text, `detail` carries extra context (validation field
    list, or the raw exception string when DEBUG), and `request_id` ties the
    response to its server-side log lines.
    """
    code: str
    message: str
    detail: Optional[Any] = None
    request_id: str


# ────────────────────────────────────────────────────────────────────────────
# Task Status
# ────────────────────────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    PACKAGING = "packaging"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EARLY_TERMINATED = "early_terminated"


# ────────────────────────────────────────────────────────────────────────────
# Upload
# ────────────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Returned after a successful video upload."""
    video_id: str = Field(..., description="Unique ID for the uploaded video")
    filename: str
    size_bytes: int
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    total_frames: Optional[int] = None


# ────────────────────────────────────────────────────────────────────────────
# Detection Task
# ────────────────────────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    """Request body to start a detection task."""
    video_id: str = Field(..., description="ID returned from /api/upload")
    video_filename: Optional[str] = Field(
        default=None,
        description="Original uploaded filename — used for history display only.",
    )
    prompt: str = Field(
        default="",
        description=(
            "Natural language description of the object to detect. "
            "E.g. '帮我检测视频中的菜园'. Leave empty when `model_id` is given. "
            "Exactly one of `prompt` / `model_id` must be supplied (validated in the endpoint)."
        ),
    )
    model_id: Optional[str] = Field(
        default=None,
        description=(
            "Use a trained model's baked-in classes instead of a natural-language "
            "prompt. Mutually exclusive with `prompt`."
        ),
    )
    detection_interval: Optional[int] = Field(
        default=None,
        ge=1,
        description="Run full detection every N frames; track in between. "
                    "Defaults to server-side setting.",
    )
    box_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    text_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    enable_vlm: Optional[bool] = Field(
        default=None,
        description="Enable VLM semantic verification. Defaults to server-side setting.",
    )


class DetectResponse(BaseModel):
    """Returned when a detection task is successfully queued."""
    task_id: str
    video_id: str
    prompt: str
    status: TaskStatus


# ────────────────────────────────────────────────────────────────────────────
# Per-Frame Result
# ────────────────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """XYXY bounding box coordinates (absolute pixels)."""
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """Single object detection on a frame."""
    track_id: Optional[int] = None        # ByteTrack assigned ID
    label: str                             # User's prompt label
    score: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    track_status: Optional[str] = None     # candidate | confirmed | rejected | lost
    vlm_verified: Optional[bool] = None
    vlm_score: Optional[float] = None
    final_score: Optional[float] = None
    visible: bool = True                   # Frontend visibility flag


class FrameResult(BaseModel):
    """All detection results for a single video frame."""
    frame_id: int                          # 0-based frame index
    timestamp: str                         # HH:MM:SS.mmm
    timestamp_seconds: float
    detections: List[Detection]
    image_filename: Optional[str] = None   # Saved result image filename (None if not saved)
    # Base64-encoded JPEG for streaming (set when streaming, empty when saved)
    image_b64: str = ""


# ────────────────────────────────────────────────────────────────────────────
# Task State
# ────────────────────────────────────────────────────────────────────────────

class TaskState(BaseModel):
    """Full task state returned by GET /api/task/{task_id}."""
    task_id: str
    video_id: str
    prompt: str
    status: TaskStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="0.0 – 1.0")
    total_frames: int = 0
    processed_frames: int = 0
    # Frames with ≥1 detection. Authoritative server-side counter: the SSE
    # frame queue is bounded/drop-oldest, so a client counting delivered
    # frames undercounts after any disconnect gap — it reconciles from this.
    detection_frame_count: int = 0
    results: List[FrameResult] = []
    error: Optional[str] = None
    zip_ready: bool = False
    early_terminated: bool = False
    termination_reason: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """Lightweight task status for polling — excludes frame results."""
    task_id: str
    video_id: str
    prompt: str
    status: TaskStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    total_frames: int = 0
    processed_frames: int = 0
    # 0 when served from the DB archive (no column without Alembic) — clients
    # must treat this as a floor, not a replacement (take max with local).
    detection_frame_count: int = 0
    error: Optional[str] = None
    zip_ready: bool = False
    early_terminated: bool = False
    termination_reason: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────────
# SSE / WebSocket streaming message
# ────────────────────────────────────────────────────────────────────────────

class StreamEvent(BaseModel):
    """
    Streamed to the client after each frame is processed.
    event_type: "frame" | "progress" | "done" | "error"
    """
    event_type: str
    task_id: str
    frame_result: Optional[FrameResult] = None
    progress: float = 0.0
    total_frames: int = 0
    processed_frames: int = 0
    detection_frame_count: int = 0
    error: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────────
# History (DB-backed)
# ────────────────────────────────────────────────────────────────────────────

class TaskHistoryItem(BaseModel):
    """One row in the past-tasks list. Sourced from the detection_tasks table."""

    model_config = ConfigDict(from_attributes=True)

    task_id: str
    video_id: str
    video_filename: Optional[str] = None
    prompt: str
    status: str
    progress: float = 0.0
    total_frames: int = 0
    processed_frames: int = 0
    error: Optional[str] = None
    zip_ready: bool = False
    early_terminated: bool = False
    termination_reason: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    finished_at: Optional[datetime] = None


# ════════════════════════════════════════════════════════════════════════════
# YOLOE custom-training workflow (REQ1/REQ2/REQ3)
# ════════════════════════════════════════════════════════════════════════════

# ── Categories ──────────────────────────────────────────────────────────────

class CategoryCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="类别名，将作为训练后模型名")
    description: Optional[str] = None


class CategoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: Optional[str] = None
    status: str = "draft"
    image_count: int = 0
    annotated_count: int = 0
    created_at: datetime
    updated_at: datetime


# ── Dataset images + annotation ─────────────────────────────────────────────

class DatasetImageItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    category_id: str
    filename: str
    width: int = 0
    height: int = 0
    annotation_status: str = "pending"
    box_count: int = 0
    created_at: datetime


class AnnotationBox(BaseModel):
    """One YOLO-format box: class id + normalized center/size in [0, 1]."""
    cls: int = Field(default=0, ge=0)
    cx: float = Field(..., ge=0.0, le=1.0)
    cy: float = Field(..., ge=0.0, le=1.0)
    w: float = Field(..., gt=0.0, le=1.0)
    h: float = Field(..., gt=0.0, le=1.0)


class AnnotationPayload(BaseModel):
    """All boxes for a single image. Empty list = a background/negative sample."""
    boxes: List[AnnotationBox] = []


class DatasetImportResult(BaseModel):
    """Summary returned after importing a pre-annotated YOLO dataset folder
    into a category (boxes folded to the single category class)."""
    imported_images: int = 0   # images successfully stored + registered
    with_annotation: int = 0   # imported images that had >=1 valid box
    background: int = 0        # imported images with no/empty label (background)
    skipped_files: int = 0     # non-image / unreadable files ignored
    message: str = ""


# ── Training jobs ────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    epochs: Optional[int] = Field(default=None, ge=1, le=1000)
    imgsz: Optional[int] = Field(default=None, ge=64, le=2048)
    batch: Optional[int] = Field(default=None)
    base_model: Optional[str] = Field(default=None, description="覆盖训练基础权重（绝对路径）")
    mosaic: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="马赛克增强概率（小目标建议低；0=关闭）")
    scale: Optional[float] = Field(default=None, ge=0.0, le=0.9, description="尺度抖动增益（小目标建议低）")
    close_mosaic: Optional[int] = Field(default=None, ge=0, description="最后 N 轮关闭马赛克")


class TrainResponse(BaseModel):
    job_id: str
    category_id: str
    model_name: str
    status: str


class TrainingJobItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    category_id: str
    model_name: str
    status: str
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    dataset_yaml: Optional[str] = None
    base_model: Optional[str] = None
    metric_map50: Optional[float] = None
    metric_map50_95: Optional[float] = None
    metrics: Optional[dict] = None
    best_pt_path: Optional[str] = None
    error: Optional[str] = None
    # True when the val set mirrored train (too few images for a real holdout) —
    # the reported mAP is then optimistic and must be shown with a warning, not
    # as a generalization metric (M-1). Sourced from the job's params JSON.
    val_is_train: bool = False
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


# ── Trained model registry (REQ3) ───────────────────────────────────────────

class TrainedModelItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    version: int = 1
    category_id: str
    training_job_id: str
    weights_path: str
    base_model: Optional[str] = None
    class_names: Optional[dict] = None
    dataset_yaml: Optional[str] = None
    num_images: int = 0
    metrics: Optional[dict] = None
    # Mirrors TrainingJobItem.val_is_train — the model's reported mAP was
    # measured on its own training set (no real holdout); the UI flags it (M-1).
    val_is_train: bool = False
    trained_started_at: Optional[datetime] = None
    trained_finished_at: Optional[datetime] = None
    created_at: datetime


# ── Image detection (REQ1) ──────────────────────────────────────────────────

class ImageDetectResultItem(BaseModel):
    image_index: int
    filename: str
    width: int = 0
    height: int = 0
    detections: List[Detection] = []
    annotated_url: str


class ImageDetectResponse(BaseModel):
    batch_id: str
    mode: str                      # "zeroshot" | "model"
    model_id: Optional[str] = None
    class_names: List[str] = []
    results: List[ImageDetectResultItem] = []


# ── Dataset merge (REQ: iterative training) ──────────────────────────────────

class DatasetMergeResult(BaseModel):
    """数据集合并操作的结果"""
    success: bool
    merged_images: int              # 成功合并的图片数
    target_total_images: int        # 合并后目标类别的总图片数
    deleted_source: bool            # 是否删除了源类别
    error: Optional[str] = None
