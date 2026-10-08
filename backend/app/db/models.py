"""
app/db/models.py
----------------
SQLAlchemy ORM models for task history persistence.

Schema is created on startup via init_db() — no Alembic for the MVP.
Bump table-name or add a migration tool when the schema starts to evolve.

NOTE: init_db() uses Base.metadata.create_all, which ONLY creates missing
tables. It does NOT alter existing tables — once a table ships, changing a
column requires Alembic (planned: Phase 3). Adding brand-new tables (as the
yoloe_* tables below) is safe.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class TaskRecord(Base):
    """One row per detection task. Mirrors TaskState plus timestamps."""

    __tablename__ = "detection_tasks"

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    video_id: Mapped[str] = mapped_column(String(36), nullable=False)
    video_filename: Mapped[Optional[str]] = mapped_column(String(512))
    prompt: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    total_frames: Mapped[int] = mapped_column(Integer, default=0)
    processed_frames: Mapped[int] = mapped_column(Integer, default=0)

    error: Mapped[Optional[str]] = mapped_column(Text)
    zip_ready: Mapped[bool] = mapped_column(Boolean, default=False)
    early_terminated: Mapped[bool] = mapped_column(Boolean, default=False)
    termination_reason: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


# ════════════════════════════════════════════════════════════════════════════
# YOLOE custom-training workflow (REQ1/REQ2/REQ3)
# Categories → uploaded images → YOLO annotation → training jobs → trained
# models. Large artifacts (images, weights) live on disk; these tables hold
# metadata + paths only. Created via create_all alongside detection_tasks.
# ════════════════════════════════════════════════════════════════════════════


class CategoryRecord(Base):
    """A user-created class to train. Its name becomes the trained model name."""

    __tablename__ = "yoloe_categories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    # draft | annotating | ready | trained
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    image_count: Mapped[int] = mapped_column(Integer, default=0)
    annotated_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DatasetImageRecord(Base):
    """One uploaded image in a category's dataset. Annotation content is the
    YOLO .txt on disk; this row mirrors status + box count for the UI."""

    __tablename__ = "yoloe_dataset_images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # filename stem (uuid)
    category_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)  # original name
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    # pending | annotated | skipped
    annotation_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    box_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class TrainingJobRecord(Base):
    """One training run. Mirrors the async-task pattern of TaskRecord."""

    __tablename__ = "yoloe_training_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    category_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)

    # pending | running | finished | failed | cancelled
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs: Mapped[int] = mapped_column(Integer, default=0)

    dataset_yaml: Mapped[Optional[str]] = mapped_column(String(1024))
    base_model: Mapped[Optional[str]] = mapped_column(String(256))
    params: Mapped[Optional[dict]] = mapped_column(JSON)

    metric_map50: Mapped[Optional[float]] = mapped_column(Float)
    metric_map50_95: Mapped[Optional[float]] = mapped_column(Float)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON)

    best_pt_path: Mapped[Optional[str]] = mapped_column(String(1024))
    pid: Mapped[Optional[int]] = mapped_column(Integer)
    error: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class TrainedModelRecord(Base):
    """Registry of trained models — backs the REQ3 model list (newest first)."""

    __tablename__ = "yoloe_trained_models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # model_id
    name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)  # == category name
    version: Mapped[int] = mapped_column(Integer, default=1)
    category_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    training_job_id: Mapped[str] = mapped_column(String(36), nullable=False)

    weights_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    base_model: Mapped[Optional[str]] = mapped_column(String(256))
    class_names: Mapped[Optional[dict]] = mapped_column(JSON)  # {0: name, ...} from model.names
    dataset_yaml: Mapped[Optional[str]] = mapped_column(String(1024))
    num_images: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON)

    trained_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    trained_finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class OperationLog(Base):
    """Lightweight audit trail for destructive/administrative actions (R-10).

    Brand-new table — created automatically by create_all (init_db) with no
    migration. Records what was deleted/affected and when so an accidental
    history wipe can be reconstructed/explained. No auth yet, so there is no
    actor column; add one when authentication lands.
    """

    __tablename__ = "operation_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    detail: Mapped[Optional[str]] = mapped_column(Text)   # human/JSON summary
    affected_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class AnnotationIterationRecord(Base):
    """One logical round of LLM-assisted annotation for a training category.

    Kept as a brand-new table so create_all() can introduce it without altering
    shipped tables. The actual async execution remains LLMAnnotationTaskRecord;
    this row adds the user-facing iteration number and the handoff point to
    training.
    """

    __tablename__ = "yoloe_annotation_iterations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    category_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    iteration_no: Mapped[int] = mapped_column(Integer, default=1, index=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)

    # pending | running | review | ready_for_training | training | trained | failed | cancelled
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    # pending_only | no_target_only | failed_only | all
    selection_mode: Mapped[str] = mapped_column(String(32), default="pending_only")
    # skip_existing | overwrite
    write_policy: Mapped[str] = mapped_column(String(32), default="skip_existing")

    llm_task_id: Mapped[Optional[str]] = mapped_column(String(36), index=True)
    training_job_id: Mapped[Optional[str]] = mapped_column(String(36), index=True)
    assist_model_id: Mapped[Optional[str]] = mapped_column(String(36), index=True)  # 辅助标注的已训练模型ID

    total_images: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    total_boxes: Mapped[int] = mapped_column(Integer, default=0)

    result_summary: Mapped[Optional[dict]] = mapped_column(JSON)
    error: Mapped[Optional[str]] = mapped_column(Text)

    @property
    def no_target_count(self) -> int:
        """Images processed successfully by the LLM but with no target found."""
        summary = self.result_summary or {}
        try:
            return int(summary.get("no_target") or 0)
        except (TypeError, ValueError):
            return 0

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class LLMAnnotationTaskRecord(Base):
    """LLM自动标注任务 - 使用大模型API批量标注数据集"""

    __tablename__ = "yoloe_llm_annotation_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    category_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)

    # LLM配置
    llm_provider: Mapped[str] = mapped_column(String(32), nullable=False)  # 固定为 qwen（后端配置）
    llm_model: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)  # 用户提示词

    # 任务状态: pending | running | finished | failed | cancelled
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)

    # 统计信息
    total_images: Mapped[int] = mapped_column(Integer, default=0)
    processed_images: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    total_boxes: Mapped[int] = mapped_column(Integer, default=0)  # 检测到的总框数

    # 执行信息
    error: Mapped[Optional[str]] = mapped_column(Text)
    result_summary: Mapped[Optional[dict]] = mapped_column(JSON)  # 详细结果

    @property
    def no_target_count(self) -> int:
        """Images processed successfully by the LLM but with no target found."""
        summary = self.result_summary or {}
        try:
            return int(summary.get("no_target") or 0)
        except (TypeError, ValueError):
            return 0

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
