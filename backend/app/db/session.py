"""
app/db/session.py
-----------------
Async SQLAlchemy engine + session factory for PostgreSQL.

The engine is created at import time using settings.DATABASE_URL. Connection
is lazy (asyncpg pool is established on first use), so a misconfigured or
unreachable DB does NOT crash module import — failures surface at the first
query and are swallowed by the persistence layer's try/except.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging import logger

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    pool_pre_ping=True,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Create tables if missing. MVP: no Alembic migration tracking."""
    from app.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Add missing assist_model_id column if it doesn't exist
        await conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'yoloe_annotation_iterations'
                    AND column_name = 'assist_model_id'
                ) THEN
                    ALTER TABLE yoloe_annotation_iterations
                    ADD COLUMN assist_model_id VARCHAR(36);

                    CREATE INDEX IF NOT EXISTS ix_yoloe_annotation_iterations_assist_model_id
                    ON yoloe_annotation_iterations(assist_model_id);
                END IF;
            END $$;
        """))

        # Lightweight data migration for bundles that were run before the
        # no_target status existed: empty auto-generated YOLO label files were
        # previously counted as annotated positives. Reclassify only zero-box
        # annotated images so training cannot accidentally consume them.
        await conn.execute(text("""
            UPDATE yoloe_dataset_images
            SET annotation_status = 'no_target'
            WHERE annotation_status = 'annotated'
              AND COALESCE(box_count, 0) = 0
        """))
        await conn.execute(text("""
            UPDATE yoloe_categories c
            SET annotated_count = COALESCE(s.annotated, 0),
                status = CASE
                    WHEN c.status = 'trained' THEN c.status
                    WHEN COALESCE(s.total, 0) = 0 THEN 'draft'
                    WHEN COALESCE(s.annotated, 0) > 0 THEN 'ready'
                    ELSE 'annotating'
                END
            FROM (
                SELECT category_id,
                       COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE annotation_status = 'annotated') AS annotated
                FROM yoloe_dataset_images
                GROUP BY category_id
            ) s
            WHERE c.id = s.category_id
        """))
        await conn.execute(text("""
            UPDATE yoloe_llm_annotation_tasks
            SET result_summary = COALESCE(result_summary::jsonb, '{}'::jsonb) || jsonb_build_object(
                    'success', 0,
                    'no_target', success_count,
                    'failed', failed_count,
                    'total_boxes', total_boxes,
                    'legacy_empty_success_reclassified', true
                ),
                success_count = 0
            WHERE success_count > 0
              AND COALESCE(total_boxes, 0) = 0
        """))
        await conn.execute(text("""
            UPDATE yoloe_annotation_iterations
            SET result_summary = COALESCE(result_summary::jsonb, '{}'::jsonb) || jsonb_build_object(
                    'success', 0,
                    'no_target', success_count,
                    'failed', failed_count,
                    'total_boxes', total_boxes,
                    'legacy_empty_success_reclassified', true
                ),
                success_count = 0,
                status = CASE WHEN status = 'ready_for_training' THEN 'failed' ELSE status END,
                error = CASE
                    WHEN status = 'ready_for_training' THEN '本轮未检测到目标。'
                    ELSE error
                END
            WHERE success_count > 0
              AND COALESCE(total_boxes, 0) = 0
        """))
    logger.info("DB schema ensured (detection_tasks)")
