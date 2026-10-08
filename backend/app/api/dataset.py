"""
app/api/dataset.py
------------------
Per-category image upload + YOLO annotation persistence (REQ2).

Raw images are stored under DATASETS_DIR/<category_id>/raw/. Working YOLO
labels are stored under ANNOTATIONS_DIR/<category_id>/<image_id>.txt (the
source of truth for annotation content); the DB row mirrors status + box count.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select

from app.core.config import settings
from app.core.logging import logger
from app.db.models import CategoryRecord, DatasetImageRecord
from app.db.session import AsyncSessionLocal
from app.models.schemas import (
    AnnotationBox, AnnotationPayload, DatasetImageItem, DatasetImportResult,
    DatasetMergeResult,
)

# Cap PIL's decompression-bomb threshold explicitly (P-1): a crafted small file
# can decode to a huge bitmap and exhaust RAM. PIL warns above its ~89 MP default
# and raises above 2×; pin it so the guard is deterministic regardless of env.
try:
    from PIL import Image as _PILImage
    _PILImage.MAX_IMAGE_PIXELS = 100_000_000  # ~100 megapixels
except Exception:  # pragma: no cover - Pillow is always installed
    pass

router = APIRouter()

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".bmp": "image/bmp", ".webp": "image/webp",
}


def _reject_unsafe(component: str) -> None:
    if "/" in component or "\\" in component or ".." in component:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "非法的标识符。")


def _guard_upload_batch(files: list, verb: str = "上传") -> None:
    """Reject an over-large batch up front and refuse when disk is low (P-1)."""
    if len(files) > settings.MAX_UPLOAD_FILES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"单次最多{verb} {settings.MAX_UPLOAD_FILES} 个文件。",
        )
    try:
        free = shutil.disk_usage(settings.DATASETS_DIR).free
    except OSError:
        free = None
    if free is not None and free < settings.MIN_FREE_DISK_BYTES:
        raise HTTPException(
            status.HTTP_507_INSUFFICIENT_STORAGE,
            "磁盘空间不足，暂时无法接收上传，请联系管理员清理后重试。",
        )


def _raw_dir(category_id: str) -> Path:
    return settings.DATASETS_DIR / category_id / "raw"


def _ann_path(category_id: str, image_id: str) -> Path:
    return settings.ANNOTATIONS_DIR / category_id / f"{image_id}.txt"


# ── pre-annotated dataset import helpers ──────────────────────────────────────
# Folders dropped by the dataset-import flow (REQ2, second data source).

_LABEL_DIR_SEGMENTS = {"images", "labels", "image", "label"}


def _match_key(rel_path: str) -> str:
    """Normalize a file's relative path into a key that pairs an image with its
    label across the common YOLO layouts.

    Drops any `images/`|`labels/` path segment and the file extension so that
    `ds/images/train/x.jpg` and `ds/labels/train/x.txt` collapse to the same
    key (`ds/train/x`). Flat layouts (`ds/x.jpg` + `ds/x.txt`) also match.
    """
    norm = (rel_path or "").replace("\\", "/")
    parts = [p for p in norm.split("/") if p and p.lower() not in _LABEL_DIR_SEGMENTS]
    if not parts:
        return ""
    parts[-1] = Path(parts[-1]).stem
    return "/".join(parts)


def _parse_label_text(text: str) -> List[str]:
    """Parse YOLO label lines, force class → 0 (single-class fold) and clamp
    coords to [0, 1]. Box lines (`c cx cy w h`) and polygon lines
    (`c x1 y1 x2 y2 ...`) are both kept; invalid lines are dropped. Returns the
    normalized label lines (each prefixed with class `0`)."""
    out: List[str] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(vals) == 4:  # box: cx cy w h — require positive size
            if vals[2] <= 0 or vals[3] <= 0:
                continue
        elif not (len(vals) >= 6 and len(vals) % 2 == 0):  # else must be polygon
            continue
        vals = [min(1.0, max(0.0, v)) for v in vals]
        out.append("0 " + " ".join(f"{v:.6f}" for v in vals))
    return out



async def _recompute_counts(session, category_id: str) -> None:
    total = await session.scalar(
        select(func.count()).select_from(DatasetImageRecord).where(
            DatasetImageRecord.category_id == category_id
        )
    ) or 0
    annotated = await session.scalar(
        select(func.count()).select_from(DatasetImageRecord).where(
            DatasetImageRecord.category_id == category_id,
            DatasetImageRecord.annotation_status == "annotated",
        )
    ) or 0
    cat = await session.get(CategoryRecord, category_id)
    if cat is None:
        return
    cat.image_count = total
    cat.annotated_count = annotated
    if cat.status != "trained":
        cat.status = "draft" if total == 0 else ("ready" if annotated > 0 else "annotating")


async def _require_category(session, category_id: str) -> CategoryRecord:
    cat = await session.get(CategoryRecord, category_id)
    if cat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "类别不存在。")
    return cat


@router.post(
    "/categories/{category_id}/images",
    response_model=List[DatasetImageItem],
    status_code=status.HTTP_201_CREATED,
    summary="上传一张或多张图片到类别数据集",
)
async def upload_images(
    category_id: str,
    files: List[UploadFile] = File(..., description="图片文件（可多选）"),
) -> List[DatasetImageItem]:
    _guard_upload_batch(files, verb="上传")
    _raw_dir(category_id).mkdir(parents=True, exist_ok=True)
    created: List[DatasetImageRecord] = []

    try:
        async with AsyncSessionLocal() as session:
            await _require_category(session, category_id)

            for f in files:
                rec = await _store_one_image(session, category_id, f)
                if rec is not None:
                    created.append(rec)

            if not created:
                raise HTTPException(
                    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    f"没有有效图片。支持：{', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}",
                )

            await session.flush()
            await _recompute_counts(session, category_id)
            await session.commit()
            for r in created:
                await session.refresh(r)
            return [DatasetImageItem.model_validate(r) for r in created]
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"upload_images failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc


async def _store_one_image(session, category_id: str, f: UploadFile):
    """把一张上传图片持久化进类别数据集：带大小上限流式落盘、校验可解码（读宽高）、
    向 session 添加一条 pending 的 DatasetImageRecord。返回该记录；若文件被跳过
    （类型不符 / 超限 / 不可读）返回 None。调用方负责 flush/commit 与 _recompute_counts。

    供 upload_images 与一站式自动标注端点共用，保证两条路径产出**完全相同**的磁盘 +
    数据库表示（raw 原图 + pending 行），下游标注/训练逻辑无需区分来源。
    """
    from PIL import Image  # local import; Pillow is a dependency

    suffix = Path(f.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        logger.info(f"skip non-image upload: {f.filename}")
        return None

    image_id = uuid.uuid4().hex
    dest = _raw_dir(category_id) / f"{image_id}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    oversize = False
    try:
        async with aiofiles.open(dest, "wb") as out:
            while chunk := await f.read(4 * 1024 * 1024):
                total += len(chunk)
                if total > settings.MAX_IMAGE_BYTES:
                    oversize = True
                    break
                await out.write(chunk)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.warning(f"image write failed {f.filename}: {exc}")
        return None
    if oversize:
        dest.unlink(missing_ok=True)
        logger.info(
            f"skip oversize image {f.filename} "
            f"(> {settings.MAX_IMAGE_BYTES // (1024 * 1024)} MB)"
        )
        return None

    w = h = 0
    try:
        with Image.open(dest) as im:
            w, h = im.size
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.warning(f"invalid image {f.filename}: {exc}")
        return None

    rec = DatasetImageRecord(
        id=image_id,
        category_id=category_id,
        filename=f.filename or f"{image_id}{suffix}",
        stored_path=str(dest.resolve()),
        width=w, height=h,
        annotation_status="pending",
        box_count=0,
    )
    session.add(rec)
    return rec


@router.post(
    "/categories/{category_id}/dataset/import",
    response_model=DatasetImportResult,
    status_code=status.HTTP_201_CREATED,
    summary="导入已标注的 YOLO 数据集文件夹（标注框折叠为单类别）",
)
async def import_dataset(
    category_id: str,
    files: List[UploadFile] = File(..., description="数据集中的图片与 .txt 标注文件"),
    rel_paths: List[str] = Form(
        default=[],
        description="每个文件的相对路径（webkitRelativePath），与 files 顺序一一对应；"
                    "缺省时回退用文件名匹配。",
    ),
) -> DatasetImportResult:
    """Import a pre-annotated YOLO dataset folder into a category — the second
    dataset source alongside in-browser annotation. Images land in the SAME
    on-disk + DB representation the manual-annotation flow produces (raw image +
    working YOLO label + an `annotated` DatasetImageRecord), so training and
    everything downstream work unchanged. All boxes are folded to the single
    category class (class 0)."""
    from PIL import Image  # local import; Pillow is a dependency

    _reject_unsafe(category_id)
    _guard_upload_batch(files, verb="导入")

    # Fail fast before writing any files if the category is missing.
    try:
        async with AsyncSessionLocal() as session:
            await _require_category(session, category_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"import_dataset category lookup failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc

    # ── Pass 1: classify files. Stream images to disk under uuid names; read
    # the (tiny) label texts into memory. Order-independent — images and labels
    # are paired by match key afterwards, so a label may precede or follow its
    # image in the multipart stream.
    raw_dir = _raw_dir(category_id)
    raw_dir.mkdir(parents=True, exist_ok=True)
    labels_by_key: dict[str, List[str]] = {}
    image_entries: List[dict] = []  # {image_id, key, stored_path, filename, w, h}
    skipped = 0

    for i, f in enumerate(files):
        rel = (rel_paths[i] if i < len(rel_paths) else "") or (f.filename or "")
        suffix = Path(rel).suffix.lower()
        key = _match_key(rel)

        if suffix == ".txt":
            try:
                raw = await f.read()
                lines = _parse_label_text(raw.decode("utf-8", errors="replace"))
            except Exception as exc:
                logger.info(f"skip unreadable label {rel}: {exc}")
                continue
            if key:
                labels_by_key.setdefault(key, lines)
            continue

        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            skipped += 1
            continue

        image_id = uuid.uuid4().hex
        dest = raw_dir / f"{image_id}{suffix}"
        try:
            total = 0
            oversize = False
            async with aiofiles.open(dest, "wb") as out:
                while chunk := await f.read(4 * 1024 * 1024):
                    total += len(chunk)
                    if total > settings.MAX_IMAGE_BYTES:
                        oversize = True
                        break
                    await out.write(chunk)
            if oversize:
                dest.unlink(missing_ok=True)
                logger.info(
                    f"skip oversize image {rel} "
                    f"(> {settings.MAX_IMAGE_BYTES // (1024 * 1024)} MB)"
                )
                skipped += 1
                continue
            with Image.open(dest) as im:
                w, h = im.size
        except Exception as exc:
            dest.unlink(missing_ok=True)
            logger.warning(f"skip invalid image {rel}: {exc}")
            skipped += 1
            continue

        image_entries.append({
            "image_id": image_id, "key": key, "stored_path": str(dest.resolve()),
            "filename": rel or f"{image_id}{suffix}", "w": w, "h": h,
        })

    if not image_entries:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"未发现有效图片。支持：{', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}",
        )

    # ── Pass 2: write per-image YOLO labels + DB rows. Every imported image is
    # `annotated`; one with no/empty matched label is a background sample.
    (settings.ANNOTATIONS_DIR / category_id).mkdir(parents=True, exist_ok=True)
    with_annotation = 0
    try:
        async with AsyncSessionLocal() as session:
            await _require_category(session, category_id)
            for e in image_entries:
                lines = labels_by_key.get(e["key"], [])
                _ann_path(category_id, e["image_id"]).write_text(
                    "\n".join(lines), encoding="utf-8"
                )
                if lines:
                    with_annotation += 1
                session.add(DatasetImageRecord(
                    id=e["image_id"],
                    category_id=category_id,
                    filename=e["filename"],
                    stored_path=e["stored_path"],
                    width=e["w"], height=e["h"],
                    annotation_status="annotated",
                    box_count=len(lines),
                ))
            await session.flush()
            await _recompute_counts(session, category_id)
            await session.commit()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"import_dataset persist failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc

    imported = len(image_entries)
    background = imported - with_annotation
    logger.info(
        f"import_dataset cat={category_id}: imported={imported} "
        f"annotated={with_annotation} background={background} skipped={skipped}"
    )
    return DatasetImportResult(
        imported_images=imported,
        with_annotation=with_annotation,
        background=background,
        skipped_files=skipped,
        message=(
            f"导入 {imported} 张图片（{with_annotation} 张含标注，{background} 张背景）"
            + (f"，忽略 {skipped} 个无效文件" if skipped else "")
        ),
    )


@router.get(
    "/categories/{category_id}/images",
    response_model=List[DatasetImageItem],
    summary="列出类别下的图片",
)
async def list_images(category_id: str) -> List[DatasetImageItem]:
    try:
        async with AsyncSessionLocal() as session:
            await _require_category(session, category_id)
            rows = (
                await session.execute(
                    select(DatasetImageRecord)
                    .where(DatasetImageRecord.category_id == category_id)
                    .order_by(DatasetImageRecord.created_at.asc())
                )
            ).scalars().all()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"list_images failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc
    return [DatasetImageItem.model_validate(r) for r in rows]


@router.get(
    "/categories/{category_id}/images/{image_id}/file",
    response_class=FileResponse,
    summary="获取原始图片（用于标注）",
)
async def get_image_file(category_id: str, image_id: str):
    _reject_unsafe(category_id)
    _reject_unsafe(image_id)
    raw_dir = _raw_dir(category_id)
    matches = list(raw_dir.glob(f"{image_id}.*"))
    if not matches:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "图片不存在。")
    p = matches[0]
    return FileResponse(path=str(p), media_type=_MIME.get(p.suffix.lower(), "image/jpeg"))


@router.get(
    "/categories/{category_id}/images/{image_id}/annotation",
    response_model=AnnotationPayload,
    summary="读取某图片的 YOLO 标注",
)
async def get_annotation(category_id: str, image_id: str) -> AnnotationPayload:
    _reject_unsafe(category_id)
    _reject_unsafe(image_id)
    p = _ann_path(category_id, image_id)
    if not p.exists():
        return AnnotationPayload(boxes=[])
    boxes: List[AnnotationBox] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(x) for x in parts[1:5])
            boxes.append(AnnotationBox(cls=cls, cx=cx, cy=cy, w=w, h=h))
        except (ValueError, TypeError):
            continue
    return AnnotationPayload(boxes=boxes)


@router.put(
    "/categories/{category_id}/images/{image_id}/annotation",
    response_model=DatasetImageItem,
    summary="保存某图片的 YOLO 标注（空列表=背景样本）",
)
async def save_annotation(
    category_id: str, image_id: str, body: AnnotationPayload
) -> DatasetImageItem:
    _reject_unsafe(category_id)
    _reject_unsafe(image_id)
    try:
        async with AsyncSessionLocal() as session:
            await _require_category(session, category_id)
            img = await session.get(DatasetImageRecord, image_id)
            if img is None or img.category_id != category_id:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "图片不存在。")

            ann = _ann_path(category_id, image_id)
            ann.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                f"{b.cls} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}" for b in body.boxes
            ]
            ann.write_text("\n".join(lines), encoding="utf-8")

            if body.boxes:
                img.annotation_status = "annotated"
                img.box_count = len(body.boxes)
            else:
                # Manual save with no boxes means this frame has been checked but
                # contains no target. Keep it out of the positive training set.
                img.annotation_status = "no_target"
                img.box_count = 0
            await session.flush()
            await _recompute_counts(session, category_id)
            await session.commit()
            await session.refresh(img)
            return DatasetImageItem.model_validate(img)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"save_annotation failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc


@router.delete(
    "/categories/{category_id}/images/{image_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="从数据集删除一张图片",
)
async def delete_image(category_id: str, image_id: str):
    _reject_unsafe(category_id)
    _reject_unsafe(image_id)
    try:
        async with AsyncSessionLocal() as session:
            img = await session.get(DatasetImageRecord, image_id)
            if img is None or img.category_id != category_id:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "图片不存在。")
            await session.delete(img)
            await session.flush()
            await _recompute_counts(session, category_id)
            await session.commit()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"delete_image failed: {exc}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "数据库不可用。") from exc

    for p in _raw_dir(category_id).glob(f"{image_id}.*"):
        p.unlink(missing_ok=True)
    _ann_path(category_id, image_id).unlink(missing_ok=True)
    return None


@router.post(
    "/categories/merge",
    response_model=DatasetMergeResult,
    summary="合并两个类别的数据集",
)
async def merge_datasets(
    target_category_id: str = Form(..., description="目标类别ID（数据合并到此类别）"),
    source_category_id: str = Form(..., description="源类别ID（数据来源）"),
    delete_source: bool = Form(False, description="合并后是否删除源类别"),
) -> DatasetMergeResult:
    """
    将源类别的数据集合并到目标类别。
    
    步骤：
    1. 验证两个类别存在
    2. 复制源类别的原图到目标类别
    3. 复制源类别的标注到目标类别  
    4. 更新数据库记录（将源图片的category_id改为目标）
    5. 更新目标类别的统计信息
    6. 如果 delete_source=True，删除源类别
    """
    from app.models.schemas import DatasetMergeResult
    
    _reject_unsafe(target_category_id)
    _reject_unsafe(source_category_id)
    
    if target_category_id == source_category_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "目标类别和源类别不能相同",
        )
    
    try:
        async with AsyncSessionLocal() as session:
            # 1. 验证类别存在
            target_cat = await session.get(CategoryRecord, target_category_id)
            source_cat = await session.get(CategoryRecord, source_category_id)
            
            if target_cat is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, f"目标类别不存在")
            if source_cat is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, f"源类别不存在")
            
            # 2. 获取源类别的所有图片
            source_images = (
                await session.execute(
                    select(DatasetImageRecord).where(
                        DatasetImageRecord.category_id == source_category_id
                    )
                )
            ).scalars().all()
            
            if not source_images:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    "源类别没有图片，无需合并",
                )
            
            merged_count = 0
            target_raw_dir = _raw_dir(target_category_id)
            target_raw_dir.mkdir(parents=True, exist_ok=True)
            
            # 3. 复制文件并更新数据库
            for img in source_images:
                try:
                    # 复制原图
                    source_raw_path = Path(img.stored_path)
                    if source_raw_path.exists():
                        target_raw_path = target_raw_dir / source_raw_path.name
                        shutil.copy2(source_raw_path, target_raw_path)
                        
                        # 更新数据库记录中的路径
                        img.stored_path = str(target_raw_path)
                    
                    # 复制标注文件（如果存在）
                    source_ann = _ann_path(source_category_id, img.id)
                    if source_ann.exists():
                        target_ann = _ann_path(target_category_id, img.id)
                        target_ann.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_ann, target_ann)
                    
                    # 更新数据库记录的类别ID
                    img.category_id = target_category_id
                    session.add(img)
                    merged_count += 1
                    
                except Exception as e:
                    logger.warning(f"合并图片 {img.id} 失败: {e}")
                    continue
            
            # 4. 重新计算目标类别统计
            await _recompute_counts(session, target_category_id)
            
            # 5. 删除源类别（如果需要）
            deleted = False
            if delete_source and merged_count > 0:
                # 删除源类别记录
                await session.delete(source_cat)
                deleted = True
                
                # 删除源类别的文件夹（如果为空或已移走文件）
                source_raw_dir = _raw_dir(source_category_id)
                if source_raw_dir.exists():
                    try:
                        shutil.rmtree(source_raw_dir, ignore_errors=True)
                    except Exception as e:
                        logger.warning(f"删除源类别目录失败: {e}")
                
                # 删除源类别的标注目录
                source_ann_dir = settings.ANNOTATIONS_DIR / source_category_id
                if source_ann_dir.exists():
                    try:
                        shutil.rmtree(source_ann_dir, ignore_errors=True)
                    except Exception as e:
                        logger.warning(f"删除源标注目录失败: {e}")
            
            await session.commit()
            
            # 6. 获取合并后的总数
            target_total = await session.scalar(
                select(func.count()).select_from(DatasetImageRecord).where(
                    DatasetImageRecord.category_id == target_category_id
                )
            ) or 0
            
            return DatasetMergeResult(
                success=True,
                merged_images=merged_count,
                target_total_images=target_total,
                deleted_source=deleted,
            )
            
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"合并数据集失败: {exc}")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"合并数据集失败: {str(exc)}",
        ) from exc
