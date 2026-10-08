"""
app/services/pipeline.py
-------------------------
The main video processing pipeline.

Pipeline flow per task:
  1. Open video with OpenCV
  2. For every frame:
       a) If it's a "detection frame"  →  run detector + tracker.update()
       b) Otherwise (tracking-only frame) →  tracker.update(last_detections)
  3. Draw visualized result on the frame
  4. Save annotated frame as JPEG
  5. Push FrameResult (with base64 thumbnail) to the task queue → SSE
  6. After all frames: package results into a ZIP

Detection vs Tracking:
  - Detection runs every DETECTION_INTERVAL frames (configurable).
  - Between detection frames, we pass the LAST set of detections to ByteTrack,
    which propagates them with a Kalman filter — MUCH cheaper than GPU inference.
  - This gives ~5x throughput improvement at default interval=5.
"""

from __future__ import annotations

import asyncio
import json
import os
import zipfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import BoundingBox, Detection, FrameResult
from app.services.best_shot import BestShotScorer, BestShotSelector
from app.services.detector import RawDetection, get_detector, get_trained_detector
from app.services.frame_analyzer import create_frame_analyzer
from app.services.frame_quality import create_quality_checker
from app.services.frame_save_gate import FrameSaveGate
from app.services.fusion_engine import FusionEngine
from app.services.color_filter import create_color_filter, get_preset_color_filter
from app.services.task_manager import TaskManager
from app.services.tracker import TrackedObject, create_tracker
from app.services.visualizer import draw_detections, frame_to_base64
from app.services.vlm_service import get_vlm_service
from app.utils.video_utils import format_timestamp, get_video_info


# ────────────────────────────────────────────────────────────────────────────
# Main pipeline coroutine
# ────────────────────────────────────────────────────────────────────────────

async def run_detection_pipeline(
    task_id: str,
    video_path: Path,
    prompt: str,
    task_manager: TaskManager,
    vlm_query: str = "",
    enable_vlm: bool = False,
    detection_interval: Optional[int] = None,
    box_threshold: Optional[float] = None,
    text_threshold: Optional[float] = None,
    original_prompt: str = "",
    label_mapping: Optional[dict] = None,
    color_filters: Optional[List[dict]] = None,
    weights_path: Optional[str] = None,
) -> None:
    """
    Async coroutine that processes a video file and streams results.

    Designed to be run inside a background asyncio task via
    `asyncio.create_task(...)`.

    Heavy work (GPU inference) is offloaded to a thread pool via
    `asyncio.to_thread(...)` so it does NOT block the event loop.

    Args:
        original_prompt: Original user input (Chinese)
        label_mapping: Dict mapping English labels to Chinese labels
        color_filters: List of color filter rules for post-processing
        weights_path: When set, detect with this trained model (best.pt) using
            its baked-in classes instead of the open-vocabulary singleton.
    """
    # ── Config ────────────────────────────────────────────────────────────
    det_interval = detection_interval or settings.DETECTION_INTERVAL
    box_thr = box_threshold or settings.BOX_THRESHOLD
    txt_thr = text_threshold or settings.TEXT_THRESHOLD

    # ── Output directory for this task ────────────────────────────────────
    task_results_dir = settings.RESULTS_DIR / task_id
    task_results_dir.mkdir(parents=True, exist_ok=True)

    # ── Video info ────────────────────────────────────────────────────────
    try:
        info = get_video_info(video_path)
    except Exception as exc:
        logger.error(f"[{task_id}] Failed to read video: {exc}")
        await task_manager.set_failed(task_id, str(exc))
        await task_manager.push_error(task_id, str(exc))
        task_manager.release_results(task_id)
        return

    await task_manager.set_running(task_id, info["total_frames"])
    logger.info(
        f"[{task_id}] Processing video: {video_path.name} | "
        f"frames={info['total_frames']} fps={info['fps']:.2f}"
    )

    # ── Acquire GPU semaphore (limits concurrent GPU tasks) ───────────────
    loop = asyncio.get_running_loop()
    async with task_manager.semaphore:
        try:
            await asyncio.to_thread(
                _sync_pipeline,
                task_id=task_id,
                video_path=video_path,
                prompt=prompt,
                vlm_query=vlm_query,
                enable_vlm=enable_vlm,
                task_results_dir=task_results_dir,
                task_manager=task_manager,
                loop=loop,
                fps=info["fps"],
                total_frames=info["total_frames"],
                det_interval=det_interval,
                box_thr=box_thr,
                txt_thr=txt_thr,
                original_prompt=original_prompt,
                label_mapping=label_mapping,
                color_filters=color_filters,
                weights_path=weights_path,
            )
        except Exception as exc:
            logger.exception(f"[{task_id}] Pipeline error: {exc}")
            await task_manager.set_failed(task_id, str(exc))
            await task_manager.push_error(task_id, str(exc))
            task_manager.cleanup_flags(task_id)
            task_manager.release_results(task_id)
            return

    # ── Cancel path: skip ZIP, emit cancelled, close stream ──────────────
    if task_manager.is_cancelled(task_id):
        await task_manager.set_cancelled(task_id)
        await task_manager.push_cancelled(task_id)
        await task_manager.push_done(task_id)
        task_manager.cleanup_flags(task_id)
        task_manager.release_results(task_id)
        logger.info(f"[{task_id}] Pipeline cancelled.")
        return

    # ── Manual termination path: package ZIP with partial results ────────
    if task_manager.is_terminated(task_id):
        state = task_manager.get_task(task_id)
        termination_reason = f"User manually terminated after processing {state.processed_frames} frames"
        await task_manager.set_early_terminated(task_id, termination_reason)
        await task_manager.push_packaging(task_id)
        try:
            await asyncio.to_thread(
                _package_zip,
                task_id=task_id,
                task_results_dir=task_results_dir,
                task_state=state,
            )
        except Exception as exc:
            logger.error(f"[{task_id}] ZIP creation failed: {exc}")
            await task_manager.set_failed(task_id, f"ZIP error: {exc}")

        await task_manager.push_early_terminated(task_id, termination_reason)
        await task_manager.push_done(task_id)
        task_manager.cleanup_flags(task_id)
        task_manager.release_results(task_id)
        logger.info(f"[{task_id}] Pipeline manually terminated and packaged.")
        return

    # ── Package ZIP ───────────────────────────────────────────────────────
    # Notify clients first — ZIP packaging can take minutes for long videos
    # and we don't want the SSE stream to look dead.
    await task_manager.push_packaging(task_id)
    try:
        await asyncio.to_thread(
            _package_zip,
            task_id=task_id,
            task_results_dir=task_results_dir,
            task_state=task_manager.get_task(task_id),
        )
        await task_manager.set_finished(task_id)
    except Exception as exc:
        logger.error(f"[{task_id}] ZIP creation failed: {exc}")
        await task_manager.set_failed(task_id, f"ZIP error: {exc}")

    await task_manager.push_done(task_id)
    task_manager.cleanup_flags(task_id)
    task_manager.release_results(task_id)
    logger.info(f"[{task_id}] Pipeline complete.")


# ────────────────────────────────────────────────────────────────────────────
# Synchronous heavy-lifting (runs in thread pool)
# ────────────────────────────────────────────────────────────────────────────

def _sync_pipeline(
    task_id: str,
    video_path: Path,
    prompt: str,
    vlm_query: str,
    enable_vlm: bool,
    task_results_dir: Path,
    task_manager: TaskManager,
    loop: asyncio.AbstractEventLoop,
    fps: float,
    total_frames: int,
    det_interval: int,
    box_thr: float,
    txt_thr: float,
    original_prompt: str = "",
    label_mapping: Optional[dict] = None,
    color_filters: Optional[List[dict]] = None,
    weights_path: Optional[str] = None,
) -> None:
    detector = get_trained_detector(weights_path) if weights_path else get_detector()
    tracker = create_tracker(fps=fps)
    fusion = FusionEngine()
    save_gate = FrameSaveGate(
        mode=settings.SAVE_FRAMES_MODE,
        cooldown_sec=settings.TRACK_SAVE_COOLDOWN_SEC,
    )
    # Best-shot: one optimal snapshot per target, chosen across its whole track
    # lifetime and written at the end (see _flush_best_shots). When active, the
    # per-frame save_gate is bypassed — nothing is written inline.
    best_shot_mode = settings.SAVE_FRAMES_MODE == "best_shot"
    best_shot_selector: Optional[BestShotSelector] = None
    best_shot_scorer: Optional[BestShotScorer] = None
    if best_shot_mode:
        best_shot_selector = BestShotSelector(min_score=settings.BEST_SHOT_MIN_SCORE)
        best_shot_scorer = BestShotScorer(
            w_sharpness=settings.BEST_SHOT_WEIGHT_SHARPNESS,
            w_area=settings.BEST_SHOT_WEIGHT_AREA,
            w_confidence=settings.BEST_SHOT_WEIGHT_CONFIDENCE,
            sharpness_ref=settings.BEST_SHOT_SHARPNESS_REF,
            area_ref=settings.BEST_SHOT_AREA_REF,
            edge_margin_px=settings.BEST_SHOT_EDGE_MARGIN_PX,
            edge_penalty=settings.BEST_SHOT_EDGE_PENALTY,
        )
        logger.info(f"[{task_id}] Best-shot mode: one optimal snapshot per target.")
    vlm = get_vlm_service() if enable_vlm else None

    # Use original Chinese prompt for display if available
    display_label = original_prompt if original_prompt else prompt
    if not label_mapping:
        label_mapping = {}

    # Initialize frame quality checker
    quality_checker = None
    if settings.ENABLE_FRAME_QUALITY_CHECK:
        quality_checker = create_quality_checker(
            black_threshold=settings.BLACK_FRAME_THRESHOLD,
            dark_threshold=settings.DARK_FRAME_THRESHOLD,
            bright_threshold=settings.BRIGHT_FRAME_THRESHOLD,
            min_std_dev=settings.MIN_FRAME_STD_DEV,
        )
        logger.info(
            f"[{task_id}] Frame quality check enabled: "
            f"black<{settings.BLACK_FRAME_THRESHOLD}, "
            f"dark<{settings.DARK_FRAME_THRESHOLD}, "
            f"bright>{settings.BRIGHT_FRAME_THRESHOLD}"
        )

    # Initialize color filter
    color_filter = create_color_filter(color_filters)
    if color_filter.enabled:
        logger.info(
            f"[{task_id}] Color filter enabled with {len(color_filters)} rules"
        )

    # Initialize adaptive keyframe analyzer
    frame_analyzer = None
    if settings.ENABLE_ADAPTIVE_KEYFRAME:
        frame_analyzer = create_frame_analyzer(
            diff_threshold=settings.FRAME_DIFF_THRESHOLD,
            hash_threshold=settings.PHASH_THRESHOLD,
            hist_threshold=settings.HIST_THRESHOLD,
            min_interval=settings.MIN_DETECTION_INTERVAL,
            max_interval=settings.MAX_DETECTION_INTERVAL,
        )
        logger.info(
            f"[{task_id}] Adaptive keyframe detection enabled: "
            f"interval=[{settings.MIN_DETECTION_INTERVAL}, {settings.MAX_DETECTION_INTERVAL}], "
            f"diff={settings.FRAME_DIFF_THRESHOLD}, "
            f"hash={settings.PHASH_THRESHOLD}, "
            f"hist={settings.HIST_THRESHOLD}"
        )

    if enable_vlm:
        logger.info(f"[{task_id}] VLM verification enabled, query='{vlm_query}'")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    frame_idx = 0
    last_raw_detections: List[RawDetection] = []

    # Statistics for adaptive keyframe detection
    detection_count = 0
    skip_count = 0

    try:
        while True:
            task_manager.wait_if_paused(task_id)
            if task_manager.is_cancelled(task_id):
                logger.info(f"[{task_id}] Cancelled at frame {frame_idx}.")
                break
            if task_manager.is_terminated(task_id):
                logger.info(f"[{task_id}] Manually terminated at frame {frame_idx}.")
                break

            ret, frame = cap.read()
            if not ret:
                break

            ts_seconds = frame_idx / fps
            ts_str = format_timestamp(ts_seconds)

            # Check frame quality before processing
            if quality_checker is not None:
                is_good, quality_reason = quality_checker.check_frame_fast(frame)
                if not is_good:
                    skip_count += 1
                    if frame_idx % 100 == 0:  # Log periodically to avoid spam
                        logger.debug(
                            f"[{task_id}] Frame {frame_idx} skipped: {quality_reason}"
                        )
                    frame_idx += 1
                    continue

            # Determine if detection is needed
            is_detection_frame = False
            detection_reason = ""

            if frame_analyzer is not None:
                # Adaptive interval mode
                should_detect, reason, next_interval = frame_analyzer.should_detect(frame)
                is_detection_frame = should_detect
                detection_reason = reason

                if should_detect:
                    # Update interval for next detection cycle
                    frame_analyzer.update_interval(next_interval)
                    detection_count += 1

                    if frame_idx > 0:
                        logger.debug(
                            f"[{task_id}] Frame {frame_idx}: Detection triggered "
                            f"(reason={reason}, next_interval={next_interval})"
                        )
            else:
                # Fixed interval mode (original behavior)
                is_detection_frame = (frame_idx % det_interval == 0)
                detection_reason = "fixed_interval"

            if is_detection_frame:
                raw_detections = detector.predict(
                    image=frame,
                    prompt=prompt,
                    box_threshold=box_thr,
                    text_threshold=txt_thr,
                )

                # Apply color penalty to raw detections (降权而非过滤)
                if color_filter.enabled and raw_detections:
                    detections_data = [
                        {
                            "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                            "label": d.label, "score": d.score,
                        }
                        for d in raw_detections
                    ]
                    validation_results = color_filter.validate_batch(frame, detections_data)

                    # Apply penalty factors to detection scores
                    penalty_count = 0
                    for det, (is_valid, penalty_factor) in zip(raw_detections, validation_results):
                        det.color_penalty = penalty_factor
                        if penalty_factor < 1.0:
                            penalty_count += 1

                    if penalty_count > 0:
                        logger.debug(
                            f"[{task_id}] Frame {frame_idx}: Color penalty applied to "
                            f"{penalty_count}/{len(raw_detections)} detections"
                        )

                last_raw_detections = raw_detections

                if not frame_analyzer:
                    detection_count += 1
            else:
                raw_detections = last_raw_detections
                skip_count += 1

            h, w = frame.shape[:2]
            tracked_objects = tracker.update(raw_detections, image_shape=(h, w))

            # Map English labels back to Chinese
            for obj in tracked_objects:
                if obj.label in label_mapping:
                    obj.label = label_mapping[obj.label]
                elif display_label:
                    # Fallback: use original prompt as label
                    obj.label = display_label

            fusion.update_tracks(tracked_objects, current_time=ts_seconds)

            # VLM verification: only on detection frames and for tracks needing verification
            if vlm and is_detection_frame and tracked_objects:
                to_verify = fusion.get_tracks_needing_verification(
                    tracked_objects, current_time=ts_seconds,
                )
                for obj in to_verify[:settings.VLM_MAX_CONCURRENT]:
                    try:
                        crop = vlm.crop_detection(
                            frame, obj.x1, obj.y1, obj.x2, obj.y2,
                        )
                        if crop.size == 0:
                            continue
                        result = vlm.verify_crop(crop, vlm_query)
                        if result is not None:
                            fusion.apply_vlm_result(
                                obj.track_id, result, current_time=ts_seconds,
                            )
                    except Exception as e:
                        logger.warning(
                            f"[{task_id}] VLM verify failed for track {obj.track_id}: {e}"
                        )

            schema_detections = _build_detections(tracked_objects, fusion)

            annotated = draw_detections(
                frame=frame,
                tracked_objects=tracked_objects,
                timestamp=ts_str,
                show_confidence=True,
                fusion_engine=fusion,
                corner_style=settings.CORNER_STYLE_BOX,
                corner_length=settings.CORNER_LENGTH,
                box_thickness=settings.BOX_THICKNESS,
            )

            # Decide which frames reach disk.
            #   • best_shot: nothing inline — every detection is offered to the
            #     selector, which keeps each target's single best frame and the
            #     winners are written once at the end (_flush_best_shots).
            #   • otherwise: SAVE_FRAMES_MODE via the save_gate (in
            #     "unique_targets" mode it dedups by ByteTrack track_id).
            should_save_frame = False
            if best_shot_mode:
                for det in schema_detections:
                    if det.track_id is None:
                        continue
                    shot_score = best_shot_scorer.score(
                        frame, det.bbox.x1, det.bbox.y1,
                        det.bbox.x2, det.bbox.y2, det.score,
                    )
                    best_shot_selector.consider(
                        track_id=det.track_id,
                        score=shot_score,
                        frame_idx=frame_idx,
                        encode=lambda a=annotated: _encode_jpeg(a),
                    )
            else:
                should_save_frame = save_gate.should_save(
                    is_detection_frame=is_detection_frame,
                    detections=schema_detections,
                    ts_seconds=ts_seconds,
                )

            # Save frame to disk only if needed
            img_filename = None
            if should_save_frame:
                img_filename = _make_frame_filename(frame_idx, ts_str)
                img_path = task_results_dir / img_filename
                cv2.imwrite(
                    str(img_path),
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]
                )

            # Always generate base64 for SSE streaming (lower quality for bandwidth)
            img_b64 = frame_to_base64(annotated, quality=70)

            frame_result = FrameResult(
                frame_id=frame_idx,
                timestamp=ts_str,
                timestamp_seconds=round(ts_seconds, 3),
                detections=schema_detections,
                image_filename=img_filename,
                image_b64=img_b64,
            )

            task_manager.add_frame_result_sync(task_id, frame_result)
            if not loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    task_manager.push_frame(task_id, frame_result),
                    loop,
                )

            frame_idx += 1

    finally:
        cap.release()

    # Best-shot: write each target's single best snapshot now that all frames
    # have been scored. Skipped on cancel (that path discards the ZIP); a
    # manual terminate still flushes, packaging the partial best shots.
    if best_shot_mode and not task_manager.is_cancelled(task_id):
        written = _flush_best_shots(best_shot_selector, task_results_dir, fps)
        logger.info(
            f"[{task_id}] Best-shot: wrote {written} snapshot(s) "
            f"for {best_shot_selector.track_count} target(s)."
        )

    # Log statistics
    if quality_checker:
        quality_skip_ratio = skip_count / frame_idx if frame_idx > 0 else 0
        logger.info(
            f"[{task_id}] Processed {frame_idx} frames. "
            f"Quality skipped: {skip_count} ({quality_skip_ratio:.1%})"
        )
    if frame_analyzer:
        logger.info(
            f"[{task_id}] Adaptive keyframe detections: {detection_count}"
        )
    else:
        logger.info(f"[{task_id}] Processed {frame_idx} frames.")


# ────────────────────────────────────────────────────────────────────────────
# ZIP packaging
# ────────────────────────────────────────────────────────────────────────────

def _package_zip(
    task_id: str,
    task_results_dir: Path,
    task_state,
) -> None:
    """
    Create a ZIP archive containing:
      - All annotated frame JPEGs (based on SAVE_FRAMES_MODE)
      - results.json  (full detection metadata)
      - results.csv   (CSV summary)
      - summary.txt   (processing statistics)
    """
    zip_path = task_results_dir / "results.zip"

    # ── Build JSON / CSV data ─────────────────────────────────────────────
    records = []
    for fr in task_state.results:
        for det in fr.detections:
            records.append(
                {
                    "frame_id": fr.frame_id,
                    "timestamp": fr.timestamp,
                    "timestamp_seconds": fr.timestamp_seconds,
                    "track_id": det.track_id,
                    "detected_label": det.label,
                    "score": det.score,
                    "track_status": det.track_status,
                    "vlm_verified": det.vlm_verified,
                    "vlm_score": det.vlm_score,
                    "final_score": det.final_score,
                    "bbox_x1": det.bbox.x1,
                    "bbox_y1": det.bbox.y1,
                    "bbox_x2": det.bbox.x2,
                    "bbox_y2": det.bbox.y2,
                }
            )

    # Write JSON
    json_path = task_results_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Write CSV
    csv_path = task_results_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        if records:
            header = ",".join(records[0].keys())
            f.write(header + "\n")
            for row in records:
                f.write(",".join(str(v) for v in row.values()) + "\n")

    # ── Count saved frames ────────────────────────────────────────────────
    saved_frames = list(task_results_dir.glob("frame_*.jpg"))
    total_frames = len(task_state.results)

    # Write summary
    summary_path = task_results_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Total Frames Processed: {total_frames}\n")
        f.write(f"Frames Saved to ZIP: {len(saved_frames)}\n")
        f.write(f"Save Mode: {settings.SAVE_FRAMES_MODE}\n")
        f.write(f"Total Detections: {len(records)}\n")
        if total_frames > 0:
            save_ratio = len(saved_frames) / total_frames * 100
            f.write(f"Frame Save Ratio: {save_ratio:.1f}%\n")
        if task_state.early_terminated:
            f.write(f"\nEarly Termination: Yes\n")
            f.write(f"Termination Reason: {task_state.termination_reason}\n")

    # ── Pack ZIP ──────────────────────────────────────────────────────────
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Images (only those that were actually saved)
        for img_file in sorted(saved_frames):
            zf.write(img_file, img_file.name)
        # Metadata
        zf.write(json_path, "results.json")
        zf.write(csv_path, "results.csv")
        zf.write(summary_path, "summary.txt")

    logger.info(
        f"[{task_id}] ZIP created: {zip_path} | "
        f"Saved {len(saved_frames)}/{total_frames} frames ({settings.SAVE_FRAMES_MODE} mode)"
    )


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _make_frame_filename(frame_id: int, timestamp: str) -> str:
    """
    Build a filename like: frame_000125_00-00-05-000.jpg
    """
    ts_safe = timestamp.replace(":", "-").replace(".", "-")
    return f"frame_{frame_id:06d}_{ts_safe}.jpg"


def _encode_jpeg(frame: np.ndarray) -> bytes:
    """Encode a BGR frame as JPEG bytes at the configured save quality.

    Used by best-shot to retain a candidate snapshot in memory until the video
    ends (cheap: ~one image per live track), then flush it to disk verbatim.
    """
    ok, buf = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("Failed to JPEG-encode best-shot frame.")
    return buf.tobytes()


def _flush_best_shots(
    selector: BestShotSelector,
    task_results_dir: Path,
    fps: float,
) -> int:
    """Write every target's best snapshot to disk; return the file count.

    Filenames reuse the standard frame_<idx>_<ts>.jpg scheme so the snapshots
    are picked up by _package_zip's glob and served by /api/frame like any
    other saved frame.
    """
    written = 0
    for frame_idx, jpeg_bytes in selector.winners().items():
        ts_str = format_timestamp(frame_idx / fps) if fps > 0 else "00-00-00-000"
        img_path = task_results_dir / _make_frame_filename(frame_idx, ts_str)
        try:
            img_path.write_bytes(jpeg_bytes)
            written += 1
        except OSError as exc:
            logger.warning(f"Best-shot write failed for frame {frame_idx}: {exc}")
    return written


def _build_detections(
    tracked_objects: List[TrackedObject],
    fusion: FusionEngine,
) -> List[Detection]:
    results = []
    for obj in tracked_objects:
        ts = fusion.get_track(obj.track_id)
        track_status = ts.status if ts else "tentative"
        vlm_verified = ts.vlm_verified if ts else None
        vlm_score = round(ts.vlm_score_avg, 4) if ts and ts.vlm_scores else None
        final_score = round(ts.final_score, 4) if ts else None

        # Determine visibility: only show confirmed by default
        visible = track_status == "confirmed"

        results.append(
            Detection(
                track_id=obj.track_id,
                label=obj.label,
                score=round(obj.score, 4),
                bbox=BoundingBox(
                    x1=obj.x1, y1=obj.y1,
                    x2=obj.x2, y2=obj.y2,
                ),
                track_status=track_status,
                vlm_verified=vlm_verified,
                vlm_score=vlm_score,
                final_score=final_score,
                visible=visible,
            )
        )
    return results
