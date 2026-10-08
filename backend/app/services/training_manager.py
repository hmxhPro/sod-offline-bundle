"""
app/services/training_manager.py
--------------------------------
Runs YOLOE training as an isolated subprocess and tracks it in PostgreSQL.

Why subprocess (not a thread): the backend runs a single uvicorn worker that
also serves the live video-detection SSE pipeline. `model.train()` is a
multi-minute blocking GPU job; isolating it in a child process keeps the event
loop responsive, makes the job killable, and prevents a training crash/OOM
from taking down detection.

Progress: a monitor coroutine tails `runs/train/<name>/results.csv` (one row
per finished epoch) and writes epoch/progress/mAP into the training_jobs row.
On exit, the child's `__YOLOE_TRAIN_JSON__ {...}` stdout line is the
authoritative completion signal (best_pt + metrics); results.csv is the
fallback.

Concurrency: a single global training slot (one job at a time) — starting a
job while one is active raises TrainingBusyError (→ HTTP 409).
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import func as sql_func, select, update
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.core.logging import logger
from app.utils.naming import strip_timestamp_suffix

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent  # .../backend
_TRAIN_SCRIPT = _BACKEND_ROOT / "YOLOWorld" / "train_yoloe.py"
_TRAIN_JSON_PREFIX = "__YOLOE_TRAIN_JSON__"

# ── Cross-platform process-group control ────────────────────────────────────
# The training child spawns its own dataloader worker children; we must be able
# to reap the WHOLE tree on cancel/shutdown/crash so no worker is orphaned
# holding the GPU (R-1/R-2/R-9).
#   POSIX  : start_new_session=True makes proc.pid a process-group leader, then
#            os.killpg(pid, SIG*) signals the whole group.
#   Windows: os.killpg / SIGTERM don't exist. We create the child in its own
#            process group (CREATE_NEW_PROCESS_GROUP) so a graceful CTRL_BREAK
#            can be delivered, and fall back to `taskkill /T` which force-kills
#            the entire child tree.
_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:
    # A process spawned with CREATE_NEW_PROCESS_GROUP is the root of a new group
    # and can receive CTRL_BREAK_EVENT (CTRL_C_EVENT cannot be targeted at a
    # group other than the caller's, so CTRL_BREAK is the one to use).
    _SPAWN_KWARGS: dict[str, Any] = {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    }
else:
    # POSIX: give the child its own session/process group (pgid == proc.pid).
    _SPAWN_KWARGS = {"start_new_session": True}


class TrainingBusyError(RuntimeError):
    """Raised when a training job is requested while another is running."""


def _slug(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z一-鿿_-]+", "_", name).strip("_")
    return s or "model"


def _find_map_columns(fieldnames: list[str]) -> tuple[Optional[str], Optional[str]]:
    """Locate the mAP50 and mAP50-95 column names in a results.csv header
    (names vary across ultralytics versions; tolerate spaces)."""
    m50 = m5095 = None
    for raw in fieldnames:
        k = raw.strip()
        low = k.lower()
        if "map50-95" in low or "map_0.5:0.95" in low:
            m5095 = raw
        elif "map50" in low or ("map_0.5" in low and "0.95" not in low):
            m50 = raw
    return m50, m5095


class TrainingManager:
    def __init__(self) -> None:
        self._active_job_id: Optional[str] = None
        self._lock = asyncio.Lock()
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._tasks: set[asyncio.Task] = set()
        # Jobs the user explicitly cancelled — _monitor checks this before
        # overwriting status with "failed" when the killed process exits non-zero
        # (otherwise a cancel→SIGTERM→rc!=0 race relabels "cancelled" as "failed").
        self._cancelled: set[str] = set()

    @property
    def active_job_id(self) -> Optional[str]:
        return self._active_job_id

    async def start(self, category: dict, images: list[dict], params: dict) -> str:
        """Finalize the dataset, insert a job row, and spawn training.

        `category` = {"id", "name"}; `images` = annotated [{"id","stored_path"}];
        `params` may carry epochs/imgsz/batch/base_model. Returns job_id.
        """
        # ── claim the single training slot ───────────────────────────────
        async with self._lock:
            if self._active_job_id is not None:
                raise TrainingBusyError(
                    f"已有训练任务进行中（{self._active_job_id}），请等待其完成。"
                )
            job_id = str(uuid.uuid4())
            self._active_job_id = job_id

        try:
            base_model = (params.get("base_model") or settings.train_base_model or "").strip()
            if not base_model:
                raise RuntimeError(
                    "未配置训练基础权重（.env 的 TRAIN_BASE_MODEL）。"
                )
            if not Path(base_model).exists():
                raise FileNotFoundError(f"基础权重文件不存在: {base_model}")

            epochs = int(params.get("epochs") or settings.TRAIN_DEFAULT_EPOCHS)
            imgsz = int(params.get("imgsz") or settings.TRAIN_DEFAULT_IMGSZ)
            batch = int(params.get("batch") or settings.TRAIN_DEFAULT_BATCH)
            # Explicit None check (not `or`): workers=0 is a valid choice
            # (load in main process) and must not fall through to the default.
            _workers = params.get("workers")
            workers = settings.TRAIN_DEFAULT_WORKERS if _workers is None else int(_workers)
            # Small-object augmentation knobs. Explicit None check: 0.0 / 0 are
            # valid (= disable mosaic / no scale jitter) and must not fall back.
            _mosaic = params.get("mosaic")
            mosaic = settings.TRAIN_MOSAIC if _mosaic is None else float(_mosaic)
            _scale = params.get("scale")
            scale = settings.TRAIN_SCALE if _scale is None else float(_scale)
            _close = params.get("close_mosaic")
            close_mosaic = settings.TRAIN_CLOSE_MOSAIC if _close is None else int(_close)

            # next version for this category (single-training lock → no race)
            version = await self._next_version(category["id"])
            run_name = f"{_slug(category['name'])}_v{version}_{job_id[:8]}"

            # ── build the frozen dataset (off the event loop) ────────────
            from app.services import dataset_builder

            ds = await asyncio.to_thread(
                dataset_builder.finalize,
                category["id"], category["name"], job_id, images,
            )
            ds["base_model"] = base_model

            await self._persist_create(
                job_id=job_id,
                category_id=category["id"],
                model_name=category["name"],
                total_epochs=epochs,
                dataset_yaml=ds["dataset_yaml"],
                base_model=base_model,
                params={
                    "epochs": epochs, "imgsz": imgsz, "batch": batch,
                    "workers": workers,
                    "mosaic": mosaic, "scale": scale, "close_mosaic": close_mosaic,
                    "base_model": base_model, "version": version,
                    "num_images": ds["num_images"], "val_is_train": ds["val_is_train"],
                },
            )

            project = str(settings.TRAIN_RUNS_DIR.resolve())
            cmd = [
                sys.executable, str(_TRAIN_SCRIPT),
                "--data", ds["dataset_yaml"],
                "--model", base_model,
                "--epochs", str(epochs),
                "--imgsz", str(imgsz),
                "--batch", str(batch),
                "--workers", str(workers),
                "--mosaic", str(mosaic),
                "--scale", str(scale),
                "--close-mosaic", str(close_mosaic),
                "--device", settings.DEVICE,
                "--project", project,
                "--name", run_name,
                "--json",
            ]
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("YOLO_OFFLINE", "1")
            env.setdefault("ULTRALYTICS_OFFLINE", "1")
            # Don't stream tqdm '\r' progress bars into the captured pipe — they
            # are huge and newline-free. Progress is read from results.csv (see
            # _poll_progress); completion comes from the JSON sentinel below.
            env.setdefault("TQDM_DISABLE", "1")

            logger.info(f"Spawning training job {job_id}: {' '.join(cmd)}")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(_BACKEND_ROOT),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                # Own session/process group so the whole training tree (the
                # script + its dataloader worker children) can be reaped as a
                # group on cancel/shutdown/crash instead of being orphaned and
                # left holding the GPU (R-1/R-2/R-9). On POSIX pgid == proc.pid;
                # on Windows the child roots a new CREATE_NEW_PROCESS_GROUP tree.
                # See _SPAWN_KWARGS for the per-platform flag.
                **_SPAWN_KWARGS,
            )
            self._procs[job_id] = proc

            run_dir = settings.TRAIN_RUNS_DIR.resolve() / run_name
            await self._persist_update(
                job_id, status="running", pid=proc.pid, started_at=sql_func.now()
            )

            task = asyncio.create_task(
                self._monitor(job_id, proc, run_dir, epochs, category, version, ds)
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return job_id

        except Exception as exc:
            # Failed before/at spawn — release the slot and mark the job failed.
            self._active_job_id = None
            self._procs.pop(job_id, None)
            await self._persist_update(job_id, status="failed", error=str(exc))
            logger.error(f"Training job {job_id} failed to start: {exc}")
            raise

    async def _monitor(
        self, job_id: str, proc: asyncio.subprocess.Process,
        run_dir: Path, total_epochs: int, category: dict, version: int, ds: dict,
    ) -> None:
        stdout_buf: list[str] = []

        async def _drain() -> None:
            # Read fixed-size chunks, NOT lines. ultralytics/tqdm redraw the
            # epoch progress bar in place with '\r' and emit no '\n' until the
            # bar closes, so one "line" exceeds asyncio's 64 KiB StreamReader
            # limit and `async for`/readline() raises LimitOverrunError, killing
            # this task. Once the reader dies the child's stdout socket backs up
            # and the training process blocks forever in write(). read(n) never
            # raises on long lines.
            assert proc.stdout is not None
            while True:
                chunk = await proc.stdout.read(65536)
                if not chunk:
                    break
                stdout_buf.append(chunk.decode(errors="replace"))
                if len(stdout_buf) > 600:
                    del stdout_buf[0]

        drain_task = asyncio.create_task(_drain())
        wait_task = asyncio.ensure_future(proc.wait())
        try:
            while not wait_task.done():
                done, _ = await asyncio.wait({wait_task}, timeout=3.0)
                if wait_task in done:
                    break
                await self._poll_progress(job_id, run_dir, total_epochs)
            await wait_task
            try:
                await asyncio.wait_for(drain_task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                drain_task.cancel()

            rc = proc.returncode
            if rc == 0:
                info = self._parse_sentinel(stdout_buf)
                best_pt = (info or {}).get("best_pt") or str(run_dir / "weights" / "best.pt")
                metrics = (info or {}).get("metrics") or self._read_last_metrics(run_dir)
                m50, m5095 = self._pick_map(metrics)
                if not Path(best_pt).exists():
                    raise FileNotFoundError(f"训练结束但未找到 best.pt: {best_pt}")

                # M-2: a run with no real mAP (None / <= floor, e.g. an all-zero
                # broken run) is marked needs_review and NOT registered as a
                # selectable model — keep undeployable models out of production.
                deployable = self._is_deployable(m50)
                await self._persist_update(
                    job_id, status="finished" if deployable else "needs_review",
                    progress=1.0, current_epoch=total_epochs, metrics=metrics,
                    metric_map50=m50, metric_map50_95=m5095,
                    best_pt_path=best_pt, finished_at=sql_func.now(),
                )
                if deployable:
                    await self._insert_trained_model(
                        job_id=job_id, category=category, version=version,
                        best_pt=best_pt, base_model=ds.get("base_model"),
                        dataset_yaml=ds["dataset_yaml"], num_images=ds["num_images"],
                        metrics=metrics, val_is_train=bool(ds.get("val_is_train")),
                    )
                    await self._set_category_status(category["id"], "trained")
                    logger.info(f"Training job {job_id} finished: best={best_pt} mAP50={m50}")
                else:
                    logger.warning(
                        f"Training job {job_id} mAP50={m50} <= deploy floor "
                        f"{settings.MIN_DEPLOYABLE_MAP50}; marked needs_review, not registered."
                    )
            elif job_id in self._cancelled:
                # Cancel already persisted status='cancelled'; the non-zero rc is
                # just the SIGTERM/SIGKILL we sent — don't relabel it 'failed'.
                logger.info(f"Training job {job_id} cancelled (rc={rc}).")
            else:
                tail = "".join(stdout_buf)[-4000:]
                await self._persist_update(
                    job_id, status="failed",
                    error=f"训练进程退出码 {rc}。\n--- 日志尾部 ---\n{tail}",
                    finished_at=sql_func.now(),
                )
                logger.error(f"Training job {job_id} failed (rc={rc})")
        except Exception as exc:
            await self._persist_update(
                job_id, status="failed", error=str(exc), finished_at=sql_func.now()
            )
            logger.error(f"Training job {job_id} monitor error: {exc}")
        finally:
            self._procs.pop(job_id, None)
            self._active_job_id = None
            self._cancelled.discard(job_id)

    async def cancel(self, job_id: str) -> bool:
        """Terminate a running training subprocess and its worker children.

        Escalates SIGTERM → (grace) → SIGKILL across the whole process group so
        ultralytics' dataloader children are not orphaned holding the GPU (R-9).
        """
        proc = self._procs.get(job_id)
        if proc is None or proc.returncode is not None:
            return False
        self._cancelled.add(job_id)
        await self._kill_group(proc, grace=10.0)
        await self._persist_update(
            job_id, status="cancelled", error="用户取消", finished_at=sql_func.now()
        )
        return True

    @staticmethod
    async def _kill_group(proc: asyncio.subprocess.Process, grace: float) -> None:
        """Gracefully stop the child's whole process tree, then force-kill after
        `grace` seconds. Cross-platform:
          POSIX  : SIGTERM the process group, wait, then SIGKILL the group.
          Windows: CTRL_BREAK_EVENT to the new process group, wait, then
                   `taskkill /F /T` to force-kill the child + its worker tree.
        No-op / swallows "already gone" errors. Relies on the per-platform
        _SPAWN_KWARGS so proc.pid is the group/tree root."""
        if proc.returncode is not None:
            return
        # ── phase 1: ask nicely ─────────────────────────────────────────────
        TrainingManager._signal_tree(proc, force=False)
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace)
            return
        except asyncio.TimeoutError:
            pass
        # ── phase 2: force kill the whole tree ──────────────────────────────
        TrainingManager._signal_tree(proc, force=True)

    @staticmethod
    def _signal_tree(proc: asyncio.subprocess.Process, force: bool) -> None:
        """Send a stop (force=False) or kill (force=True) signal to the child's
        entire process tree. Swallows races where the process already exited."""
        if proc.returncode is not None:
            return
        pid = proc.pid
        try:
            if _IS_WINDOWS:
                if force:
                    # /T = tree (dataloader workers), /F = force. Best-effort.
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True,
                    )
                else:
                    # Graceful: CTRL_BREAK to the child's own process group.
                    try:
                        proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                    except (ValueError, OSError):
                        # Fall back to a plain terminate if the group signal is
                        # unavailable for any reason.
                        proc.terminate()
            else:
                os.killpg(pid, signal.SIGKILL if force else signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            # Already gone or not signalable — nothing to do.
            pass

    async def terminate_all(self, grace: float = 10.0) -> None:
        """Kill every live training subprocess group. Called from the app
        shutdown hook so a deploy/restart never orphans a training run (R-2)."""
        live = [(jid, p) for jid, p in list(self._procs.items()) if p.returncode is None]
        if not live:
            return
        logger.info(f"terminate_all: killing {len(live)} training subprocess group(s)")
        # Mark these as intentionally stopped so _monitor's rc!=0 handler routes
        # them to the 'cancelled' branch instead of relabeling them 'failed'
        # (mirrors cancel()'s race-guard for the shutdown path).
        for jid, _ in live:
            self._cancelled.add(jid)
        for _, proc in live:                     # phase 1: signal them all
            self._signal_tree(proc, force=False)
        for _, proc in live:                     # phase 2: wait, then force-kill
            try:
                await asyncio.wait_for(proc.wait(), timeout=grace)
            except asyncio.TimeoutError:
                self._signal_tree(proc, force=True)

    @staticmethod
    def _is_deployable(m50: Optional[float]) -> bool:
        """A run is deployable only if it produced a real mAP50 above the floor.
        None / <= floor (e.g. all-zero broken runs) stay out of the selectable
        model list (M-2)."""
        return m50 is not None and m50 > settings.MIN_DEPLOYABLE_MAP50

    # ── progress / metrics parsing ──────────────────────────────────────

    async def _poll_progress(self, job_id: str, run_dir: Path, total_epochs: int) -> None:
        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            return
        try:
            rows = self._read_csv_rows(csv_path)
            if not rows:
                return
            # Read the actual epoch number from the last row's "epoch" column instead
            # of counting rows. This is more robust if the CSV is partially flushed
            # or if there are any malformed rows.
            last_row = rows[-1]
            epoch_str = last_row.get("epoch", "").strip()
            try:
                current_epoch = int(float(epoch_str)) if epoch_str else len(rows)
            except (ValueError, TypeError):
                # Fallback to row count if epoch column is missing or malformed
                current_epoch = len(rows)

            m50, m5095 = self._pick_map(self._row_to_metrics(last_row))
            progress = min(current_epoch / max(total_epochs, 1), 0.99)
            await self._persist_update(
                job_id, current_epoch=current_epoch, progress=progress,
                metric_map50=m50, metric_map50_95=m5095,
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug(f"poll_progress({job_id}) skipped: {exc}")

    @staticmethod
    def _read_csv_rows(csv_path: Path) -> list[dict]:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            return [{(k or "").strip(): v for k, v in row.items()} for row in reader]

    def _read_last_metrics(self, run_dir: Path) -> dict:
        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            return {}
        try:
            rows = self._read_csv_rows(csv_path)
            return self._row_to_metrics(rows[-1]) if rows else {}
        except Exception:
            return {}

    @staticmethod
    def _row_to_metrics(row: dict) -> dict:
        out: dict = {}
        for k, v in row.items():
            try:
                out[k.strip()] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _pick_map(metrics: dict) -> tuple[Optional[float], Optional[float]]:
        if not metrics:
            return None, None
        m50, m5095 = TrainingManager._find_map_keys(list(metrics.keys()))
        return (metrics.get(m50) if m50 else None,
                metrics.get(m5095) if m5095 else None)

    @staticmethod
    def _find_map_keys(keys: list[str]) -> tuple[Optional[str], Optional[str]]:
        return _find_map_columns(keys)

    @staticmethod
    def _parse_sentinel(buf: list[str]) -> Optional[dict]:
        # buf now holds arbitrary chunks (not whole lines), so the sentinel may
        # span a chunk boundary — search the joined text, not each element.
        text = "".join(buf)
        idx = text.rfind(_TRAIN_JSON_PREFIX)
        if idx == -1:
            return None
        rest = text[idx + len(_TRAIN_JSON_PREFIX):].split("\n", 1)[0].strip()
        try:
            return json.loads(rest)
        except json.JSONDecodeError:
            return None

    # ── DB persistence (best-effort, mirrors task_manager) ──────────────

    async def _next_version(self, category_id: str) -> int:
        try:
            from app.db.session import AsyncSessionLocal
            from app.db.models import TrainedModelRecord
            async with AsyncSessionLocal() as session:
                n = await session.scalar(
                    select(sql_func.count(TrainedModelRecord.id)).where(
                        TrainedModelRecord.category_id == category_id
                    )
                )
            return int(n or 0) + 1
        except Exception as exc:
            logger.warning(f"next_version failed, defaulting to 1: {exc}")
            return 1

    async def _persist_create(self, job_id: str, **fields: Any) -> None:
        try:
            from app.db.session import AsyncSessionLocal
            from app.db.models import TrainingJobRecord
            async with AsyncSessionLocal() as session:
                session.add(TrainingJobRecord(id=job_id, status="pending", **fields))
                await session.commit()
        except (SQLAlchemyError, OSError) as exc:
            logger.warning(f"DB persist (job create) failed for {job_id}: {exc}")
        except Exception as exc:
            logger.error(f"DB persist (job create) UNEXPECTED error for {job_id}: {exc}")

    async def _persist_update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        try:
            from app.db.session import AsyncSessionLocal
            from app.db.models import TrainingJobRecord
            async with AsyncSessionLocal() as session:
                await session.execute(
                    update(TrainingJobRecord)
                    .where(TrainingJobRecord.id == job_id)
                    .values(**fields)
                )
                await session.commit()
        except (SQLAlchemyError, OSError) as exc:
            logger.warning(f"DB persist (job update) failed for {job_id}: {exc}")
        except Exception as exc:
            logger.error(f"DB persist (job update) UNEXPECTED error for {job_id}: {exc}")

    async def _insert_trained_model(
        self, job_id: str, category: dict, version: int, best_pt: str,
        base_model: Optional[str], dataset_yaml: str, num_images: int, metrics: dict,
        val_is_train: bool = False,
    ) -> None:
        try:
            from app.db.session import AsyncSessionLocal
            from app.db.models import TrainingJobRecord, TrainedModelRecord
            # Embed val_is_train into a COPY of metrics (M-1) so the trained-model
            # row carries the "metrics measured on the training set" flag without
            # a schema migration — and without mutating the job's own metrics dict
            # (already persisted). _find_map_columns / the UI only read mAP keys,
            # so the extra boolean key is ignored by metric parsing.
            model_metrics = {**(metrics or {}), "val_is_train": bool(val_is_train)}
            async with AsyncSessionLocal() as session:
                job = await session.get(TrainingJobRecord, job_id)
                session.add(TrainedModelRecord(
                    id=str(uuid.uuid4()),
                    name=category["name"],
                    version=version,
                    category_id=category["id"],
                    training_job_id=job_id,
                    weights_path=best_pt,
                    base_model=base_model,
                    class_names={"0": strip_timestamp_suffix(category["name"])},  # Phase 1 single-class
                    dataset_yaml=dataset_yaml,
                    num_images=num_images,
                    metrics=model_metrics,
                    trained_started_at=getattr(job, "started_at", None),
                    trained_finished_at=datetime.now(timezone.utc),
                ))
                await session.commit()
        except (SQLAlchemyError, OSError) as exc:
            logger.warning(f"DB insert trained_model failed for {job_id}: {exc}")
        except Exception as exc:
            logger.error(f"DB insert trained_model UNEXPECTED error for {job_id}: {exc}")

    async def _set_category_status(self, category_id: str, status: str) -> None:
        try:
            from app.db.session import AsyncSessionLocal
            from app.db.models import CategoryRecord
            async with AsyncSessionLocal() as session:
                await session.execute(
                    update(CategoryRecord)
                    .where(CategoryRecord.id == category_id)
                    .values(status=status)
                )
                await session.commit()
        except Exception as exc:
            logger.warning(f"set_category_status failed for {category_id}: {exc}")


# Global singleton
training_manager = TrainingManager()
