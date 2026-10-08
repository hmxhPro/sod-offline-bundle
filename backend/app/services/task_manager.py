"""
app/services/task_manager.py
-----------------------------
In-process task registry and async queue management.

Persists task lifecycle to PostgreSQL on every state transition. DB writes
are best-effort: outages log warnings but never propagate, so the detection
pipeline keeps working even if Postgres is down.

For production scaling, replace the asyncio.Queue with Celery + Redis.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any, Dict, Optional

from sqlalchemy.exc import SQLAlchemyError

from app.models.schemas import DetectRequest, FrameResult, TaskState, TaskStatus
from app.core.config import settings
from app.core.logging import logger


# Statuses a task can no longer leave — used for the in-memory LRU (R-5) and
# to decide when a task's frame results can be released.
_TERMINAL_STATUSES = frozenset({
    TaskStatus.FINISHED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.EARLY_TERMINATED,
})


class TaskManager:
    """
    Manages detection task lifecycle:
    - Creates tasks, tracks their state.
    - Routes completed frames to per-task async queues consumed by SSE streams.
    - Supports up to MAX_CONCURRENT_TASKS parallel GPU jobs via a semaphore.
    - Exposes cancel / pause / resume controls per task.
    """

    def __init__(self, max_concurrent: int = 2) -> None:
        self._tasks: Dict[str, TaskState] = {}
        # Per-task queues for streaming (task_id -> asyncio.Queue)
        self._queues: Dict[str, asyncio.Queue] = {}
        # Per-task control flags used by the synchronous pipeline loop.
        # cancel: set → loop exits after the current frame
        # pause:  clear → loop pauses; set again → loop resumes
        # terminate: set → loop exits and packages results
        self._cancel_flags: Dict[str, threading.Event] = {}
        self._pause_flags: Dict[str, threading.Event] = {}
        self._terminate_flags: Dict[str, threading.Event] = {}
        self._max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None
        # Async lock for async state mutations
        self._state_lock = asyncio.Lock()
        # Threading lock for sync state mutations from worker threads
        self._sync_state_lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────────────

    async def create_task(self, request: DetectRequest) -> TaskState:
        """Register a new task and return its initial state."""
        task_id = str(uuid.uuid4())
        state = TaskState(
            task_id=task_id,
            video_id=request.video_id,
            prompt=request.prompt,
            status=TaskStatus.PENDING,
        )
        self._tasks[task_id] = state
        self._queues[task_id] = asyncio.Queue(maxsize=settings.TASK_QUEUE_MAXSIZE)
        # Initialize control flags — "pause" starts "set" (i.e. not paused).
        self._cancel_flags[task_id] = threading.Event()
        self._terminate_flags[task_id] = threading.Event()
        pause = threading.Event()
        pause.set()
        self._pause_flags[task_id] = pause
        logger.info(f"Task created: {task_id} | prompt='{request.prompt}'")
        await self._persist_create(state, request.video_filename)
        return state

    def get_task(self, task_id: str) -> Optional[TaskState]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[TaskState]:
        return list(self._tasks.values())

    # ── Frame streaming ──────────────────────────────────────────────────────
    # The per-task queue is BOUNDED (settings.TASK_QUEUE_MAXSIZE). A client that
    # opens the SSE stream and walks away would otherwise let the queue grow with
    # full base64 frames until the single process OOMs (R-4). So:
    #   • frames are droppable — when the queue is full we drop the OLDEST frame
    #     to keep the stream real-time (the frame is also on disk + in results);
    #   • control/terminal sentinels (done/error/packaging/…) are NEVER dropped
    #     and NEVER block — a blocking put on a full queue would otherwise hang
    #     the pipeline coroutine forever and never release the GPU semaphore.

    def _put_sentinel(self, task_id: str, item: tuple) -> None:
        """Enqueue a non-frame event without blocking, guaranteeing it lands.

        If the queue is full, rebuild it keeping existing sentinels in FIFO
        order and dropping frames to make room, then append `item`. Sentinels
        are few and tiny; frames are the bulk and safely droppable.
        """
        q = self._queues.get(task_id)
        if q is None:
            return
        try:
            q.put_nowait(item)
            return
        except asyncio.QueueFull:
            pass
        kept: list[tuple] = []
        while True:
            try:
                ev = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            if ev[0] != "frame":      # preserve every non-frame event
                kept.append(ev)
        kept.append(item)
        for ev in kept:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                break                 # unreachable in practice: kept << maxsize

    async def push_frame(self, task_id: str, frame: FrameResult) -> None:
        """Called by the worker after each frame is processed.

        Drops the oldest queued frame when full (bounded, drop-oldest) so a
        disconnected consumer cannot drive the queue to OOM.
        """
        q = self._queues.get(task_id)
        if q is None:
            return
        if q.full():
            try:
                q.get_nowait()       # drop oldest to preserve real-time recency
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(("frame", frame))
        except asyncio.QueueFull:
            pass

    async def push_packaging(self, task_id: str) -> None:
        """Signal that frame processing finished and ZIP packaging started."""
        self._put_sentinel(task_id, ("packaging", None))

    async def push_paused(self, task_id: str) -> None:
        self._put_sentinel(task_id, ("paused", None))

    async def push_resumed(self, task_id: str) -> None:
        self._put_sentinel(task_id, ("resumed", None))

    async def push_cancelled(self, task_id: str) -> None:
        self._put_sentinel(task_id, ("cancelled", None))

    async def push_early_terminated(self, task_id: str, reason: str) -> None:
        """Signal that the task was terminated early."""
        self._put_sentinel(task_id, ("early_terminated", reason))

    async def push_done(self, task_id: str) -> None:
        """Signal that the task is fully complete.

        Pops the queue from the registry after enqueueing the terminal
        event. The SSE consumer's *local* reference keeps the queue
        object alive long enough to drain `done`; a freshly-reconnecting
        client sees no queue and falls back to GET /api/task for state.
        This lets a page refresh during a long-running task pick up the
        live frame stream again instead of being permanently silenced.
        """
        self._put_sentinel(task_id, ("done", None))
        self._queues.pop(task_id, None)

    async def push_error(self, task_id: str, error: str) -> None:
        """Signal a processing error. Pops the queue (see push_done)."""
        self._put_sentinel(task_id, ("error", error))
        self._queues.pop(task_id, None)

    # ── Control (cancel / pause / resume) ───────────────────────────────────

    def request_cancel(self, task_id: str) -> bool:
        """Signal the pipeline to stop after the current frame."""
        flag = self._cancel_flags.get(task_id)
        if flag is None:
            return False
        flag.set()
        # Ensure a paused worker wakes up so it can see the cancel flag.
        pause = self._pause_flags.get(task_id)
        if pause is not None:
            pause.set()
        logger.info(f"Task {task_id} cancel requested.")
        return True

    def request_pause(self, task_id: str) -> bool:
        """Block the pipeline loop until request_resume is called."""
        flag = self._pause_flags.get(task_id)
        if flag is None:
            return False
        flag.clear()
        logger.info(f"Task {task_id} pause requested.")
        return True

    def request_resume(self, task_id: str) -> bool:
        flag = self._pause_flags.get(task_id)
        if flag is None:
            return False
        flag.set()
        logger.info(f"Task {task_id} resume requested.")
        return True

    def request_terminate(self, task_id: str) -> bool:
        """Signal the pipeline to stop and package results."""
        flag = self._terminate_flags.get(task_id)
        if flag is None:
            return False
        flag.set()
        # Ensure a paused worker wakes up so it can see the terminate flag.
        pause = self._pause_flags.get(task_id)
        if pause is not None:
            pause.set()
        logger.info(f"Task {task_id} terminate requested.")
        return True

    def is_cancelled(self, task_id: str) -> bool:
        flag = self._cancel_flags.get(task_id)
        return bool(flag and flag.is_set())

    def is_terminated(self, task_id: str) -> bool:
        flag = self._terminate_flags.get(task_id)
        return bool(flag and flag.is_set())

    def wait_if_paused(self, task_id: str, poll_interval: float = 0.1) -> None:
        """
        Block (in the pipeline worker thread) while the task is paused.
        Returns immediately if the task is cancelled or terminated.
        """
        pause = self._pause_flags.get(task_id)
        cancel = self._cancel_flags.get(task_id)
        terminate = self._terminate_flags.get(task_id)
        if pause is None:
            return
        # If the event is set, we are NOT paused — fall through.
        while not pause.wait(timeout=poll_interval):
            if cancel is not None and cancel.is_set():
                return
            if terminate is not None and terminate.is_set():
                return

    async def consume_stream(self, task_id: str):
        """
        Async generator that yields (event_type, payload) tuples.
        Used by the SSE endpoint.
        """
        queue = self._queues.get(task_id)
        if queue is None:
            return
        while True:
            event_type, payload = await queue.get()
            yield event_type, payload
            if event_type in ("done", "error"):
                break

    # ── State helpers ────────────────────────────────────────────────────────

    async def set_running(self, task_id: str, total_frames: int) -> None:
        async with self._state_lock:
            state = self._tasks[task_id]
            state.status = TaskStatus.RUNNING
            state.total_frames = total_frames
            logger.info(f"Task {task_id} started | total_frames={total_frames}")
        await self._persist_update(
            task_id, status=TaskStatus.RUNNING.value, total_frames=total_frames
        )

    async def add_frame_result(self, task_id: str, frame: FrameResult) -> None:
        async with self._state_lock:
            state = self._tasks[task_id]
            state.results.append(frame)
            state.processed_frames += 1
            state.progress = state.processed_frames / max(state.total_frames, 1)
            if frame.detections:
                state.detection_frame_count += 1

    def add_frame_result_sync(self, task_id: str, frame: FrameResult) -> None:
        """Called directly from the worker thread — no asyncio overhead.

        Strips image_b64 before storing to prevent unbounded memory growth.
        """
        with self._sync_state_lock:
            state = self._tasks.get(task_id)
            if state is None:
                return
            stored = frame.model_copy(update={"image_b64": ""}) if frame.image_b64 else frame
            state.results.append(stored)
            state.processed_frames += 1
            state.progress = state.processed_frames / max(state.total_frames, 1)
            if frame.detections:
                state.detection_frame_count += 1

    async def set_finished(self, task_id: str) -> None:
        async with self._state_lock:
            state = self._tasks[task_id]
            state.status = TaskStatus.FINISHED
            state.progress = 1.0
            state.zip_ready = True
            logger.info(f"Task {task_id} finished | frames={state.processed_frames}")
        await self._persist_update(
            task_id,
            status=TaskStatus.FINISHED.value,
            progress=1.0,
            processed_frames=state.processed_frames,
            zip_ready=True,
        )
        self._evict_terminal_overflow()

    async def set_paused(self, task_id: str) -> bool:
        async with self._state_lock:
            state = self._tasks.get(task_id)
            if state is not None and state.status == TaskStatus.RUNNING:
                state.status = TaskStatus.PAUSED
                logger.info(f"Task {task_id} paused | frames={state.processed_frames}")
                paused_now = True
            else:
                paused_now = False
        if paused_now:
            await self._persist_update(task_id, status=TaskStatus.PAUSED.value)
        return paused_now

    async def set_resumed(self, task_id: str) -> bool:
        async with self._state_lock:
            state = self._tasks.get(task_id)
            if state is not None and state.status == TaskStatus.PAUSED:
                state.status = TaskStatus.RUNNING
                logger.info(f"Task {task_id} resumed | frames={state.processed_frames}")
                resumed_now = True
            else:
                resumed_now = False
        if resumed_now:
            await self._persist_update(task_id, status=TaskStatus.RUNNING.value)
        return resumed_now

    async def set_cancelled(self, task_id: str) -> None:
        async with self._state_lock:
            state = self._tasks.get(task_id)
            if state is None:
                return
            state.status = TaskStatus.CANCELLED
            logger.info(f"Task {task_id} cancelled | frames={state.processed_frames}")
        await self._persist_update(
            task_id,
            status=TaskStatus.CANCELLED.value,
            processed_frames=state.processed_frames,
            progress=state.progress,
        )
        self._evict_terminal_overflow()

    async def set_early_terminated(self, task_id: str, reason: str) -> None:
        async with self._state_lock:
            state = self._tasks.get(task_id)
            if state is None:
                return
            state.status = TaskStatus.EARLY_TERMINATED
            state.early_terminated = True
            state.termination_reason = reason
            state.zip_ready = True
            logger.info(f"Task {task_id} early terminated: {reason} | frames={state.processed_frames}")
        await self._persist_update(
            task_id,
            status=TaskStatus.EARLY_TERMINATED.value,
            early_terminated=True,
            termination_reason=reason,
            zip_ready=True,
            processed_frames=state.processed_frames,
            progress=state.progress,
        )
        self._evict_terminal_overflow()

    async def set_failed(self, task_id: str, error: str) -> None:
        async with self._state_lock:
            state = self._tasks[task_id]
            state.status = TaskStatus.FAILED
            state.error = error
            logger.error(f"Task {task_id} failed: {error}")
        await self._persist_update(
            task_id, status=TaskStatus.FAILED.value, error=error
        )
        self._evict_terminal_overflow()

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    # ── DB persistence (best-effort) ────────────────────────────────────────
    # All persistence helpers swallow exceptions and only log a warning so
    # a Postgres outage never crashes detection. The DB is a write-side
    # archive; the source of truth for in-flight tasks is still memory.

    async def _persist_create(
        self, state: TaskState, video_filename: Optional[str]
    ) -> None:
        try:
            from app.db.session import AsyncSessionLocal
            from app.db.models import TaskRecord

            async with AsyncSessionLocal() as session:
                session.add(
                    TaskRecord(
                        task_id=state.task_id,
                        video_id=state.video_id,
                        video_filename=video_filename,
                        prompt=state.prompt,
                        status=state.status.value,
                    )
                )
                await session.commit()
        except (SQLAlchemyError, OSError) as exc:
            # Expected when the DB is down / flaky — best-effort archive, swallow.
            logger.warning(f"DB persist (create) failed for {state.task_id}: {exc}")
        except Exception as exc:
            # Unexpected (likely a code/schema bug) — surface loudly (R-7) so it
            # isn't mistaken for a routine DB outage, but still don't crash.
            logger.error(f"DB persist (create) UNEXPECTED error for {state.task_id}: {exc}")

    async def _persist_update(self, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        try:
            from sqlalchemy import update, func as sql_func
            from app.db.session import AsyncSessionLocal
            from app.db.models import TaskRecord

            # Stamp finished_at automatically on terminal status transitions.
            terminal = {"finished", "failed", "cancelled", "early_terminated"}
            if fields.get("status") in terminal and "finished_at" not in fields:
                fields["finished_at"] = sql_func.now()

            async with AsyncSessionLocal() as session:
                await session.execute(
                    update(TaskRecord)
                    .where(TaskRecord.task_id == task_id)
                    .values(**fields)
                )
                await session.commit()
        except (SQLAlchemyError, OSError) as exc:
            logger.warning(f"DB persist (update) failed for {task_id}: {exc}")
        except Exception as exc:
            logger.error(f"DB persist (update) UNEXPECTED error for {task_id}: {exc}")

    def cleanup_queue(self, task_id: str) -> None:
        """Drop the stream queue for a task.

        The normal lifecycle now pops the queue from inside push_done /
        push_error (so a client disconnect no longer destroys it and a
        reconnecting browser tab picks up the in-flight stream). This
        helper remains for explicit teardown paths (tests, admin tools).
        """
        self._queues.pop(task_id, None)

    def cleanup_flags(self, task_id: str) -> None:
        """Remove control flags once a task reaches a terminal state."""
        self._cancel_flags.pop(task_id, None)
        self._pause_flags.pop(task_id, None)
        self._terminate_flags.pop(task_id, None)

    def remove_task(self, task_id: str) -> None:
        """Drop *all* in-memory state for a task. Idempotent.

        Used by the history-delete endpoint after the DB row and on-disk
        artifacts have been removed, so a task is fully forgotten.
        """
        self._tasks.pop(task_id, None)
        self._queues.pop(task_id, None)
        self._cancel_flags.pop(task_id, None)
        self._pause_flags.pop(task_id, None)
        self._terminate_flags.pop(task_id, None)

    def release_results(self, task_id: str) -> None:
        """Drop the in-memory per-frame results for a terminal task (R-5).

        The full results.json is already on disk (pipeline._package_zip) and the
        lightweight status lives in the DB, so the frame list is dead weight once
        a task is terminal — and on long videos it grows unbounded. Call this
        only AFTER packaging has read state.results. Idempotent.
        """
        with self._sync_state_lock:
            state = self._tasks.get(task_id)
            if state is not None and state.results:
                state.results = []

    def _evict_terminal_overflow(self) -> None:
        """Evict the oldest terminal tasks beyond MAX_RETAINED_TASKS (R-5).

        Active tasks are never evicted. Evicted tasks stay downloadable and
        queryable via the DB archive + on-disk ZIP (see download_results /
        get_task fallbacks), so this only bounds memory, not durability.
        Insertion order is preserved by dict ordering (oldest first).
        """
        cap = settings.MAX_RETAINED_TASKS
        with self._sync_state_lock:
            terminal_ids = [
                tid for tid, st in self._tasks.items()
                if st.status in _TERMINAL_STATUSES
            ]
            overflow = len(terminal_ids) - cap
            to_evict = terminal_ids[:overflow] if overflow > 0 else []
        # remove_task pops without the sync lock — do it outside to avoid
        # re-entrant locking (threading.Lock is non-reentrant).
        for tid in to_evict:
            self.remove_task(tid)
        if to_evict:
            logger.debug(f"LRU evicted {len(to_evict)} terminal task(s) from memory")


# Global singleton (replaced in tests via dependency injection)
task_manager = TaskManager()
