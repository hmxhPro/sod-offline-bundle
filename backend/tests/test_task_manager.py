"""
tests/test_task_manager.py
--------------------------
Covers the R-4 (bounded SSE queue) and R-5 (results release + LRU) fixes —
the in-memory memory-safety guarantees, exercised without a DB or GPU.
"""

from __future__ import annotations

import asyncio

import pytest

from app.models.schemas import FrameResult, TaskState, TaskStatus
from app.services.task_manager import TaskManager


def _frame(i: int) -> FrameResult:
    return FrameResult(
        frame_id=i, timestamp="00:00:00.000", timestamp_seconds=0.0, detections=[]
    )


def _drain(q: asyncio.Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


# ── R-4: bounded queue, drop-oldest for frames ──────────────────────────────

async def test_push_frame_drops_oldest_when_full():
    tm = TaskManager()
    tid = "t1"
    tm._queues[tid] = asyncio.Queue(maxsize=3)
    for i in range(5):
        await tm.push_frame(tid, _frame(i))
    q = tm._queues[tid]
    assert q.qsize() == 3
    ids = [payload.frame_id for (_kind, payload) in _drain(q)]
    # oldest two (0,1) dropped; newest three retained, in order
    assert ids == [2, 3, 4]


async def test_sentinels_never_dropped_and_keep_order():
    tm = TaskManager()
    tid = "t2"
    tm._queues[tid] = asyncio.Queue(maxsize=3)
    q = tm._queues[tid]
    for i in range(3):                      # fill with frames
        await tm.push_frame(tid, _frame(i))
    await tm.push_packaging(tid)            # must land, dropping a frame
    await tm.push_early_terminated(tid, "reason")
    kinds = [k for (k, _p) in _drain(q)]
    assert q.qsize() == 0
    assert "packaging" in kinds and "early_terminated" in kinds
    assert len(kinds) <= 3                  # stayed bounded
    assert kinds.index("packaging") < kinds.index("early_terminated")  # FIFO order


async def test_terminal_sentinel_never_blocks():
    # The bug the bounded queue would have introduced: a blocking put on a full
    # queue hangs the pipeline coroutine forever. push_done must return promptly.
    tm = TaskManager()
    tid = "t3"
    tm._queues[tid] = asyncio.Queue(maxsize=2)
    q = tm._queues[tid]
    for i in range(2):
        await tm.push_frame(tid, _frame(i))
    await asyncio.wait_for(tm.push_done(tid), timeout=1.0)  # would hang pre-fix
    assert tid not in tm._queues                            # push_done pops the registry
    assert "done" in [k for (k, _p) in _drain(q)]           # but landed in the live ref


# ── R-5: release results + LRU eviction ─────────────────────────────────────

def test_release_results_clears_frames():
    tm = TaskManager()
    tid = "t4"
    st = TaskState(task_id=tid, video_id="v", prompt="p", status=TaskStatus.FINISHED)
    st.results = [_frame(i) for i in range(10)]
    tm._tasks[tid] = st
    tm.release_results(tid)
    assert tm._tasks[tid].results == []
    tm.release_results(tid)                # idempotent
    assert tm._tasks[tid].results == []


def test_lru_evicts_oldest_terminal_only(monkeypatch):
    import app.services.task_manager as tmmod
    monkeypatch.setattr(tmmod.settings, "MAX_RETAINED_TASKS", 3)
    tm = TaskManager()
    for i in range(5):                     # 5 terminal tasks, oldest first
        tid = f"f{i}"
        tm._tasks[tid] = TaskState(
            task_id=tid, video_id="v", prompt="p", status=TaskStatus.FINISHED
        )
    tm._tasks["active"] = TaskState(
        task_id="active", video_id="v", prompt="p", status=TaskStatus.RUNNING
    )
    tm._evict_terminal_overflow()
    # cap=3 over 5 terminal → evict 2 oldest; active is never evicted
    assert "f0" not in tm._tasks and "f1" not in tm._tasks
    assert {"f2", "f3", "f4", "active"} <= set(tm._tasks)


# ── detection_frame_count: authoritative frames-with-detections counter ─────

def _det_frame(i: int) -> FrameResult:
    from app.models.schemas import BoundingBox, Detection

    return FrameResult(
        frame_id=i,
        timestamp="00:00:00.000",
        timestamp_seconds=0.0,
        detections=[
            Detection(label="x", score=0.9, bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1))
        ],
    )


def test_detection_frame_count_counts_only_detection_frames():
    tm = TaskManager()
    tid = "t6"
    tm._tasks[tid] = TaskState(
        task_id=tid, video_id="v", prompt="p", status=TaskStatus.RUNNING
    )
    tm.add_frame_result_sync(tid, _frame(0))       # no detections
    tm.add_frame_result_sync(tid, _det_frame(1))
    tm.add_frame_result_sync(tid, _det_frame(2))
    tm.add_frame_result_sync(tid, _frame(3))       # no detections
    st = tm._tasks[tid]
    assert st.processed_frames == 4
    assert st.detection_frame_count == 2


def test_detection_frame_count_survives_release_results():
    tm = TaskManager()
    tid = "t7"
    st = TaskState(task_id=tid, video_id="v", prompt="p", status=TaskStatus.RUNNING)
    tm._tasks[tid] = st
    tm.add_frame_result_sync(tid, _det_frame(0))
    st.status = TaskStatus.FINISHED
    tm.release_results(tid)
    assert tm._tasks[tid].results == []
    assert tm._tasks[tid].detection_frame_count == 1
