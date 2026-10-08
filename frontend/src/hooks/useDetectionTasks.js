/**
 * src/hooks/useDetectionTasks.js
 * -------------------------------
 * Manage MULTIPLE concurrent detection workflows in one state tree.
 *
 * Each "task" represents one video and goes through phases:
 *   queued → uploading → pending → running → finished | failed
 *
 * The hook exposes:
 *   - tasks: TaskItem[]
 *   - addFiles(files): push new queued tasks
 *   - removeTask(id)
 *   - clearAll()
 *   - startAll(prompt, detectionInterval): fire upload+detect+stream for every queued task
 *   - startOne(id, prompt, detectionInterval)
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import {
  uploadVideo, startDetection, getStreamUrl, getTask,
  cancelDetection, pauseDetection, resumeDetection, terminateDetection,
} from '../services/api'
import { loadTasks, saveTasks } from '../services/taskStorage'

let _nextId = 1
const makeId = () => `t_${Date.now()}_${_nextId++}`

// Tasks that have reached an end state. When new videos are uploaded these are
// collapsed to a summary row so the workspace focuses on the current task(s).
const TERMINAL_STATUSES = ['finished', 'failed', 'cancelled', 'early_terminated']

function newTask(file) {
  return {
    id: makeId(),
    file,
    fileName: file.name,
    fileSize: file.size,

    videoId: null,
    taskId: null,
    taskStatus: 'queued',   // queued | uploading | pending | running | paused | packaging | finished | failed | cancelled | early_terminated

    uploadProgress: 0,
    progress: 0,
    processedFrames: 0,
    totalFrames: 0,

    videoInfo: null,
    latestFrame: null,
    allFrames: [],
    detectedFrameCount: 0,

    error: null,
    zipReady: false,
    earlyTerminated: false,
    terminationReason: null,
    collapsed: false,        // UI: summary-row when a newer task is uploaded
  }
}

export function useDetectionTasks() {
  const [tasks, setTasks] = useState(() => {
    // Lazy init: rehydrate from localStorage. Tasks that were mid-upload
    // when the previous page closed have lost their File handle and can't
    // continue — surface them as failed so the user knows to re-pick.
    const persisted = loadTasks()
    return persisted.map((t) =>
      ['queued', 'uploading'].includes(t.taskStatus)
        ? { ...t, taskStatus: 'failed', error: '页面刷新导致上传中断，请重新选择文件' }
        : t
    )
  })
  const tasksRef = useRef([])
  const streamsRef = useRef(new Map()) // id -> EventSource
  // reconcileWithServer needs to reopen the SSE stream, but openStream's
  // onerror needs reconcileWithServer — the ref breaks the definition cycle.
  const openStreamRef = useRef(null)
  // id -> consecutive stream reopens without a single delivered event. Caps
  // the reconnect loop when SSE errors right back (e.g. a stale DB row still
  // says "running" after a backend restart, so /api/stream 404s forever).
  const reopenAttemptsRef = useRef(new Map())
  // id -> generation token. Each reconcileWithServer invocation bumps it so a
  // superseded (older) reconcile loop exits instead of running concurrently.
  const reconcileGenRef = useRef(new Map())

  // Keep tasksRef in sync with tasks state
  useEffect(() => {
    tasksRef.current = tasks
  }, [tasks])

  // Cleanup all EventSource connections on unmount
  useEffect(() => {
    return () => {
      streamsRef.current.forEach((es) => es.close())
      streamsRef.current.clear()
    }
  }, [])

  // Persist the task list to localStorage on every change so a refresh
  // can restore it. Sanitization (drop File, image_b64) lives in saveTasks.
  useEffect(() => {
    saveTasks(tasks)
  }, [tasks])

  const patchTask = useCallback((id, patch) => {
    setTasks((prev) =>
      prev.map((t) =>
        t.id === id
          ? (typeof patch === 'function' ? { ...t, ...patch(t) } : { ...t, ...patch })
          : t
      )
    )
  }, [])

  const closeStream = useCallback((id) => {
    const es = streamsRef.current.get(id)
    if (es) {
      es.close()
      streamsRef.current.delete(id)
    }
    reopenAttemptsRef.current.delete(id)
    reconcileGenRef.current.delete(id)
  }, [])

  // ── Public: add newly selected files as queued tasks ────────────────────
  const addFiles = useCallback((files) => {
    const list = Array.from(files || []).filter(Boolean)
    if (!list.length) return
    // Collapse already-ended tasks so the workspace focuses on the new
    // (current) ones. In-progress tasks stay expanded (don't hide live view).
    setTasks((prev) => [
      ...prev.map((t) =>
        TERMINAL_STATUSES.includes(t.taskStatus) ? { ...t, collapsed: true } : t
      ),
      ...list.map(newTask),
    ])
  }, [])

  const removeTask = useCallback((id) => {
    closeStream(id)
    setTasks((prev) => prev.filter((t) => t.id !== id))
  }, [closeStream])

  const clearAll = useCallback(() => {
    tasks.forEach((t) => closeStream(t.id))
    setTasks([])
  }, [tasks, closeStream])

  // Toggle (or explicitly set) a task's collapsed summary-row state.
  const toggleCollapse = useCallback((id, value) => {
    patchTask(id, (t) => ({
      collapsed: typeof value === 'boolean' ? value : !t.collapsed,
    }))
  }, [patchTask])

  const resetOne = useCallback((id) => {
    closeStream(id)
    setTasks((prev) =>
      prev.map((t) =>
        t.id === id
          ? { ...t, ...newTask(t.file), id: t.id } // keep same id
          : t
      )
    )
  }, [closeStream])

  // ── SSE event handler ──────────────────────────────────────────────────
  const handleStreamEvent = useCallback((id, data) => {
    // Validate data structure to prevent crashes on malformed SSE
    if (!data || typeof data !== 'object') {
      console.warn('Invalid SSE data received:', data)
      return
    }

    // A delivered event proves the stream works — reset the reopen budget.
    reopenAttemptsRef.current.delete(id)

    switch (data.event_type) {
      case 'frame':
        setTasks((prev) =>
          prev.map((t) => {
            if (t.id !== id) return t
            const fr = data.frame_result
            if (!fr || typeof fr !== 'object') return t
            const { image_b64, ...rest } = fr
            // Keep image_b64 as a fallback when the frame wasn't saved to disk
            // (otherwise the thumbnail has nothing to render).
            const meta = fr.image_filename ? rest : { ...rest, image_b64 }
            const hasDetection = Array.isArray(fr.detections) && fr.detections.length > 0
            // If we're paused, a late frame should still update the preview
            // and counters but must NOT kick us back to "running".
            const nextStatus = t.taskStatus === 'paused' ? 'paused' : 'running'
            // Cap at 300 to prevent React from slowing down on long videos
            const nextAllFrames = hasDetection
              ? [...t.allFrames, { ...meta, taskId: t.taskId }].slice(-300)
              : t.allFrames
            return {
              ...t,
              taskStatus: nextStatus,
              progress: data.progress ?? t.progress,
              processedFrames: data.processed_frames ?? t.processedFrames,
              totalFrames: data.total_frames ?? t.totalFrames,
              latestFrame: fr ?? t.latestFrame,
              allFrames: nextAllFrames,
              // The server counter is authoritative — and read at DELIVERY
              // time, so it may already include frames still in flight.
              // Never add the local +1 on top of it: replayed backlogs after
              // a reconnect would inflate the count permanently. The local
              // +1 only serves backends without the field.
              detectedFrameCount: data.detection_frame_count != null
                ? Math.max(t.detectedFrameCount || 0, data.detection_frame_count)
                : (t.detectedFrameCount || 0) + (hasDetection ? 1 : 0),
            }
          })
        )
        break
      case 'paused':
        patchTask(id, { taskStatus: 'paused' })
        break
      case 'resumed':
        patchTask(id, { taskStatus: 'running' })
        break
      case 'cancelled':
        patchTask(id, (t) => ({
          taskStatus: 'cancelled',
          processedFrames: data.processed_frames ?? t.processedFrames,
          totalFrames: data.total_frames ?? t.totalFrames,
          detectedFrameCount: Math.max(t.detectedFrameCount || 0, data.detection_frame_count ?? 0),
        }))
        break
      case 'packaging':
        patchTask(id, (t) => ({
          taskStatus: 'packaging',
          progress: 1.0,
          processedFrames: data.processed_frames ?? t.processedFrames,
          totalFrames: data.total_frames ?? t.totalFrames,
          detectedFrameCount: Math.max(t.detectedFrameCount || 0, data.detection_frame_count ?? 0),
        }))
        break
      case 'early_terminated':
        patchTask(id, (t) => ({
          taskStatus: 'early_terminated',
          progress: data.progress ?? t.progress,
          processedFrames: data.processed_frames ?? t.processedFrames,
          totalFrames: data.total_frames ?? t.totalFrames,
          detectedFrameCount: Math.max(t.detectedFrameCount || 0, data.detection_frame_count ?? 0),
          earlyTerminated: true,
          terminationReason: data.error || 'Early termination triggered',
        }))
        break
      case 'done':
        closeStream(id)
        patchTask(id, (t) => {
          // If we already transitioned to 'cancelled' or 'early_terminated', don't override.
          if (t.taskStatus === 'cancelled' || t.taskStatus === 'early_terminated') return {}
          return {
            taskStatus: 'finished',
            progress: 1.0,
            processedFrames: data.processed_frames ?? t.processedFrames,
            detectedFrameCount: Math.max(t.detectedFrameCount || 0, data.detection_frame_count ?? 0),
            zipReady: true,
          }
        })
        break
      case 'error':
        closeStream(id)
        patchTask(id, {
          taskStatus: 'failed',
          error: data.error || 'Unknown server error.',
        })
        break
      default:
        break
    }
  }, [closeStream, patchTask])

  // ── Reconcile with server when SSE drops mid-task ──────────────────────
  // Polls /api/task/{taskId} until the server reports a terminal state,
  // so a mid-stream disconnect (e.g. during slow ZIP packaging) does NOT
  // get misreported as "failed" on the client.
  const reconcileWithServer = useCallback(async (id) => {
    const t0 = tasksRef.current.find((t) => t.id === id)
    const taskId = t0?.taskId
    if (!taskId) {
      patchTask(id, { taskStatus: 'failed', error: '流连接已断开' })
      return
    }

    // Supersede any older reconcile loop still sleeping for this id.
    const gen = (reconcileGenRef.current.get(id) || 0) + 1
    reconcileGenRef.current.set(id, gen)

    const POLL_INTERVAL_MS = 4000
    const MAX_ATTEMPTS = 90  // ~6 min total grace period
    const MAX_STREAM_REOPENS = 5
    for (let i = 0; i < MAX_ATTEMPTS; i++) {
      // Bail out if a newer reconcile took over, or if the task was removed
      // or restarted (different backend taskId) while we slept — a stale
      // loop must not patch state or touch streams for a run it no longer
      // owns.
      if (reconcileGenRef.current.get(id) !== gen) return
      if (tasksRef.current.find((t) => t.id === id)?.taskId !== taskId) return
      try {
        const state = await getTask(taskId)
        // Re-check after the await: a stream 'done'/'error' may have landed
        // while this poll was in flight (its closeStream deletes the gen
        // entry) — patching the stale response would flick the card from
        // finished back to running and reopen a dead stream.
        if (reconcileGenRef.current.get(id) !== gen) return
        // Server-side detected-frame counter: authoritative floor for the
        // local count (SSE frames can be dropped server-side mid-disconnect).
        const mergeCount = (t) =>
          Math.max(t.detectedFrameCount || 0, state.detection_frame_count || 0)
        if (state.status === 'finished') {
          closeStream(id)
          patchTask(id, (t) => ({
            taskStatus: 'finished',
            progress: 1.0,
            processedFrames: state.processed_frames,
            totalFrames: state.total_frames,
            detectedFrameCount: mergeCount(t),
            zipReady: !!state.zip_ready,
          }))
          return
        }
        if (state.status === 'early_terminated') {
          closeStream(id)
          patchTask(id, (t) => ({
            taskStatus: 'early_terminated',
            progress: state.progress,
            processedFrames: state.processed_frames,
            totalFrames: state.total_frames,
            detectedFrameCount: mergeCount(t),
            zipReady: !!state.zip_ready,
            earlyTerminated: true,
            terminationReason: state.termination_reason || 'Early termination triggered',
          }))
          return
        }
        if (state.status === 'failed') {
          closeStream(id)
          patchTask(id, {
            taskStatus: 'failed',
            error: state.error || '任务在服务器端失败',
          })
          return
        }
        if (state.status === 'cancelled') {
          closeStream(id)
          patchTask(id, (t) => ({
            taskStatus: 'cancelled',
            processedFrames: state.processed_frames,
            totalFrames: state.total_frames,
            detectedFrameCount: mergeCount(t),
          }))
          return
        }
        if (state.status === 'paused') {
          patchTask(id, (t) => ({
            taskStatus: 'paused',
            progress: state.progress,
            processedFrames: state.processed_frames,
            totalFrames: state.total_frames,
            detectedFrameCount: mergeCount(t),
            error: null,
          }))
        } else {
          // Still running / packaging.
          const nextStatus =
            state.status === 'packaging' ||
            state.processed_frames >= state.total_frames
              ? 'packaging'
              : 'running'
          patchTask(id, (t) => ({
            taskStatus: nextStatus,
            progress: state.progress,
            processedFrames: state.processed_frames,
            totalFrames: state.total_frames,
            detectedFrameCount: mergeCount(t),
            error: null,
          }))
        }
        // The task is still alive and SSE is the only source of frame events
        // (live view, thumbnails, detected-frame counter) — the backend
        // keeps the event queue alive for reconnects, so reattach. But keep
        // polling as a watchdog: the server's single terminal sentinel can
        // be stolen by a zombie generator on a half-dead connection, in
        // which case the reopened stream would idle on keepalives until the
        // backend's synthetic-terminal heartbeat — this poll loop is the
        // client-side belt to that suspender.
        const attempts = reopenAttemptsRef.current.get(id) || 0
        if (!streamsRef.current.has(id) && attempts < MAX_STREAM_REOPENS) {
          reopenAttemptsRef.current.set(id, attempts + 1)
          openStreamRef.current?.(id, taskId)
        }
        if (streamsRef.current.has(id)) {
          // The attempts entry is cleared when the stream delivers any
          // event — at that point SSE is proven healthy and owns the task.
          if (!reopenAttemptsRef.current.has(id)) return
          // A paused task can stay paused indefinitely; with a stream
          // attached, resume/cancel will arrive there. Don't burn the
          // 6-min poll budget into a spurious 'failed'.
          if (state.status === 'paused') return
        }
      } catch (err) {
        // FastAPI returns 404 with detail "Task '...' not found." once the
        // backend restarts and loses its in-memory task. The axios
        // interceptor flattens that detail into err.message — bail out
        // immediately so we don't burn the whole 6-min grace period.
        if (err?.message?.includes('not found')) {
          patchTask(id, { taskStatus: 'failed', error: '服务已重启，任务已不存在' })
          return
        }
        // Network blip — keep retrying within grace period.
        console.warn('reconcile poll failed, retrying:', err?.message)
      }
      await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS))
    }
    // Poll budget exhausted. With a live stream still attached, leave the
    // status alone — SSE (or the backend's synthetic-terminal heartbeat)
    // will deliver the ending; stamping 'failed' here would mislabel a
    // long-running task whose stream is healthy but quiet (e.g. packaging).
    if (!streamsRef.current.has(id)) {
      patchTask(id, { taskStatus: 'failed', error: '与服务器失联，已超过等待时长' })
    }
  }, [patchTask, closeStream])

  // ── Open (or reopen) the SSE stream for a backend task ────────────────
  // Extracted so both the initial run and the post-refresh rehydrate path
  // can use it.
  const openStream = useCallback((id, taskId) => {
    // Close any prior stream for this id before opening a new one.
    const prev = streamsRef.current.get(id)
    if (prev) prev.close()

    const es = new EventSource(getStreamUrl(taskId))
    streamsRef.current.set(id, es)

    es.onmessage = (evt) => {
      try {
        handleStreamEvent(id, JSON.parse(evt.data))
      } catch (e) {
        console.error('SSE parse failed:', e)
      }
    }
    es.onerror = async () => {
      es.close()
      streamsRef.current.delete(id)
      // SSE can drop during long ZIP packaging even with heartbeats
      // (proxies, WiFi, sleep). Before declaring failure, ask the
      // server for the authoritative task state.
      try {
        await reconcileWithServer(id)
      } catch (err) {
        patchTask(id, { taskStatus: 'failed', error: err.message || '流连接失败' })
      }
    }
  }, [handleStreamEvent, reconcileWithServer, patchTask])

  // Late binding for reconcileWithServer (declared before openStream).
  useEffect(() => {
    openStreamRef.current = openStream
  }, [openStream])

  // ── Run a single task through the full workflow ────────────────────────
  const runTask = useCallback(async (id, prompt, detectionInterval, enableVlm, modelId) => {
    const current = tasksRef.current.find((t) => t.id === id)
    if (!current || !current.file) return

    patchTask(id, { taskStatus: 'uploading', uploadProgress: 0, error: null })

    try {
      // 1. Upload
      const videoInfo = await uploadVideo(current.file, (pct) => {
        patchTask(id, { uploadProgress: pct })
      })
      patchTask(id, { videoInfo, videoId: videoInfo.video_id, uploadProgress: 100 })

      // 2. Start detection. With a trained model selected, classes are baked
      // into the weights so we send model_id and an empty prompt (the backend
      // requires exactly one of prompt / model_id).
      patchTask(id, { taskStatus: 'pending' })
      const task = await startDetection({
        video_id: videoInfo.video_id,
        video_filename: videoInfo.filename,
        prompt: modelId ? '' : prompt,
        model_id: modelId || undefined,
        detection_interval: detectionInterval || undefined,
        enable_vlm: enableVlm,
      })
      patchTask(id, { taskId: task.task_id, taskStatus: 'pending' })

      // 3. SSE
      openStream(id, task.task_id)
    } catch (err) {
      patchTask(id, { taskStatus: 'failed', error: err.message || String(err) })
    }
  }, [patchTask, openStream])

  // ── Public: start every queued task in parallel ────────────────────────
  const startAll = useCallback(async (prompt, detectionInterval, enableVlm, modelId) => {
    const toStart = tasksRef.current.filter((t) =>
      ['queued', 'failed', 'cancelled'].includes(t.taskStatus)
    )

    // Reset non-queued tasks first
    for (const t of toStart) {
      if (t.taskStatus !== 'queued') {
        closeStream(t.id)
        setTasks((prev) =>
          prev.map((x) =>
            x.id === t.id ? { ...x, ...newTask(x.file), id: x.id } : x
          )
        )
      }
    }

    // Capture IDs before async operations to avoid stale references
    const ids = toStart.map((t) => t.id)

    // Validate tasks still exist before running
    await Promise.all(
      ids.map((id) => {
        const exists = tasksRef.current.find((t) => t.id === id)
        return exists ? runTask(id, prompt, detectionInterval, enableVlm, modelId) : Promise.resolve()
      })
    )
  }, [runTask, closeStream])

  const startOne = useCallback((id, prompt, detectionInterval, enableVlm, modelId) => {
    return runTask(id, prompt, detectionInterval, enableVlm, modelId)
  }, [runTask])

  // ── Rehydrate persisted tasks on mount ─────────────────────────────────
  // For each task with a known backend taskId, sync state from the server
  // and reopen SSE if the server still says it's active. Runs once.
  useEffect(() => {
    const ACTIVE = ['pending', 'running', 'paused', 'packaging']
    // tasksRef is sync'd from the lazy-init `tasks` by an earlier effect
    // that fires before this one. Capture the snapshot synchronously.
    const initial = tasksRef.current.length ? tasksRef.current : tasks
    initial.forEach(async (t) => {
      if (!t.taskId) return  // never reached the backend — nothing to rehydrate
      try {
        const state = await getTask(t.taskId)
        patchTask(t.id, (cur) => ({
          taskStatus: state.status,
          progress: state.progress ?? 0,
          processedFrames: state.processed_frames ?? 0,
          totalFrames: state.total_frames ?? 0,
          detectedFrameCount: Math.max(
            cur.detectedFrameCount || 0,
            state.detection_frame_count || 0
          ),
          zipReady: !!state.zip_ready,
          earlyTerminated: !!state.early_terminated,
          terminationReason: state.termination_reason ?? null,
          error: state.error ?? null,
        }))
        if (ACTIVE.includes(state.status)) {
          openStream(t.id, t.taskId)
        }
      } catch (err) {
        const msg = err?.message || ''
        const reason = msg.includes('not found')
          ? '服务已重启，任务已不存在'
          : (msg || '恢复任务失败')
        patchTask(t.id, { taskStatus: 'failed', error: reason })
      }
    })
    // Mount-only — intentionally empty deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Control actions: cancel / pause / resume ──────────────────────────
  const _getBackendTaskId = (id) => {
    const t = tasksRef.current.find((x) => x.id === id)
    return t?.taskId ?? null
  }

  const cancel = useCallback(async (id) => {
    const backendId = _getBackendTaskId(id)
    if (!backendId) {
      // Task never reached the server — just drop it locally.
      patchTask(id, { taskStatus: 'cancelled' })
      return
    }
    try {
      await cancelDetection(backendId)
      // Optimistic UI; the 'cancelled' SSE event will confirm.
      patchTask(id, (t) =>
        ['finished', 'failed', 'cancelled'].includes(t.taskStatus)
          ? {}
          : { taskStatus: 'cancelled' }
      )
    } catch (err) {
      patchTask(id, { error: err.message || String(err) })
    }
  }, [patchTask])

  const pause = useCallback(async (id) => {
    const backendId = _getBackendTaskId(id)
    if (!backendId) return
    try {
      await pauseDetection(backendId)
      patchTask(id, (t) =>
        t.taskStatus === 'running' ? { taskStatus: 'paused' } : {}
      )
    } catch (err) {
      patchTask(id, { error: err.message || String(err) })
    }
  }, [patchTask])

  const resume = useCallback(async (id) => {
    const backendId = _getBackendTaskId(id)
    if (!backendId) return
    try {
      await resumeDetection(backendId)
      patchTask(id, (t) =>
        t.taskStatus === 'paused' ? { taskStatus: 'running' } : {}
      )
    } catch (err) {
      patchTask(id, { error: err.message || String(err) })
    }
  }, [patchTask])

  const terminate = useCallback(async (id) => {
    const backendId = _getBackendTaskId(id)
    if (!backendId) return
    try {
      await terminateDetection(backendId)
      // Optimistic UI; the 'early_terminated' SSE event will confirm.
      patchTask(id, (t) =>
        ['running', 'paused'].includes(t.taskStatus)
          ? { taskStatus: 'packaging' }
          : {}
      )
    } catch (err) {
      patchTask(id, { error: err.message || String(err) })
    }
  }, [patchTask])

  return {
    tasks,
    addFiles,
    removeTask,
    clearAll,
    toggleCollapse,
    resetOne,
    startAll,
    startOne,
    cancel,
    pause,
    resume,
    terminate,
  }
}
