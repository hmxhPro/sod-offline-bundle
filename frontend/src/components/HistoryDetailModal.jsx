/**
 * src/components/HistoryDetailModal.jsx
 * --------------------------------------
 * Modal showing one past task's metadata + saved-frame thumbnail grid.
 *
 * Loads in parallel:
 *   - GET /api/task/{taskId}        for status fields (DB-backed if not in memory)
 *   - GET /api/task/{taskId}/frames for the list of saved jpg filenames
 *
 * Clicking a thumbnail opens FramePreview full-screen.
 */

import React, { useEffect, useState } from 'react'
import { X, Download, Clock, FileText, Loader2, ImageOff } from 'lucide-react'
import { getTask, getTaskFrames, getFrameUrl, getDownloadUrl } from '../services/api'
import FramePreview from './FramePreview'

const STATUS_META = {
  pending:          { label: '排队中',  cls: 'bg-brand-50 text-brand-700' },
  running:          { label: '检测中',  cls: 'bg-brand-100 text-brand-700' },
  paused:           { label: '已暂停',  cls: 'bg-amber-50 text-amber-700' },
  packaging:        { label: '打包中',  cls: 'bg-brand-100 text-brand-700' },
  finished:         { label: '已完成',  cls: 'bg-emerald-50 text-emerald-700' },
  failed:           { label: '失败',    cls: 'bg-red-50 text-red-700' },
  cancelled:        { label: '已取消',  cls: 'bg-ink-100 text-ink-600' },
  early_terminated: { label: '提前终止', cls: 'bg-amber-50 text-amber-800' },
}

function formatDateTime(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleString()
}

export default function HistoryDetailModal({ taskId, onClose }) {
  const [task, setTask] = useState(null)
  const [frames, setFrames] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [previewIdx, setPreviewIdx] = useState(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    Promise.allSettled([getTask(taskId), getTaskFrames(taskId)])
      .then(([taskRes, framesRes]) => {
        if (cancelled) return
        if (taskRes.status === 'fulfilled') {
          setTask(taskRes.value)
        } else {
          setError(taskRes.reason?.message || '任务详情加载失败')
        }
        if (framesRes.status === 'fulfilled') {
          setFrames(framesRes.value)
        } else {
          // Frames endpoint 404 just means no saved frames — non-fatal.
          setFrames([])
        }
      })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [taskId])

  // Esc closes — but only when no inner FramePreview is open
  // (FramePreview handles its own Esc).
  useEffect(() => {
    if (previewIdx != null) return
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose, previewIdx])

  // Lock body scroll while open
  useEffect(() => {
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = prev }
  }, [])

  const status = task?.status
  const meta = STATUS_META[status] ?? { label: status || '未知', cls: 'bg-ink-100 text-ink-600' }
  const canDownload = !!task?.zip_ready
  const previewFrames = frames.map((filename) => ({
    taskId,
    image_filename: filename,
    frame_id: filename,           // FramePreview displays this in the toolbar
    timestamp: filename,
    detections: [],
  }))

  return (
    <div
      className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="relative bg-surface rounded-2xl shadow-2xl w-full max-w-4xl max-h-[88vh] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Header ───────────────────────────────────────────────── */}
        <div className="flex items-start justify-between gap-4 px-5 py-4 border-b border-ink-100">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <FileText size={16} className="text-brand-500 flex-shrink-0" />
              <h3 className="font-semibold text-ink-800 truncate" title={task?.video_filename || taskId}>
                {task?.video_filename || taskId.slice(0, 8)}
              </h3>
              <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${meta.cls}`}>
                {meta.label}
              </span>
            </div>
            {task?.prompt && (
              <p className="text-sm text-ink-600 truncate" title={task.prompt}>
                提示词：{task.prompt}
              </p>
            )}
            <div className="mt-1.5 flex items-center gap-4 text-xs text-ink-500 flex-wrap">
              {task?.created_at && (
                <span className="flex items-center gap-1">
                  <Clock size={11} /> {formatDateTime(task.created_at)}
                </span>
              )}
              {task && (
                <span>{task.processed_frames}/{task.total_frames || '?'} 帧</span>
              )}
              {frames.length > 0 && <span>{frames.length} 张缩略图</span>}
            </div>
            {task?.error && (
              <div className="mt-2 px-2.5 py-1.5 rounded-lg bg-red-50 border border-red-200 text-red-700 text-xs">
                ⚠ {task.error}
              </div>
            )}
            {task?.early_terminated && task.termination_reason && (
              <div className="mt-2 px-2.5 py-1.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-xs">
                ℹ {task.termination_reason}
              </div>
            )}
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {canDownload && (
              <a
                href={getDownloadUrl(taskId)}
                download
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-brand-500 text-white hover:bg-brand-600 text-sm font-medium"
              >
                <Download size={14} />
                下载 ZIP
              </a>
            )}
            <button
              type="button"
              onClick={onClose}
              title="关闭"
              className="p-2 rounded-lg text-ink-500 hover:text-red-500 hover:bg-red-50"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* ── Body ─────────────────────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto p-5">
          {loading && (
            <div className="flex items-center justify-center py-16 text-ink-400">
              <Loader2 size={20} className="animate-spin mr-2" />
              <span className="text-sm">加载中…</span>
            </div>
          )}
          {!loading && error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-red-700 text-sm">
              {error}
            </div>
          )}
          {!loading && !error && frames.length === 0 && (
            <div className="flex flex-col items-center justify-center py-16 text-ink-400">
              <ImageOff size={28} className="mb-2" />
              <span className="text-sm">该任务无保存的关键帧</span>
            </div>
          )}
          {!loading && frames.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
              {frames.map((filename, i) => (
                <button
                  key={filename}
                  type="button"
                  onClick={() => setPreviewIdx(i)}
                  className="group relative aspect-video bg-ink-100 rounded-lg overflow-hidden focus:outline-none focus:ring-2 focus:ring-brand-500"
                >
                  <img
                    src={getFrameUrl(taskId, filename)}
                    alt={filename}
                    loading="lazy"
                    className="w-full h-full object-cover transition-transform group-hover:scale-105"
                  />
                  <div className="absolute inset-x-0 bottom-0 px-2 py-1 bg-gradient-to-t from-black/70 to-transparent text-white text-[10px] font-mono truncate">
                    {filename}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Full-screen frame preview (z-50, above this modal) ─────── */}
      {previewIdx != null && previewFrames.length > 0 && (
        <FramePreview
          frames={previewFrames}
          index={Math.min(previewIdx, previewFrames.length - 1)}
          onChangeIndex={setPreviewIdx}
          onClose={() => setPreviewIdx(null)}
        />
      )}
    </div>
  )
}
