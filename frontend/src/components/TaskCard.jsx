/**
 * src/components/TaskCard.jsx
 * ----------------------------
 * One card per uploaded video in the multi-task workspace.
 *
 * Sections:
 *   - Header:   filename, size, video meta, status badge, actions
 *   - Progress: upload / detection progress bar + error
 *   - Live:     real-time detection viewer (auto-shown once frames arrive)
 *   - Modal:    full-screen FramePreview on thumbnail click
 */

import React, { useState, useMemo } from 'react'
import {
  Film, Trash2, RotateCcw, Download, AlertCircle, CheckCircle2, Clock,
  Pause, Play, X, Ban, StopCircle, ChevronDown, ChevronUp,
} from 'lucide-react'
import ProgressBar from './ProgressBar'
import ResultViewer from './ResultViewer'
import FramePreview from './FramePreview'
import { getDownloadUrl } from '../services/api'

const STATUS_META = {
  queued:    { label: '等待中',   className: 'bg-ink-100 text-ink-600',        icon: Clock },
  uploading: { label: '上传中',   className: 'bg-brand-50 text-brand-600',     icon: Clock },
  pending:   { label: '排队中',   className: 'bg-brand-50 text-brand-600',     icon: Clock },
  running:   { label: '检测中',   className: 'bg-brand-100 text-brand-700',    icon: Clock },
  paused:    { label: '已暂停',   className: 'bg-amber-50 text-amber-700',     icon: Pause },
  packaging: { label: '打包中',   className: 'bg-brand-100 text-brand-700',    icon: Clock },
  finished:  { label: '已完成',   className: 'bg-emerald-50 text-emerald-600', icon: CheckCircle2 },
  failed:    { label: '失败',     className: 'bg-red-50 text-red-600',         icon: AlertCircle },
  cancelled: { label: '已取消',   className: 'bg-ink-100 text-ink-600',        icon: Ban },
  terminated: { label: '已终止',  className: 'bg-amber-50 text-amber-700',     icon: StopCircle },
  early_terminated: { label: '提前终止', className: 'bg-amber-50 text-amber-700', icon: CheckCircle2 },
}

function formatSize(bytes) {
  if (!bytes) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1e6) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1e9) return `${(bytes / 1e6).toFixed(1)} MB`
  return `${(bytes / 1e9).toFixed(2)} GB`
}

export default function TaskCard({ task, onRemove, onRetry, onCancel, onPause, onResume, onTerminate, onToggleCollapse }) {
  const meta = STATUS_META[task.taskStatus] ?? STATUS_META.queued
  const StatusIcon = meta.icon
  const collapsed = !!task.collapsed

  const canRemove = !['uploading', 'pending', 'running', 'paused', 'packaging'].includes(task.taskStatus)
  const canRetry = ['failed', 'cancelled'].includes(task.taskStatus)
  const canDownload = task.zipReady && task.taskId
  const canPause = task.taskStatus === 'running'
  const canResume = task.taskStatus === 'paused'
  const canCancel = ['pending', 'running', 'paused'].includes(task.taskStatus)
  const canTerminate = ['running', 'paused'].includes(task.taskStatus)
  const hasFrames =
    !!task.latestFrame || task.allFrames.length > 0 || (task.detectedFrameCount || 0) > 0
  const showViewer =
    ['running', 'paused', 'packaging', 'finished', 'cancelled', 'early_terminated'].includes(task.taskStatus) || hasFrames

  // Preview modal state — index into the "previewList"
  const [previewIdx, setPreviewIdx] = useState(null)

  // previewList: detected history + optionally the current live frame at the end
  const previewList = useMemo(() => {
    const out = [...task.allFrames]
    if (task.latestFrame) {
      // Only append the live frame if it isn't already the last one in history
      const last = out[out.length - 1]
      if (!last || last.frame_id !== task.latestFrame.frame_id) {
        out.push({ ...task.latestFrame, taskId: task.taskId })
      }
    }
    return out
  }, [task.allFrames, task.latestFrame, task.taskId])

  const openPreviewAt = (idx) => setPreviewIdx(idx)
  const openLive = () => {
    const i = previewList.findIndex(
      (f) => f.frame_id === task.latestFrame?.frame_id
    )
    setPreviewIdx(i >= 0 ? i : previewList.length - 1)
  }
  const closePreview = () => setPreviewIdx(null)

  return (
    <div className="card p-4 flex flex-col gap-3">
      {/* ── Header ────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-xl bg-brand-50 text-brand-500 flex-shrink-0">
          <Film size={18} />
        </div>
        <div
          className="flex-1 min-w-0 cursor-pointer"
          onClick={() => onToggleCollapse?.(task.id)}
          title={collapsed ? '展开详情' : '折叠'}
        >
          <div className="flex items-center gap-2 flex-wrap">
            <p className="font-semibold text-ink-800 truncate" title={task.fileName}>
              {task.fileName}
            </p>
            <span
              className={`inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full font-medium ${meta.className}`}
            >
              <StatusIcon size={11} />
              {meta.label}
            </span>
            {task.detectedFrameCount > 0 && (
              <span className="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-medium">
                {task.detectedFrameCount} 帧检测到目标
              </span>
            )}
          </div>
          <p className="text-ink-500 text-xs mt-0.5 flex items-center gap-2 flex-wrap">
            <span>{formatSize(task.fileSize)}</span>
            {task.videoInfo?.total_frames != null && (
              <span>· {task.videoInfo.total_frames} 帧</span>
            )}
            {task.videoInfo?.duration_seconds != null && (
              <span>· {task.videoInfo.duration_seconds.toFixed(1)} s</span>
            )}
            {task.videoInfo?.fps != null && (
              <span>· {task.videoInfo.fps.toFixed(1)} fps</span>
            )}
          </p>
        </div>

        <div className="flex items-center gap-1 flex-wrap justify-end">
          {canPause && (
            <button
              type="button"
              onClick={() => onPause(task.id)}
              title="暂停检测"
              className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-amber-600 hover:bg-amber-50 text-xs"
            >
              <Pause size={14} />
              <span>暂停</span>
            </button>
          )}
          {canResume && (
            <button
              type="button"
              onClick={() => onResume(task.id)}
              title="继续检测"
              className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-emerald-600 hover:bg-emerald-50 text-xs"
            >
              <Play size={14} />
              <span>继续</span>
            </button>
          )}
          {canTerminate && (
            <button
              type="button"
              onClick={() => {
                if (window.confirm('确定要终止并打包结果吗？已处理的帧将被打包成 ZIP 文件。')) {
                  onTerminate(task.id)
                }
              }}
              title="终止并打包已检测的帧"
              className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-orange-600 hover:bg-orange-50 text-xs"
            >
              <StopCircle size={14} />
              <span>终止并打包</span>
            </button>
          )}
          {canCancel && (
            <button
              type="button"
              onClick={() => {
                if (window.confirm('确定要取消该检测任务吗？已处理的帧将保留，但不会生成 ZIP。')) {
                  onCancel(task.id)
                }
              }}
              title="取消检测（不打包）"
              className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-ink-500 hover:text-red-500 hover:bg-red-50 text-xs"
            >
              <X size={14} />
              <span>取消</span>
            </button>
          )}
          {canDownload && (
            <a
              href={getDownloadUrl(task.taskId)}
              download
              title="下载 ZIP"
              className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-emerald-600 hover:bg-emerald-50 text-xs"
            >
              <Download size={14} />
              <span>下载</span>
            </a>
          )}
          {canRetry && (
            <button
              type="button"
              onClick={() => onRetry(task.id)}
              title="重试"
              className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-ink-500 hover:bg-ink-100 text-xs"
            >
              <RotateCcw size={14} />
              <span>重试</span>
            </button>
          )}
          {canRemove && (
            <button
              type="button"
              onClick={() => onRemove(task.id)}
              title="移除"
              className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-ink-400 hover:text-red-500 hover:bg-red-50 text-xs"
            >
              <Trash2 size={14} />
              <span>移除</span>
            </button>
          )}
          <button
            type="button"
            onClick={() => onToggleCollapse?.(task.id)}
            title={collapsed ? '展开详情' : '折叠'}
            className="inline-flex items-center justify-center w-7 h-7 rounded-lg text-ink-400 hover:text-ink-700 hover:bg-ink-100"
          >
            {collapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
          </button>
        </div>
      </div>

      {/* ── Progress ─────────────────────────────────────────────────── */}
      {!collapsed && task.taskStatus !== 'queued' && (
        <ProgressBar
          taskStatus={task.taskStatus}
          progress={task.progress}
          processedFrames={task.processedFrames}
          totalFrames={task.totalFrames}
          uploadProgress={task.uploadProgress}
          compact
        />
      )}

      {/* ── Error ────────────────────────────────────────────────────── */}
      {!collapsed && task.error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-red-700 text-xs">
          ⚠ {task.error}
        </div>
      )}

      {/* ── Early Termination Notice ─────────────────────────────────── */}
      {!collapsed && task.earlyTerminated && task.terminationReason && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-amber-800 text-xs">
          ℹ️ {task.terminationReason}
        </div>
      )}

      {/* ── Real-time viewer (always visible once frames start) ──────── */}
      {!collapsed && showViewer && (
        <div className="pt-2 border-t border-ink-100">
          <ResultViewer
            taskId={task.taskId}
            latestFrame={task.latestFrame}
            allFrames={task.allFrames}
            detectedFrameCount={task.detectedFrameCount}
            taskStatus={task.taskStatus}
            onOpenPreview={openPreviewAt}
            onOpenLiveFrame={openLive}
          />
        </div>
      )}

      {/* ── Modal preview ────────────────────────────────────────────── */}
      {!collapsed && previewIdx != null && previewList.length > 0 && (
        <FramePreview
          frames={previewList}
          index={Math.min(previewIdx, previewList.length - 1)}
          onChangeIndex={setPreviewIdx}
          onClose={closePreview}
        />
      )}
    </div>
  )
}
