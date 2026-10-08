/**
 * src/components/ProgressBar.jsx
 * --------------------------------
 * Light-theme progress indicator used for each task row.
 */

import React from 'react'
import { Loader2 } from 'lucide-react'

export default function ProgressBar({
  taskStatus,
  progress,
  processedFrames,
  totalFrames,
  uploadProgress,
  compact = false,
}) {
  const isUploading = taskStatus === 'uploading'
  const isRunning = taskStatus === 'running' || taskStatus === 'pending'
  const isPaused = taskStatus === 'paused'
  const isPackaging = taskStatus === 'packaging'
  const isFinished = taskStatus === 'finished'
  const isFailed = taskStatus === 'failed'
  const isCancelled = taskStatus === 'cancelled'

  if (taskStatus === 'idle' || taskStatus === 'queued') return null

  const pct = isUploading
    ? uploadProgress
    : Math.round((progress ?? 0) * 100)

  const label = isUploading
    ? `上传中 ${pct}%`
    : taskStatus === 'pending'
    ? '任务排队中…'
    : isRunning
    ? `检测中 ${processedFrames} / ${totalFrames || '?'} 帧`
    : isPaused
    ? `已暂停 · ${processedFrames} / ${totalFrames || '?'} 帧`
    : isPackaging
    ? `打包结果中… 共 ${processedFrames} 帧`
    : isFinished
    ? `完成 · 共 ${processedFrames} 帧`
    : isCancelled
    ? `已取消 · 处理 ${processedFrames} / ${totalFrames || '?'} 帧`
    : isFailed
    ? '任务失败'
    : ''

  const barColor = isFailed || isCancelled
    ? 'bg-ink-400'
    : isFinished
    ? 'bg-emerald-500'
    : isPaused
    ? 'bg-amber-400'
    : isPackaging
    ? 'bg-brand-400 progress-glow'
    : 'bg-brand-500 progress-glow'

  const pctColor = isFailed
    ? 'text-red-500'
    : isCancelled
    ? 'text-ink-500'
    : isFinished
    ? 'text-emerald-600'
    : isPaused
    ? 'text-amber-600'
    : 'text-brand-600'

  return (
    <div className="flex flex-col gap-1.5">
      <div className={`flex items-center justify-between ${compact ? 'text-xs' : 'text-sm'}`}>
        <div className="flex items-center gap-2 text-ink-700">
          {(isUploading || isRunning || isPackaging) && (
            <Loader2 size={13} className="animate-spin text-brand-500" />
          )}
          <span>{label}</span>
        </div>
        <span className={`font-mono ${pctColor}`}>{pct}%</span>
      </div>

      <div className="h-1.5 bg-ink-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
    </div>
  )
}
