/**
 * src/components/ResultViewer.jsx
 * ---------------------------------
 * Real-time detection viewer for a single task.
 *
 * Layout:
 *   - Top: LIVE big view (auto-updates on every SSE frame event)
 *   - Bottom: scrollable grid of "detected-target frames" (thumbnails)
 *
 * Thumbnail clicks bubble up via onOpenPreview(index) so the parent can
 * display a full-screen FramePreview modal.
 */

import React, { useRef, useEffect, useState } from 'react'
import { Clock, Target, Hash, Maximize2 } from 'lucide-react'
import { getFrameUrl } from '../services/api'

function FrameThumbnail({ frame, onClick, idx }) {
  const detCount = frame.detections?.length ?? 0
  const imgSrc = frame.taskId && frame.image_filename
    ? getFrameUrl(frame.taskId, frame.image_filename)
    : frame.image_b64
      ? `data:image/jpeg;base64,${frame.image_b64}`
      : null
  return (
    <button
      type="button"
      onClick={() => onClick(idx)}
      className="frame-card-enter group relative rounded-lg overflow-hidden border border-ink-200 hover:border-brand-400 bg-surface transition-colors focus:outline-none focus:ring-2 focus:ring-brand-200"
      title="点击放大"
    >
      <img
        src={imgSrc}
        alt={`Frame ${frame.frame_id}`}
        className="w-full h-24 object-cover"
        loading="lazy"
      />
      <div className="absolute inset-0 bg-black/0 group-hover:bg-black/15 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100">
        <Maximize2 size={16} className="text-white drop-shadow" />
      </div>
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/75 to-transparent px-2 py-1">
        <p className="text-white text-[11px] font-mono truncate">{frame.timestamp}</p>
        {detCount > 0 && (
          <p className="text-emerald-200 text-[11px]">{detCount} 个目标</p>
        )}
      </div>
    </button>
  )
}

function DetectionBadge({ det }) {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-brand-200 bg-brand-50 text-brand-900 text-sm">
      <Target size={12} className="flex-shrink-0" />
      <span className="truncate">{det.label}</span>
      <span className="ml-auto text-xs">{(det.score * 100).toFixed(0)}%</span>
      {det.track_id != null && (
        <span className="text-xs font-mono">#{det.track_id}</span>
      )}
      {det.vlm_verified && (
        <span className="text-xs" title="AI 已复核">🔍</span>
      )}
    </div>
  )
}

export default function ResultViewer({
  taskId,
  latestFrame,
  allFrames = [],
  detectedFrameCount,
  taskStatus,
  onOpenPreview,     // (idx: number) => void
  onOpenLiveFrame,   // () => void  — enlarge the current live frame (not in history)
}) {
  const gridRef = useRef(null)

  // allFrames is capped (useDetectionTasks keeps the last 300 for thumbnails)
  // and thinned on refresh (frames never saved to disk are dropped from
  // localStorage); detectedFrameCount is the uncapped counter, floored by the
  // server's authoritative count. max() covers tasks persisted before the
  // counter existed.
  const trueDetected = Math.max(detectedFrameCount || 0, allFrames.length)

  useEffect(() => {
    if (gridRef.current && taskStatus === 'running') {
      gridRef.current.scrollTop = gridRef.current.scrollHeight
    }
  }, [allFrames.length, taskStatus])

  if (!latestFrame && allFrames.length === 0 && trueDetected === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-ink-400 border-2 border-dashed border-ink-200 rounded-xl bg-ink-50/60">
        <p className="text-sm">等待第一帧结果…</p>
      </div>
    )
  }

  const latestDets = latestFrame?.detections ?? []
  const liveDetCount = latestDets.length
  // Prefer the disk URL so the live view stays usable after a page
  // refresh (image_b64 is stripped before persistence).
  const liveSrc = latestFrame
    ? (taskId && latestFrame.image_filename
        ? getFrameUrl(taskId, latestFrame.image_filename)
        : latestFrame.image_b64
          ? `data:image/jpeg;base64,${latestFrame.image_b64}`
          : null)
    : null

  return (
    <div className="flex flex-col gap-3">
      {/* ── Live view ───────────────────────────────────────────────────── */}
      {latestFrame && liveSrc && (
        <div className="rounded-xl overflow-hidden border border-ink-200 bg-surface">
          <div className="flex items-center justify-between px-4 py-2 bg-ink-50 border-b border-ink-200">
            <div className="flex items-center gap-3 text-xs">
              <div className="flex items-center gap-1.5 text-brand-600 font-mono">
                <Clock size={12} />
                <span>{latestFrame.timestamp}</span>
              </div>
              <div className="flex items-center gap-1.5 text-ink-500">
                <Hash size={12} />
                <span>帧 {latestFrame.frame_id}</span>
              </div>
              <div className="flex items-center gap-1.5 text-ink-500">
                <Target size={12} />
                <span>{liveDetCount} 个目标</span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {taskStatus === 'running' && (
                <span className="flex items-center gap-1.5 text-xs text-emerald-600">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 soft-pulse" />
                  实时检测中
                </span>
              )}
              {onOpenLiveFrame && (
                <button
                  type="button"
                  onClick={onOpenLiveFrame}
                  className="flex items-center gap-1 px-2 py-1 rounded-lg text-ink-500 hover:text-brand-600 hover:bg-surface text-xs"
                  title="放大查看"
                >
                  <Maximize2 size={12} />
                  放大
                </button>
              )}
            </div>
          </div>

          <button
            type="button"
            onClick={onOpenLiveFrame}
            className="block w-full cursor-zoom-in bg-slate-900/90"
            title="点击放大"
          >
            <img
              src={liveSrc}
              alt={`Detection frame ${latestFrame.frame_id}`}
              className="w-full object-contain max-h-[420px]"
            />
          </button>

          {liveDetCount > 0 && (
            <div className="px-4 py-3 flex flex-wrap gap-2 bg-surface">
              {latestDets.map((det, i) => (
                <DetectionBadge key={i} det={det} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── History (only detection-positive frames) ────────────────────── */}
      {trueDetected > 0 && (
        <div>
          <h4 className="text-ink-600 text-xs mb-2 flex items-center gap-1.5">
            <Target size={12} className="text-brand-500" />
            检测到目标的帧 · <span className="font-mono text-ink-800">{trueDetected}</span>
            <span className="text-ink-400">
              {allFrames.length === 0
                ? '（缩略图不可用，完整结果见下载的 ZIP）'
                : trueDetected > allFrames.length
                  ? `（仅展示最近 ${allFrames.length} 帧缩略图，点击放大）`
                  : '（点击缩略图放大）'}
            </span>
          </h4>
          {allFrames.length > 0 && (
            <div
              ref={gridRef}
              className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2 max-h-56 overflow-y-auto pr-1"
            >
              {allFrames.map((frame, i) => (
                <FrameThumbnail
                  key={`${frame.frame_id}-${i}`}
                  frame={frame}
                  idx={i}
                  onClick={onOpenPreview}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
