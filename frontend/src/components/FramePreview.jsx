/**
 * src/components/FramePreview.jsx
 * ---------------------------------
 * Full-screen modal for previewing a single annotated frame.
 *
 * Features:
 *   - ESC to close
 *   - Click backdrop to close
 *   - ← / → arrow keys to navigate (or on-screen buttons)
 *   - Displays timestamp, frame_id, detection list
 *   - "Download" button opens the image in a new tab (for Save As)
 */

import React, { useEffect, useCallback } from 'react'
import { X, ChevronLeft, ChevronRight, Download, Clock, Hash, Target } from 'lucide-react'
import { getFrameUrl } from '../services/api'

function resolveSrc(frame) {
  if (!frame) return null
  if (frame.taskId && frame.image_filename) {
    return getFrameUrl(frame.taskId, frame.image_filename)
  }
  if (frame.image_b64) return `data:image/jpeg;base64,${frame.image_b64}`
  return null
}

export default function FramePreview({ frames, index, onClose, onChangeIndex }) {
  const frame = frames?.[index] ?? null
  const total = frames?.length ?? 0
  const hasPrev = index > 0
  const hasNext = index < total - 1

  const prev = useCallback(() => {
    if (hasPrev) onChangeIndex(index - 1)
  }, [hasPrev, index, onChangeIndex])

  const next = useCallback(() => {
    if (hasNext) onChangeIndex(index + 1)
  }, [hasNext, index, onChangeIndex])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowLeft') prev()
      else if (e.key === 'ArrowRight') next()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose, prev, next])

  // Lock body scroll while open
  useEffect(() => {
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = prev
    }
  }, [])

  if (!frame) return null
  const src = resolveSrc(frame)
  const detections = frame.detections ?? []

  return (
    <div
      className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
    >
      {/* Close */}
      <button
        type="button"
        onClick={onClose}
        className="absolute top-4 right-4 p-2.5 rounded-full bg-white/10 text-white hover:bg-white/20 transition-colors"
        title="关闭 (Esc)"
      >
        <X size={18} />
      </button>

      {/* Nav: prev */}
      {hasPrev && (
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); prev() }}
          className="absolute left-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-white/10 text-white hover:bg-white/20 transition-colors"
          title="上一帧 (←)"
        >
          <ChevronLeft size={22} />
        </button>
      )}

      {/* Nav: next */}
      {hasNext && (
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); next() }}
          className="absolute right-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-white/10 text-white hover:bg-white/20 transition-colors"
          title="下一帧 (→)"
        >
          <ChevronRight size={22} />
        </button>
      )}

      {/* Content */}
      <div
        className="relative max-w-6xl w-full max-h-[92vh] flex flex-col bg-slate-900 rounded-2xl overflow-hidden shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Toolbar */}
        <div className="flex items-center justify-between px-5 py-3 bg-slate-800/80 border-b border-slate-700 text-sm">
          <div className="flex items-center gap-4 text-white/90">
            <span className="flex items-center gap-1.5 text-emerald-200 font-mono">
              <Clock size={13} /> {frame.timestamp}
            </span>
            <span className="flex items-center gap-1.5 text-slate-300">
              <Hash size={13} /> 帧 {frame.frame_id}
            </span>
            <span className="flex items-center gap-1.5 text-slate-300">
              <Target size={13} /> {detections.length} 个目标
            </span>
            <span className="text-slate-400 text-xs hidden sm:inline">
              {index + 1} / {total}
            </span>
          </div>
          {src && (
            <a
              href={src}
              target="_blank"
              rel="noopener noreferrer"
              download={frame.image_filename || `frame_${frame.frame_id}.jpg`}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-brand-500 text-white hover:bg-brand-600 text-xs font-medium"
              onClick={(e) => e.stopPropagation()}
            >
              <Download size={13} />
              下载图片
            </a>
          )}
        </div>

        {/* Image */}
        <div className="flex-1 min-h-0 flex items-center justify-center bg-black">
          {src ? (
            <img
              src={src}
              alt={`Frame ${frame.frame_id}`}
              className="max-w-full max-h-[72vh] object-contain"
            />
          ) : (
            <div className="text-slate-400 text-sm p-8">无法加载图像</div>
          )}
        </div>

        {/* Detection list */}
        {detections.length > 0 && (
          <div className="px-5 py-3 bg-slate-800/60 border-t border-slate-700 flex flex-wrap gap-2 max-h-36 overflow-y-auto">
            {detections.map((d, i) => (
              <span
                key={i}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs text-white/90"
              >
                <Target size={11} className="text-emerald-200" />
                <span className="truncate max-w-[12rem]">{d.label}</span>
                <span className="text-slate-400">{(d.score * 100).toFixed(0)}%</span>
                {d.track_id != null && (
                  <span className="text-emerald-200 font-mono">#{d.track_id}</span>
                )}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
