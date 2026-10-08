/**
 * src/components/KonvaAnnotator.jsx
 * --------------------------------
 * react-konva YOLO bounding-box annotation workspace (REQ2).
 *
 * Given a category, this component lets the user:
 *   - upload images into the category dataset,
 *   - draw / move / resize / delete boxes on each image,
 *   - save annotations (persisted as YOLO-normalized cx/cy/w/h on the backend).
 *
 * Phase 1 is single-class: every box is class 0 (the category itself). An
 * example + instructions are shown alongside the canvas.
 *
 * Boxes are stored in NORMALIZED form ({cx,cy,w,h} in [0,1]) — the source of
 * truth — and converted to stage pixels only for rendering/interaction, so a
 * window resize never corrupts coordinates.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Stage, Layer, Rect, Transformer, Image as KImage } from 'react-konva'
import {
  ChevronLeft, ChevronRight, Save, Trash2, Loader2, ImageOff, Check, MousePointer2,
} from 'lucide-react'
import ImageUploader from './ImageUploader'
import {
  getCategoryImages, getImageFileUrl, getImageAnnotation, saveImageAnnotation,
  uploadCategoryImages, deleteCategoryImage,
} from '../services/api'

let _bid = 1
const makeBoxId = () => `b_${_bid++}`
const clamp01 = (v) => Math.max(0, Math.min(1, v))
const isTrainableImage = (im) => im?.annotation_status === 'annotated' && (im?.box_count || 0) > 0

export default function KonvaAnnotator({ category, onDataChanged, initialTrainableOnly = false }) {
  const [images, setImages] = useState([])
  const [trainableOnly, setTrainableOnly] = useState(initialTrainableOnly)
  const [idx, setIdx] = useState(0)
  const [imgEl, setImgEl] = useState(null)
  const [natural, setNatural] = useState({ w: 0, h: 0 })
  const [containerW, setContainerW] = useState(640)
  const [boxes, setBoxes] = useState([]) // normalized {id,cls,cx,cy,w,h}
  const [selectedId, setSelectedId] = useState(null)
  const [draft, setDraft] = useState(null) // in-progress rect (stage px) {x,y,w,h}
  const [saving, setSaving] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadPct, setUploadPct] = useState(0)
  const [error, setError] = useState(null)
  const [allCount, setAllCount] = useState(0)
  const [trainableCount, setTrainableCount] = useState(0)

  const containerRef = useRef(null)
  const trRef = useRef(null)
  const rectRefs = useRef(new Map())
  const initedRef = useRef(null)
  const drawingRef = useRef(false)
  const dirtyRef = useRef(false)

  const current = images[idx] || null
  const scale = natural.w ? containerW / natural.w : 1
  const stageW = natural.w ? containerW : 0
  const stageH = natural.h ? Math.round(natural.h * scale) : 0
  const ready = stageW > 0 && stageH > 0 && imgEl

  // ── measure container width ───────────────────────────────────────────
  useEffect(() => {
    const measure = () => {
      if (containerRef.current) {
        setContainerW(Math.max(320, Math.floor(containerRef.current.clientWidth)))
      }
    }
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [])

  // ── fetch image list for the category ─────────────────────────────────
  const fetchImages = useCallback(async () => {
    if (!category) {
      setImages([])
      setAllCount(0)
      setTrainableCount(0)
      return
    }
    setError(null)
    try {
      const rows = await getCategoryImages(category.id)
      const trainable = rows.filter(isTrainableImage)
      setAllCount(rows.length)
      setTrainableCount(trainable.length)
      const visibleRows = trainableOnly ? trainable : rows
      setImages(visibleRows)
      setIdx((i) => Math.min(i, Math.max(0, visibleRows.length - 1)))
    } catch (e) {
      setError(e?.message || '加载图片失败')
    }
    // 仅依赖类别 id（而非整个 category 对象）：保存后父级会用 getCategory 刷新出
    // 一个「同 id、新引用」的 category，若依赖整个对象会让本回调换新身份，从而触发
    // 下方 [fetchImages] 的 effect 把索引重置到第 0 张。只在真正切换类别时才重建。
  }, [category?.id, trainableOnly]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setIdx(0)
    fetchImages()
  }, [fetchImages])

  // ── load the current image element ────────────────────────────────────
  useEffect(() => {
    setSelectedId(null)
    setDraft(null)
    initedRef.current = null
    dirtyRef.current = false
    if (!current || !category) {
      setImgEl(null)
      setNatural({ w: 0, h: 0 })
      setBoxes([])
      return
    }
    const im = new window.Image()
    im.src = getImageFileUrl(category.id, current.id)
    im.onload = () => {
      setImgEl(im)
      setNatural({ w: im.naturalWidth, h: im.naturalHeight })
    }
    im.onerror = () => setError('图片加载失败')
    return () => {
      im.onload = null
      im.onerror = null
    }
  }, [current?.id, category?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── initialize boxes from saved annotation once image dims known ──────
  useEffect(() => {
    if (!current || !category || !natural.w || initedRef.current === current.id) return
    let cancelled = false
    ;(async () => {
      try {
        const ann = await getImageAnnotation(category.id, current.id)
        if (cancelled) return
        setBoxes(
          (ann.boxes || []).map((b) => ({
            id: makeBoxId(), cls: b.cls || 0, cx: b.cx, cy: b.cy, w: b.w, h: b.h,
          }))
        )
        initedRef.current = current.id
        dirtyRef.current = false
      } catch (e) {
        if (!cancelled) setError(e?.message || '读取标注失败')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [current?.id, natural.w, category?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── attach transformer to the selected rect ───────────────────────────
  useEffect(() => {
    const tr = trRef.current
    if (!tr) return
    const node = selectedId && rectRefs.current.get(selectedId)
    tr.nodes(node ? [node] : [])
    tr.getLayer()?.batchDraw()
  }, [selectedId, boxes, stageW, stageH])

  // ── delete selected box on Delete/Backspace ───────────────────────────
  useEffect(() => {
    const onKey = (e) => {
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId) {
        e.preventDefault()
        removeBox(selectedId)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selectedId]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── box helpers (normalized <-> stage px) ─────────────────────────────
  const toRect = (b) => ({
    x: (b.cx - b.w / 2) * stageW,
    y: (b.cy - b.h / 2) * stageH,
    width: b.w * stageW,
    height: b.h * stageH,
  })

  const markDirty = () => {
    dirtyRef.current = true
  }

  const removeBox = (id) => {
    setBoxes((prev) => prev.filter((b) => b.id !== id))
    rectRefs.current.delete(id)
    if (selectedId === id) setSelectedId(null)
    markDirty()
  }

  const updateBoxFromDrag = (id, node) => {
    const b = boxes.find((x) => x.id === id)
    if (!b) return
    const width = b.w * stageW
    const height = b.h * stageH
    let x = Math.max(0, Math.min(node.x(), stageW - width))
    let y = Math.max(0, Math.min(node.y(), stageH - height))
    node.position({ x, y })
    setBoxes((prev) =>
      prev.map((p) =>
        p.id === id ? { ...p, cx: clamp01((x + width / 2) / stageW), cy: clamp01((y + height / 2) / stageH) } : p
      )
    )
    markDirty()
  }

  const bakeTransform = (id, node) => {
    let width = Math.max(4, node.width() * node.scaleX())
    let height = Math.max(4, node.height() * node.scaleY())
    node.scaleX(1)
    node.scaleY(1)
    let x = Math.max(0, Math.min(node.x(), stageW - width))
    let y = Math.max(0, Math.min(node.y(), stageH - height))
    width = Math.min(width, stageW - x)
    height = Math.min(height, stageH - y)
    node.position({ x, y })
    node.width(width)
    node.height(height)
    setBoxes((prev) =>
      prev.map((p) =>
        p.id === id
          ? {
              ...p,
              cx: clamp01((x + width / 2) / stageW),
              cy: clamp01((y + height / 2) / stageH),
              w: clamp01(width / stageW),
              h: clamp01(height / stageH),
            }
          : p
      )
    )
    markDirty()
  }

  // ── stage drawing (new box) ───────────────────────────────────────────
  const pointer = (stage) => {
    const p = stage.getPointerPosition()
    return { x: Math.max(0, Math.min(p.x, stageW)), y: Math.max(0, Math.min(p.y, stageH)) }
  }

  const onStageMouseDown = (e) => {
    const cls = e.target.getClassName?.()
    // Only start drawing on the background image or empty stage; boxes/handles
    // are excluded so clicks there select/resize instead.
    if (e.target !== e.target.getStage() && cls !== 'Image') return
    setSelectedId(null)
    const { x, y } = pointer(e.target.getStage())
    drawingRef.current = true
    setDraft({ x, y, w: 0, h: 0 })
  }

  const onStageMouseMove = (e) => {
    if (!drawingRef.current) return
    const { x, y } = pointer(e.target.getStage())
    setDraft((d) => (d ? { ...d, w: x - d.x, h: y - d.y } : d))
  }

  const onStageMouseUp = () => {
    if (!drawingRef.current) return
    drawingRef.current = false
    setDraft((d) => {
      if (d) {
        const x = Math.min(d.x, d.x + d.w)
        const y = Math.min(d.y, d.y + d.h)
        const w = Math.abs(d.w)
        const h = Math.abs(d.h)
        if (w > 4 && h > 4) {
          const nb = {
            id: makeBoxId(),
            cls: 0,
            cx: clamp01((x + w / 2) / stageW),
            cy: clamp01((y + h / 2) / stageH),
            w: clamp01(w / stageW),
            h: clamp01(h / stageH),
          }
          setBoxes((prev) => [...prev, nb])
          setSelectedId(nb.id)
          markDirty()
        }
      }
      return null
    })
  }

  // ── actions ───────────────────────────────────────────────────────────
  const handleSave = async () => {
    if (!current || !category) return true   // nothing to save → don't block nav
    setSaving(true)
    setError(null)
    try {
      const payload = boxes.map((b) => ({
        cls: b.cls || 0, cx: clamp01(b.cx), cy: clamp01(b.cy), w: clamp01(b.w), h: clamp01(b.h),
      }))
      const updated = await saveImageAnnotation(category.id, current.id, payload)
      // 保存当前图片ID和索引位置，以便重新加载后恢复
      const currentImageId = current.id
      const currentIdx = idx
      // 重新获取图片列表以保持与服务器同步
      const rows = await getCategoryImages(category.id)
      const trainable = rows.filter(isTrainableImage)
      setAllCount(rows.length)
      setTrainableCount(trainable.length)
      const visibleRows = trainableOnly ? trainable : rows
      setImages(visibleRows)
      // 找到当前图片在新列表中的位置
      const newIdx = visibleRows.findIndex(im => im.id === currentImageId)
      if (newIdx !== -1) {
        // 找到了当前图片，保持在当前位置
        setIdx(newIdx)
      } else {
        // 如果当前图片不在可见列表中（例如在trainableOnly模式下被过滤掉）
        // 保持在相同的索引位置（如果有效），否则调整到最接近的有效位置
        setIdx(Math.min(currentIdx, Math.max(0, visibleRows.length - 1)))
      }
      dirtyRef.current = false
      onDataChanged?.()
      return true
    } catch (e) {
      // F-3: surface the failure AND signal it to the caller so navigation is
      // blocked — silently swallowing this lost unsaved annotations on page flip.
      setError(`保存失败：${e?.message || '未知错误'}，未翻页`)
      return false
    } finally {
      setSaving(false)
    }
  }

  const goto = async (newIdx) => {
    if (newIdx < 0 || newIdx >= images.length || newIdx === idx) return
    if (dirtyRef.current) {
      const ok = await handleSave()
      if (!ok) return   // save failed → stay on this image, keep unsaved boxes
    }
    setIdx(newIdx)
  }

  const handleUpload = async (files) => {
    if (!category) return
    setUploading(true)
    setUploadPct(0)
    setError(null)
    try {
      await uploadCategoryImages(category.id, files, setUploadPct)
      await fetchImages()
      onDataChanged?.()
    } catch (e) {
      setError(e?.message || '上传失败')
    } finally {
      setUploading(false)
      setUploadPct(0)
    }
  }

  const handleDeleteImage = async () => {
    if (!current || !category) return
    if (!window.confirm('删除当前图片及其标注？')) return
    try {
      await deleteCategoryImage(category.id, current.id)
      await fetchImages()
      onDataChanged?.()
    } catch (e) {
      setError(e?.message || '删除失败')
    }
  }

  if (!category) {
    return (
      <div className="card p-10 text-center text-ink-500">
        请先在左侧选择或创建一个类别。
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <ImageUploader
        onFilesSelected={handleUpload}
        disabled={uploading}
        hasItems={images.length > 0}
        hint={`图片将存入类别「${category.name}」的数据集`}
      />
      {uploading && (
        <div className="text-xs text-brand-600">上传中… {uploadPct}%</div>
      )}
      {allCount > 0 && (
        <div className="flex flex-wrap items-center gap-2 text-xs text-ink-500">
          <span>
            当前显示 {images.length} 张；可训练标注图 {trainableCount} 张 / 总图片 {allCount} 张
          </span>
          <button
            type="button"
            onClick={() => setTrainableOnly((v) => !v)}
            className="px-2 py-1 rounded-lg border border-ink-200 text-ink-600 hover:bg-ink-50"
          >
            {trainableOnly ? '显示全部图片用于修正' : '仅显示可训练标注图'}
          </button>
        </div>
      )}
      {error && (
        <div className="p-2.5 rounded-lg border border-red-200 bg-red-50 text-red-700 text-xs">⚠ {error}</div>
      )}

      <div className="grid lg:grid-cols-[1fr_18rem] gap-4">
        {/* Canvas + controls */}
        <div className="card p-4 flex flex-col gap-3 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => goto(idx - 1)}
                disabled={idx <= 0}
                className="p-1.5 rounded-lg border border-ink-200 text-ink-600 hover:bg-ink-50 disabled:opacity-40"
              >
                <ChevronLeft size={16} />
              </button>
              <span className="text-sm text-ink-600 font-mono">
                {images.length ? idx + 1 : 0} / {images.length}
              </span>
              <button
                type="button"
                onClick={() => goto(idx + 1)}
                disabled={idx >= images.length - 1}
                className="p-1.5 rounded-lg border border-ink-200 text-ink-600 hover:bg-ink-50 disabled:opacity-40"
              >
                <ChevronRight size={16} />
              </button>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-ink-400">
                {boxes.length} 个框 {dirtyRef.current ? '· 未保存' : ''}
              </span>
              <button
                type="button"
                onClick={() => selectedId && removeBox(selectedId)}
                disabled={!selectedId}
                className="px-2.5 py-1.5 rounded-lg border border-ink-200 text-ink-600 hover:bg-ink-50 disabled:opacity-40 text-xs flex items-center gap-1"
              >
                <Trash2 size={13} /> 删框
              </button>
              <button
                type="button"
                onClick={handleDeleteImage}
                disabled={!current}
                className="px-2.5 py-1.5 rounded-lg border border-ink-200 text-ink-500 hover:bg-red-50 hover:text-red-600 disabled:opacity-40 text-xs"
              >
                删图
              </button>
              <button
                type="button"
                onClick={handleSave}
                disabled={!current || saving}
                className={[
                  'px-3 py-1.5 rounded-lg font-medium text-xs flex items-center gap-1',
                  !current || saving ? 'bg-ink-100 text-ink-400' : 'bg-brand-500 text-white hover:bg-brand-600',
                ].join(' ')}
              >
                {saving ? <Loader2 size={13} className="animate-spin" /> : <Save size={13} />}
                保存标注
              </button>
            </div>
          </div>

          <div ref={containerRef} className="w-full bg-ink-50 rounded-xl overflow-hidden">
            {images.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-ink-400">
                <ImageOff size={32} className="mb-2" />
                <span className="text-sm">
                  {trainableOnly && allCount > 0
                    ? '当前筛选下没有可训练标注图，可切换为全部图片后检查/修正'
                    : '该类别还没有图片，请在上方上传'}
                </span>
              </div>
            ) : !ready ? (
              <div className="flex items-center justify-center py-16 text-ink-400">
                <Loader2 size={20} className="animate-spin mr-2" /> 加载图片中…
              </div>
            ) : (
              <Stage
                width={stageW}
                height={stageH}
                onMouseDown={onStageMouseDown}
                onMouseMove={onStageMouseMove}
                onMouseUp={onStageMouseUp}
                style={{ cursor: 'crosshair' }}
              >
                <Layer>
                  <KImage image={imgEl} width={stageW} height={stageH} />
                  {boxes.map((b) => {
                    const r = toRect(b)
                    return (
                      <Rect
                        key={b.id}
                        ref={(node) => {
                          if (node) rectRefs.current.set(b.id, node)
                          else rectRefs.current.delete(b.id)
                        }}
                        x={r.x}
                        y={r.y}
                        width={r.width}
                        height={r.height}
                        stroke={selectedId === b.id ? '#f97316' : '#06b6d4'}
                        strokeWidth={2}
                        fill={selectedId === b.id ? 'rgba(249,115,22,0.12)' : 'rgba(6,182,212,0.10)'}
                        draggable
                        onMouseDown={(e) => {
                          e.cancelBubble = true
                          setSelectedId(b.id)
                        }}
                        onClick={() => setSelectedId(b.id)}
                        onTap={() => setSelectedId(b.id)}
                        onDragEnd={(e) => updateBoxFromDrag(b.id, e.target)}
                        onTransformEnd={(e) => bakeTransform(b.id, e.target)}
                      />
                    )
                  })}
                  {draft && (
                    <Rect
                      x={Math.min(draft.x, draft.x + draft.w)}
                      y={Math.min(draft.y, draft.y + draft.h)}
                      width={Math.abs(draft.w)}
                      height={Math.abs(draft.h)}
                      stroke="#f97316"
                      dash={[6, 4]}
                      strokeWidth={2}
                    />
                  )}
                  <Transformer
                    ref={trRef}
                    rotateEnabled={false}
                    keepRatio={false}
                    ignoreStroke
                    boundBoxFunc={(oldB, newB) => (newB.width < 6 || newB.height < 6 ? oldB : newB)}
                  />
                </Layer>
              </Stage>
            )}
          </div>

          {/* Filmstrip */}
          {images.length > 0 && (
            <div className="flex gap-2 overflow-x-auto pb-1">
              {images.map((im, i) => (
                <button
                  key={im.id}
                  type="button"
                  onClick={() => goto(i)}
                  title={im.filename}
                  className={[
                    'relative flex-shrink-0 w-16 h-16 rounded-lg overflow-hidden border-2 transition-colors',
                    i === idx ? 'border-brand-500' : 'border-transparent hover:border-ink-300',
                  ].join(' ')}
                >
                  <img
                    src={getImageFileUrl(category.id, im.id)}
                    alt={im.filename}
                    className="w-full h-full object-cover"
                  />
                  {im.annotation_status === 'annotated' && (
                    <span className="absolute top-0.5 right-0.5 bg-emerald-500 text-white rounded-full p-0.5">
                      <Check size={10} />
                    </span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Example + instructions */}
        <aside className="card p-4 flex flex-col gap-3 h-fit">
          <h4 className="font-semibold text-ink-800 text-sm flex items-center gap-2">
            <MousePointer2 size={14} className="text-brand-500" /> 标注示例
          </h4>
          <svg viewBox="0 0 200 130" className="w-full rounded-lg bg-ink-50">
            <rect x="0" y="0" width="200" height="130" fill="#eef2f7" />
            <ellipse cx="100" cy="95" rx="70" ry="18" fill="#dbe3ee" />
            <circle cx="100" cy="60" r="34" fill="#c2cede" />
            <circle cx="100" cy="60" r="20" fill="#aebccf" />
            <rect x="58" y="22" width="84" height="78" fill="none" stroke="#f97316" strokeWidth="3" />
            <rect x="58" y="12" width="60" height="14" fill="#f97316" />
            <text x="62" y="23" fontSize="11" fill="#fff" fontFamily="sans-serif">
              {category.name}
            </text>
          </svg>
          <ol className="text-xs text-ink-600 space-y-1.5 list-decimal list-inside">
            <li>在图片上<strong>按住拖拽</strong>画出目标的矩形框</li>
            <li>点击框可<strong>选中</strong>，拖动框身移动，拖角点缩放</li>
            <li>选中后按 <kbd className="px-1 bg-ink-100 rounded">Delete</kbd> 或「删框」删除</li>
            <li>框要<strong>紧贴</strong>目标边缘，尽量准确</li>
            <li>点「<strong>保存标注</strong>」后切换下一张（切换会自动保存）</li>
            <li>无目标的图片可不画框直接保存为<strong>未检测到目标</strong>，不会进入训练</li>
          </ol>
          <p className="text-[11px] text-ink-400">
            当前类别：<span className="text-ink-700 font-medium">{category.name}</span>（单类别，所有框均属于该类别）
          </p>
        </aside>
      </div>
    </div>
  )
}
