/**
 * src/components/DatasetImporter.jsx
 * ----------------------------------
 * Second dataset source for the training workspace (REQ2): instead of drawing
 * boxes online, the user picks a LOCAL FOLDER of an already-annotated YOLO
 * dataset (images + same-stem .txt labels). The whole folder is uploaded and
 * the backend pairs images↔labels, folds every box to the category's single
 * class, and registers them as `annotated` — landing in the exact same
 * representation the online annotator produces, so training works unchanged.
 *
 * Folder selection uses the non-standard `webkitdirectory` attribute, declared
 * directly on the <input> (React passes lowercase unknown attributes through).
 * It MUST be present before the element is inserted into the DOM: some browsers
 * lock the picker into single-file mode at insertion time and ignore the
 * attribute if it is added afterwards (e.g. via a post-mount effect), which
 * silently disables whole-folder selection.
 */

import React, { useMemo, useRef, useState } from 'react'
import {
  Database, FolderPlus, Loader2, CheckCircle2, AlertCircle, ImagePlus, Tag, Check,
} from 'lucide-react'
import { importCategoryDatasetBatched } from '../services/api'

const IMG_RE = /\.(jpe?g|png|bmp|webp)$/i
const TXT_RE = /\.txt$/i

export default function DatasetImporter({ category, onDataChanged }) {
  const [picked, setPicked] = useState(null) // {files, relPaths, imgN, txtN, total}
  const [uploading, setUploading] = useState(false)
  const [pct, setPct] = useState(0)
  const [batch, setBatch] = useState(null) // {index, count} when uploading in batches
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const inputRef = useRef(null)

  const handlePick = (fileList) => {
    const arr = Array.from(fileList || [])
    const files = []
    const relPaths = []
    let imgN = 0
    let txtN = 0
    for (const f of arr) {
      const rel = f.webkitRelativePath || f.name
      if (IMG_RE.test(rel)) {
        files.push(f); relPaths.push(rel); imgN++
      } else if (TXT_RE.test(rel)) {
        files.push(f); relPaths.push(rel); txtN++
      }
    }
    setError(null)
    setResult(null)
    if (imgN === 0) {
      setPicked(null)
      setError('所选文件夹内没有发现图片（支持 jpg / png / bmp / webp）。')
      return
    }
    setPicked({ files, relPaths, imgN, txtN, total: arr.length })
  }

  const handleImport = async () => {
    if (!category || !picked?.files.length || uploading) return
    setUploading(true)
    setError(null)
    setResult(null)
    setPct(0)
    setBatch(null)
    try {
      const res = await importCategoryDatasetBatched(category.id, picked.files, picked.relPaths, {
        onProgress: (p, info) => { setPct(p); if (info) setBatch(info) },
        onBatch: setBatch,
      })
      setResult(res)
      setPicked(null)
      onDataChanged?.()
    } catch (e) {
      // A batch can fail mid-way; earlier batches are already committed.
      if (e?.partial?.imported_images) {
        setError(
          `已成功导入 ${e.partial.imported_images} 张图片，但第 ${e.partial.failedBatch}/${e.partial.totalBatches} 批上传失败：` +
          `${e.message}。可重新选择该文件夹导入剩余部分。`
        )
        onDataChanged?.()
      } else {
        setError(e?.message || '导入失败')
      }
    } finally {
      setUploading(false)
      setPct(0)
      setBatch(null)
    }
  }

  const rootName = useMemo(() => {
    const rel = picked?.files?.[0]?.webkitRelativePath
    return rel ? rel.split('/')[0] : null
  }, [picked])

  if (!category) {
    return (
      <div className="card p-10 text-center text-ink-500">
        请先在左侧选择或创建一个类别。
      </div>
    )
  }

  return (
    <div className="grid lg:grid-cols-[1fr_18rem] gap-4">
      {/* ── Left: pick + status ─────────────────────────────────────────── */}
      <div className="card p-5 flex flex-col gap-4 min-w-0">
        <h3 className="font-semibold text-ink-900 flex items-center gap-2">
          <span className="p-1.5 rounded-lg bg-brand-50 text-brand-500">
            <Database size={14} />
          </span>
          上传已标注数据集
        </h3>

        {/* Pick area */}
        <div
          role="button"
          tabIndex={0}
          onClick={() => !uploading && inputRef.current?.click()}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !uploading && inputRef.current?.click()}
          className={[
            'border-2 border-dashed rounded-2xl text-center cursor-pointer transition-all duration-200 py-10 px-6',
            uploading ? 'opacity-50 cursor-not-allowed border-ink-200' : 'border-ink-200 hover:border-brand-400 hover:bg-brand-50/40',
          ].join(' ')}
        >
          <input
            ref={inputRef}
            type="file"
            multiple
            webkitdirectory=""
            directory=""
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.length) handlePick(e.target.files)
              e.target.value = ''
            }}
            disabled={uploading}
          />
          <div className="flex flex-col items-center gap-3">
            <div className="p-3 rounded-2xl bg-brand-50 text-brand-500">
              <FolderPlus size={30} />
            </div>
            <div>
              <p className="text-ink-800 font-semibold">选择本地数据集文件夹</p>
              <p className="text-ink-500 text-sm mt-1">
                选中整个 YOLO 数据集文件夹，图片与同名 .txt 标注将一起上传
              </p>
            </div>
          </div>
        </div>

        {/* Picked preview */}
        {picked && (
          <div className="flex flex-col gap-3 p-3 rounded-xl border border-ink-200 bg-ink-50/60">
            <div className="text-sm text-ink-700 flex flex-wrap items-center gap-x-4 gap-y-1">
              {rootName && (
                <span className="font-medium text-ink-900 truncate">📁 {rootName}</span>
              )}
              <span className="flex items-center gap-1"><ImagePlus size={14} className="text-brand-500" /> {picked.imgN} 张图片</span>
              <span className="flex items-center gap-1"><Tag size={14} className="text-emerald-500" /> {picked.txtN} 个标注文件</span>
            </div>
            <button
              type="button"
              onClick={handleImport}
              disabled={uploading}
              className={[
                'w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl font-semibold transition-all',
                uploading ? 'bg-ink-100 text-ink-400 cursor-not-allowed' : 'bg-brand-500 text-white hover:bg-brand-600 shadow-brand',
              ].join(' ')}
            >
              {uploading ? <Loader2 size={16} className="animate-spin" /> : <Database size={16} />}
              {uploading
                ? `导入中… ${batch && batch.count > 1 ? `第 ${batch.index}/${batch.count} 批 · ` : ''}${pct}%`
                : `导入到类别「${category.name}」`}
            </button>
          </div>
        )}

        {uploading && (
          <div className="h-1.5 bg-ink-100 rounded-full overflow-hidden">
            <div className="h-full bg-brand-500 rounded-full transition-all duration-300" style={{ width: `${pct}%` }} />
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 p-2.5 rounded-lg border border-red-200 bg-red-50 text-red-700 text-xs">
            <AlertCircle size={14} /> {error}
          </div>
        )}

        {result && (
          <div className="flex flex-col gap-1 p-3 rounded-lg border border-emerald-200 bg-emerald-50 text-emerald-800 text-sm">
            <span className="flex items-center gap-2 font-medium">
              <CheckCircle2 size={16} /> 导入完成
            </span>
            <span className="text-emerald-700 text-xs">{result.message}</span>
            <span className="text-emerald-700 text-xs">
              可在「在线标注」标签查看，或在左侧直接「开始训练」。
            </span>
          </div>
        )}
      </div>

      {/* ── Right: format help ──────────────────────────────────────────── */}
      <aside className="card p-4 flex flex-col gap-3 h-fit">
        <h4 className="font-semibold text-ink-800 text-sm flex items-center gap-2">
          <Database size={14} className="text-brand-500" /> 数据集格式
        </h4>
        <p className="text-xs text-ink-600">支持标准 YOLO 目录结构：</p>
        <pre className="text-[11px] leading-relaxed text-ink-600 bg-ink-50 rounded-lg p-2.5 overflow-x-auto">{`dataset/
├─ images/
│   ├─ a.jpg
│   └─ b.jpg
└─ labels/
    ├─ a.txt   (c cx cy w h)
    └─ b.txt`}</pre>
        <ul className="text-xs text-ink-600 space-y-1.5 list-disc list-inside">
          <li>图片与标注按<strong>同名</strong>配对（<code>a.jpg</code> ↔ <code>a.txt</code>）</li>
          <li>标注为 <strong>YOLO 归一化</strong>格式（坐标 0~1）</li>
          <li>无同名标注的图片视为<strong>背景样本</strong></li>
          <li><code>train/</code>、<code>val/</code> 等子目录可正常识别</li>
          <li>大文件夹会<strong>自动分批上传</strong>，无需手动拆分</li>
        </ul>
        <p className="text-[11px] text-ink-400">
          注：所有标注框都会归到当前类别
          <span className="text-ink-700 font-medium">「{category.name}」</span>
          （单类别）。多类别数据集将被合并为该单一类别。
        </p>
      </aside>
    </div>
  )
}
