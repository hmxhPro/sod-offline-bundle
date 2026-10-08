/**
 * src/components/ImageUploader.jsx
 * --------------------------------
 * Drag-and-drop + click-to-browse area for MULTIPLE image files.
 * Emits an array of File objects via onFilesSelected. (Image sibling of
 * VideoUploader.)
 */

import React, { useState, useRef } from 'react'
import { ImagePlus, FolderPlus } from 'lucide-react'

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp']
const EXT_RE = /\.(jpe?g|png|bmp|webp)$/i

function filterValidImages(list) {
  return Array.from(list).filter((f) => ALLOWED_TYPES.includes(f.type) || EXT_RE.test(f.name))
}

export default function ImageUploader({ onFilesSelected, disabled, hasItems, hint }) {
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef(null)

  const emit = (files) => {
    const valid = filterValidImages(files)
    const rejected = files.length - valid.length
    if (rejected > 0) alert(`已忽略 ${rejected} 个非图片文件（仅支持 jpg / png / bmp / webp）`)
    if (valid.length) onFilesSelected(valid)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragActive(false)
    if (disabled) return
    if (e.dataTransfer.files?.length) emit(e.dataTransfer.files)
  }

  return (
    <div
      className={[
        'relative border-2 border-dashed rounded-2xl text-center cursor-pointer transition-all duration-200',
        hasItems ? 'py-6 px-5' : 'py-10 px-6',
        dragActive ? 'drop-zone-active' : 'border-ink-200 hover:border-brand-400 hover:bg-brand-50/40',
        disabled ? 'opacity-50 cursor-not-allowed' : '',
      ].join(' ')}
      onDrop={handleDrop}
      onDragOver={(e) => {
        e.preventDefault()
        if (!disabled) setDragActive(true)
      }}
      onDragLeave={() => setDragActive(false)}
      onClick={() => !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.length) emit(e.target.files)
          e.target.value = ''
        }}
        disabled={disabled}
      />
      <div className="flex flex-col items-center gap-3">
        <div className="p-3 rounded-2xl bg-brand-50 text-brand-500">
          {hasItems ? <FolderPlus size={26} /> : <ImagePlus size={30} />}
        </div>
        <div>
          <p className="text-ink-800 font-semibold">
            {hasItems ? '继续添加图片' : '拖拽图片到此处，或点击批量选择'}
          </p>
          <p className="text-ink-500 text-sm mt-1">
            {hint || '支持多图同时上传 · JPG / PNG / BMP / WebP'}
          </p>
        </div>
      </div>
    </div>
  )
}
