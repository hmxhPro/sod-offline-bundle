/**
 * src/components/VideoUploader.jsx
 * ----------------------------------
 * Drag-and-drop + click-to-browse area for MULTIPLE video files.
 *
 * Emits an array of File objects via onFilesSelected.
 */

import React, { useState, useRef } from 'react'
import { UploadCloud, FolderPlus } from 'lucide-react'

const ALLOWED_TYPES = [
  'video/mp4', 'video/avi', 'video/quicktime',
  'video/x-matroska', 'video/webm', 'video/x-flv',
]
const EXT_RE = /\.(mp4|avi|mov|mkv|webm|flv)$/i

function filterValidVideos(list) {
  return Array.from(list).filter((f) =>
    ALLOWED_TYPES.includes(f.type) || EXT_RE.test(f.name)
  )
}

export default function VideoUploader({ onFilesSelected, disabled, hasTasks }) {
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef(null)

  const emit = (files) => {
    const valid = filterValidVideos(files)
    const rejected = files.length - valid.length
    if (rejected > 0) {
      alert(`已忽略 ${rejected} 个非视频文件（仅支持 mp4 / avi / mov / mkv / webm / flv）`)
    }
    if (valid.length) onFilesSelected(valid)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragActive(false)
    if (disabled) return
    if (e.dataTransfer.files?.length) emit(e.dataTransfer.files)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    if (!disabled) setDragActive(true)
  }

  const handleDragLeave = () => setDragActive(false)

  const handleClick = () => {
    if (!disabled) inputRef.current?.click()
  }

  const handleInputChange = (e) => {
    if (e.target.files?.length) emit(e.target.files)
    e.target.value = '' // allow selecting same file again
  }

  return (
    <div
      className={[
        'relative border-2 border-dashed rounded-2xl text-center cursor-pointer transition-all duration-200',
        hasTasks ? 'py-6 px-5' : 'py-10 px-6',
        dragActive
          ? 'drop-zone-active'
          : 'border-ink-200 hover:border-brand-400 hover:bg-brand-50/40',
        disabled ? 'opacity-50 cursor-not-allowed' : '',
      ].join(' ')}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={handleClick}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*,.mp4,.avi,.mov,.mkv,.webm,.flv,.m4v,.wmv,.mpg,.mpeg,.ts"
        multiple
        className="hidden"
        onChange={handleInputChange}
        disabled={disabled}
      />

      <div className="flex flex-col items-center gap-3">
        <div className="p-3 rounded-2xl bg-brand-50 text-brand-500">
          {hasTasks ? <FolderPlus size={26} /> : <UploadCloud size={30} />}
        </div>
        <div>
          <p className="text-ink-800 font-semibold">
            {hasTasks ? '继续添加视频' : '拖拽视频到此处，或点击批量选择'}
          </p>
          <p className="text-ink-500 text-sm mt-1">
            支持 <span className="text-ink-700 font-medium">多视频同时上传</span> · MP4 / AVI / MOV / MKV / WebM / FLV · 不限制单文件大小
          </p>
        </div>
      </div>
    </div>
  )
}
