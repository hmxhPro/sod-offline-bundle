/**
 * src/components/DatasetMergeDialog.jsx
 * --------------------------------------
 * 数据集合并对话框 - 将源类别的数据合并到目标类别
 */

import React, { useState, useEffect } from 'react'
import { X, AlertTriangle, Merge } from 'lucide-react'
import { mergeDatasets } from '../services/api'
import { splitTimestampName } from '../utils/displayName'

export default function DatasetMergeDialog({ categories, onClose, onSuccess }) {
  const [targetId, setTargetId] = useState('')
  const [sourceId, setSourceId] = useState('')
  const [deleteSource, setDeleteSource] = useState(false)
  const [merging, setMerging] = useState(false)
  const [error, setError] = useState(null)

  // 预览统计
  const targetCat = categories.find((c) => c.id === targetId)
  const sourceCat = categories.find((c) => c.id === sourceId)

  const canMerge = targetId && sourceId && targetId !== sourceId && !merging

  const handleMerge = async () => {
    if (!canMerge) return

    setMerging(true)
    setError(null)

    try {
      const result = await mergeDatasets(targetId, sourceId, deleteSource)

      alert(
        `✓ 合并完成！\n\n` +
        `• 已合并 ${result.merged_images} 张图片\n` +
        `• 目标类别现有 ${result.target_total_images} 张图片\n` +
        `${result.deleted_source ? '• 已删除源类别' : '• 源类别保留'}`
      )

      onSuccess?.()
      onClose()
    } catch (err) {
      setError(err?.message || '合并失败')
    } finally {
      setMerging(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-surface border border-ink-200 rounded-xl shadow-2xl max-w-md w-full p-6 space-y-5">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-ink-900 flex items-center gap-2">
            <Merge size={20} className="text-brand-600" />
            合并数据集
          </h2>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-ink-100 text-ink-500"
            disabled={merging}
          >
            <X size={20} />
          </button>
        </div>

        {/* Form */}
        <div className="space-y-4">
          {/* Target */}
          <div>
            <label className="block text-sm font-medium text-ink-700 mb-1">
              目标类别（数据合并到此）
            </label>
            <select
              value={targetId}
              onChange={(e) => setTargetId(e.target.value)}
              disabled={merging}
              className="w-full px-3 py-2 border border-ink-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-500"
            >
              <option value="">请选择目标类别</option>
              {categories.map((cat) => (
                <option key={cat.id} value={cat.id}>
                  {splitTimestampName(cat.name).display} ({cat.image_count || 0} 张)
                </option>
              ))}
            </select>
          </div>

          {/* Source */}
          <div>
            <label className="block text-sm font-medium text-ink-700 mb-1">
              源类别（数据来源）
            </label>
            <select
              value={sourceId}
              onChange={(e) => setSourceId(e.target.value)}
              disabled={merging}
              className="w-full px-3 py-2 border border-ink-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-500"
            >
              <option value="">请选择源类别</option>
              {categories
                .filter((cat) => cat.id !== targetId)
                .map((cat) => (
                  <option key={cat.id} value={cat.id}>
                    {splitTimestampName(cat.name).display} ({cat.image_count || 0} 张)
                  </option>
                ))}
            </select>
          </div>

          {/* Delete source option */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={deleteSource}
              onChange={(e) => setDeleteSource(e.target.checked)}
              disabled={merging}
              className="w-4 h-4 text-brand-600 rounded focus:ring-brand-500"
            />
            <span className="text-sm text-ink-700">合并后删除源类别</span>
          </label>

          {/* Preview */}
          {targetCat && sourceCat && (
            <div className="bg-brand-50 border border-brand-200 rounded-lg p-3 space-y-1 text-sm">
              <div className="font-medium text-brand-900">预览：</div>
              <div className="text-brand-700">
                • 目标类别当前：{targetCat.image_count || 0} 张图片
              </div>
              <div className="text-brand-700">
                • 源类别：{sourceCat.image_count || 0} 张图片
              </div>
              <div className="text-brand-900 font-medium">
                • 合并后：{(targetCat.image_count || 0) + (sourceCat.image_count || 0)} 张图片
              </div>
            </div>
          )}

          {/* Warning */}
          {deleteSource && (
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 flex gap-2 text-sm">
              <AlertTriangle size={16} className="text-amber-600 flex-shrink-0 mt-0.5" />
              <span className="text-amber-800">
                删除源类别后将无法恢复，请确认操作。
              </span>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
              {error}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex gap-3 pt-2">
          <button
            onClick={onClose}
            disabled={merging}
            className="flex-1 px-4 py-2 border border-ink-300 text-ink-700 rounded-lg hover:bg-ink-50 disabled:opacity-50"
          >
            取消
          </button>
          <button
            onClick={handleMerge}
            disabled={!canMerge}
            className="flex-1 px-4 py-2 bg-brand-600 text-white rounded-lg hover:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {merging ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                合并中...
              </>
            ) : (
              <>
                <Merge size={16} />
                确认合并
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
