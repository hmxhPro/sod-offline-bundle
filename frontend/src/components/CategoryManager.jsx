/**
 * src/components/CategoryManager.jsx
 * ----------------------------------
 * Create / list / select a training category (REQ2). The selected category's
 * name becomes the trained model name.
 */

import React, { useCallback, useEffect, useState } from 'react'
import { Plus, Check, Trash2, RefreshCw, Tag, AlertCircle, Clock, Merge } from 'lucide-react'
import { getCategories, createCategory, deleteCategory } from '../services/api'
import { nameWithBadge, splitTimestampName } from '../utils/displayName'
import DatasetMergeDialog from './DatasetMergeDialog'

const STATUS_META = {
  draft: { label: '草稿', cls: 'bg-ink-100 text-ink-600' },
  annotating: { label: '标注中', cls: 'bg-amber-50 text-amber-700' },
  ready: { label: '待训练', cls: 'bg-brand-50 text-brand-700' },
  trained: { label: '已训练', cls: 'bg-emerald-50 text-emerald-700' },
}

export default function CategoryManager({ selectedId, onSelect, reloadToken }) {
  const [cats, setCats] = useState([])
  const [name, setName] = useState('')
  const [desc, setDesc] = useState('')
  const [loading, setLoading] = useState(false)
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState(null)
  const [showMergeDialog, setShowMergeDialog] = useState(false)

  const fetchCats = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setCats(await getCategories())
    } catch (e) {
      setError(e?.message || '加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchCats()
  }, [fetchCats, reloadToken])

  const handleCreate = async () => {
    const n = name.trim()
    if (!n) return
    setCreating(true)
    setError(null)
    try {
      const c = await createCategory(n, desc.trim() || undefined)
      setName('')
      setDesc('')
      await fetchCats()
      onSelect?.(c)
    } catch (e) {
      setError(e?.message || '创建失败')
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (e, c) => {
    e.stopPropagation()
    if (!window.confirm(`删除数据集「${splitTimestampName(c.name).display}」？\n将同时删除其图片、标注与模型记录（保留已训练权重文件）。`)) return
    try {
      await deleteCategory(c.id)
      if (selectedId === c.id) onSelect?.(null)
      fetchCats()
    } catch (err) {
      setError(err?.message || '删除失败')
    }
  }

  return (
    <div className="card p-5 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-ink-900 flex items-center gap-2">
          <span className="p-1.5 rounded-lg bg-brand-50 text-brand-500">
            <Tag size={14} />
          </span>
          训练类别
        </h3>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowMergeDialog(true)}
            disabled={loading || cats.length < 2}
            className="p-1.5 rounded-lg text-ink-500 hover:text-brand-600 hover:bg-ink-50 disabled:opacity-50"
            title="合并数据集"
          >
            <Merge size={14} />
          </button>
          <button
            type="button"
            onClick={fetchCats}
            disabled={loading}
            className="p-1.5 rounded-lg text-ink-500 hover:text-brand-600 hover:bg-ink-50 disabled:opacity-50"
            title="刷新"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Create */}
      <div className="flex flex-col gap-2">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
          placeholder="新类别名（如：安全帽、绝缘子）"
          maxLength={128}
          className="w-full px-3 py-2 rounded-xl border border-ink-200 focus:border-brand-400 focus:ring-2 focus:ring-brand-100 outline-none text-sm"
        />
        <div className="flex gap-2">
          <input
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
            placeholder="描述（可选）"
            className="flex-1 px-3 py-2 rounded-xl border border-ink-200 focus:border-brand-400 focus:ring-2 focus:ring-brand-100 outline-none text-sm"
          />
          <button
            type="button"
            onClick={handleCreate}
            disabled={creating || !name.trim()}
            className={[
              'px-3 py-2 rounded-xl font-medium text-sm flex items-center gap-1 transition-colors',
              creating || !name.trim()
                ? 'bg-ink-100 text-ink-400 cursor-not-allowed'
                : 'bg-brand-500 text-white hover:bg-brand-600',
            ].join(' ')}
          >
            <Plus size={15} /> 创建
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 p-2.5 rounded-lg border border-red-200 bg-red-50 text-red-700 text-xs">
          <AlertCircle size={14} /> {error}
        </div>
      )}

      {/* List */}
      <ul className="flex flex-col gap-1.5 max-h-72 overflow-y-auto">
        {cats.length === 0 && !loading && (
          <li className="text-ink-400 text-sm text-center py-4">暂无类别，先创建一个</li>
        )}
        {cats.map((c) => {
          const meta = STATUS_META[c.status] || STATUS_META.draft
          const active = selectedId === c.id
          const { display, badge } = nameWithBadge(c.name, c.created_at)
          return (
            <li
              key={c.id}
              role="button"
              tabIndex={0}
              onClick={() => onSelect?.(c)}
              onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onSelect?.(c)}
              className={[
                'group flex items-center gap-2 px-3 py-2.5 rounded-xl cursor-pointer border transition-colors',
                active ? 'border-brand-400 bg-brand-50/60' : 'border-transparent hover:bg-ink-50',
              ].join(' ')}
            >
              <span className={`w-5 ${active ? 'text-brand-500' : 'text-transparent'}`}>
                <Check size={16} />
              </span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-ink-800 text-sm truncate">{display}</span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${meta.cls}`}>
                    {meta.label}
                  </span>
                </div>
                <div className="text-[11px] text-ink-400 mt-0.5 flex items-center gap-1.5">
                  {badge && (
                    <span className="inline-flex items-center gap-0.5">
                      <Clock size={10} /> {badge}
                    </span>
                  )}
                  <span>{c.annotated_count}/{c.image_count} 已标注</span>
                </div>
              </div>
              <button
                type="button"
                onClick={(e) => handleDelete(e, c)}
                className="opacity-0 group-hover:opacity-100 p-1 rounded-lg text-ink-400 hover:text-red-600 hover:bg-red-50 transition-all"
                title="删除类别"
              >
                <Trash2 size={13} />
              </button>
            </li>
          )
        })}
      </ul>

      {/* Merge Dialog */}
      {showMergeDialog && (
        <DatasetMergeDialog
          categories={cats}
          onClose={() => setShowMergeDialog(false)}
          onSuccess={() => {
            fetchCats()
            setShowMergeDialog(false)
          }}
        />
      )}
    </div>
  )
}
