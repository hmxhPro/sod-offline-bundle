/**
 * src/components/DetectModelSelect.jsx
 * ------------------------------------
 * Trained-model picker for the video detection workspace.
 *
 * Selecting a model makes the backend run detection with that model's baked-in
 * classes (no natural-language prompt). The default option keeps the original
 * open-vocabulary behavior, so this is always optional.
 */

import React, { useCallback, useEffect, useState } from 'react'
import { Boxes, RefreshCw } from 'lucide-react'
import { getModels } from '../services/api'

function classText(m) {
  if (!m?.class_names) return ''
  const vals = Object.values(m.class_names)
  return vals.length ? vals.join('、') : ''
}

function optionLabel(m) {
  const ver = m.version > 1 ? ` v${m.version}` : ''
  const cls = classText(m)
  return cls ? `${m.name}${ver}（${cls}）` : `${m.name}${ver}`
}

export default function DetectModelSelect({ value, onChange, disabled, reloadToken }) {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchModels = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setModels(await getModels())
    } catch (e) {
      // Keep the default option usable even when the list fails to load.
      setError(e?.message || '加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchModels()
  }, [fetchModels, reloadToken])

  // If the currently selected model disappears (deleted / list reloaded),
  // fall back to open-vocabulary so we never submit a stale model_id.
  useEffect(() => {
    if (value && models.length && !models.some((m) => m.id === value)) {
      onChange('', null)
    }
  }, [models, value, onChange])

  const handleChange = (e) => {
    const id = e.target.value
    onChange(id, models.find((m) => m.id === id) || null)
  }

  const selected = models.find((m) => m.id === value) || null

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <label className="text-ink-800 font-semibold flex items-center gap-2">
          <span className="p-1.5 rounded-lg bg-brand-50 text-brand-500">
            <Boxes size={14} />
          </span>
          检测模型（可选）
        </label>
        <button
          type="button"
          onClick={fetchModels}
          disabled={loading || disabled}
          className="p-1.5 rounded-lg text-ink-500 hover:text-brand-600 hover:bg-ink-50 disabled:opacity-50"
          title="刷新模型列表"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      <select
        value={value || ''}
        onChange={handleChange}
        disabled={disabled}
        className={[
          'w-full rounded-xl px-4 py-2.5 text-ink-800',
          'bg-surface border border-ink-200',
          'focus:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-100',
          'transition-colors duration-200',
          disabled ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer',
        ].join(' ')}
      >
        <option value="">不使用（自然语言检测）</option>
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {optionLabel(m)}
          </option>
        ))}
      </select>

      {error && (
        <p className="text-xs text-amber-600">⚠ 模型列表加载失败（{error}），可继续使用自然语言检测。</p>
      )}

      {selected ? (
        <p className="text-xs text-ink-500 leading-relaxed">
          将使用模型<span className="text-ink-700 font-medium">「{selected.name}」</span>自带类别检测
          {classText(selected) && <>：<span className="text-ink-700 font-medium">{classText(selected)}</span></>}
          ，无需填写检测目标。
        </p>
      ) : (
        <p className="text-xs text-ink-500 leading-relaxed">
          选择一个已训练模型即可用其自带类别检测；留空则使用下方的自然语言开放词汇检测。
        </p>
      )}
    </div>
  )
}
