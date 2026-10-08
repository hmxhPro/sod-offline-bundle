/**
 * src/components/ModelListPanel.jsx
 * ---------------------------------
 * Trained-model list (REQ3): newest first, hover a row for detail
 * (training time, dataset, metrics, classes).
 */

import React, { useCallback, useEffect, useState } from 'react'
import { Boxes, RefreshCw, Clock, Database, Tag, Trash2, Inbox } from 'lucide-react'
import { getModels, deleteModel } from '../services/api'
import { splitTimestampName } from '../utils/displayName'

function fmtTime(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
}

function datasetName(yamlPath) {
  if (!yamlPath) return '—'
  const parts = yamlPath.split('/')
  // .../datasets/<cat>/yolo/<job_id>/dataset.yaml -> show the job dir
  return parts.length >= 2 ? parts[parts.length - 2] : yamlPath
}

function metricStr(m) {
  if (!m) return null
  const v = m['metrics/mAP50(B)'] ?? m['mAP50'] ?? null
  return v != null ? Number(v).toFixed(3) : null
}

export default function ModelListPanel({ reloadToken }) {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchModels = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setModels(await getModels())
    } catch (e) {
      setError(e?.message || '加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchModels()
  }, [fetchModels, reloadToken])

  const handleDelete = async (e, m) => {
    e.stopPropagation()
    if (!window.confirm(`从列表删除模型「${splitTimestampName(m.name).display} v${m.version}」？（保留权重文件）`)) return
    try {
      await deleteModel(m.id)
      fetchModels()
    } catch (err) {
      setError(err?.message || '删除失败')
    }
  }

  return (
    <div className="card p-5 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-ink-900 flex items-center gap-2">
          <span className="p-1.5 rounded-lg bg-brand-50 text-brand-500">
            <Boxes size={14} />
          </span>
          已训练模型
        </h3>
        <button
          type="button"
          onClick={fetchModels}
          disabled={loading}
          className="p-1.5 rounded-lg text-ink-500 hover:text-brand-600 hover:bg-ink-50 disabled:opacity-50"
          title="刷新"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {error && (
        <div className="p-2.5 rounded-lg border border-red-200 bg-red-50 text-red-700 text-xs">⚠ {error}</div>
      )}

      {models.length === 0 && !loading ? (
        <div className="flex flex-col items-center justify-center py-8 text-ink-400">
          <Inbox size={28} className="mb-2" />
          <span className="text-sm">还没有训练好的模型</span>
        </div>
      ) : (
        <ul className="flex flex-col gap-2">
          {models.map((m) => {
            const map50 = metricStr(m.metrics) ?? (m.metric_map50 != null ? Number(m.metric_map50).toFixed(3) : null)
            return (
              <li key={m.id} className="group relative">
                <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl border border-ink-200 hover:border-brand-300 hover:bg-brand-50/30 transition-colors cursor-default">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-ink-900 text-sm truncate">{splitTimestampName(m.name).display}</span>
                      {m.version > 1 && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-ink-100 text-ink-600 font-medium">
                          v{m.version}
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-ink-400 mt-0.5 flex items-center gap-1">
                      <Clock size={11} /> {fmtTime(m.trained_finished_at || m.created_at)}
                      {m.val_is_train
                        ? <span className="ml-2 text-red-600 font-medium">mAP 训练集自测</span>
                        : (map50 && <span className="ml-2">mAP50 {map50}</span>)}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={(e) => handleDelete(e, m)}
                    className="opacity-0 group-hover:opacity-100 p-1 rounded-lg text-ink-400 hover:text-red-600 hover:bg-red-50 transition-all"
                    title="删除模型记录"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>

                {/* Hover detail popover */}
                <div className="absolute left-0 right-0 top-full mt-1 z-20 hidden group-hover:block">
                  <div className="card p-3 shadow-xl border border-ink-200 text-xs space-y-1.5">
                    <div className="font-semibold text-ink-800">{splitTimestampName(m.name).display} v{m.version}</div>
                    <div className="flex items-center gap-1.5 text-ink-600">
                      <Clock size={12} /> 训练完成：{fmtTime(m.trained_finished_at)}
                    </div>
                    <div className="flex items-center gap-1.5 text-ink-600">
                      <Database size={12} /> 数据集：{datasetName(m.dataset_yaml)}（{m.num_images} 张）
                    </div>
                    <div className="flex items-center gap-1.5 text-ink-600">
                      <Tag size={12} /> 类别：
                      {m.class_names ? Object.values(m.class_names).map((v) => splitTimestampName(v).display).join('、') : '—'}
                    </div>
                    <div className="text-ink-600">
                      {m.val_is_train ? (
                        <span className="text-red-600 font-medium">⚠ 指标在训练集自测，不可作为泛化依据</span>
                      ) : (
                        <>
                          指标：mAP50 {map50 ?? '—'}
                          {m.metric_map50_95 != null && ` · mAP50-95 ${Number(m.metric_map50_95).toFixed(3)}`}
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
