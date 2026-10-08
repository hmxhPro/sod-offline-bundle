/**
 * src/pages/HistoryPage.jsx
 * -------------------------
 * Full-page task history (replaces the old floating HistoryPanel widget). Lists
 * past tasks grouped by day (今天 / 昨天 / YYYY-MM-DD); a row opens
 * HistoryDetailModal. Supports per-row delete and a guarded wipe-all.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { History, RefreshCw, Trash2 } from 'lucide-react'
import { getTaskHistory, deleteHistoryTask, deleteAllHistory } from '../services/api'
import PageHeader from '../components/PageHeader'
import WorkspaceEmpty from '../components/WorkspaceEmpty'
import HistoryDetailModal from '../components/HistoryDetailModal'

const STATUS_META = {
  pending:          { label: '排队中',  cls: 'bg-brand-50 text-brand-700' },
  running:          { label: '检测中',  cls: 'bg-brand-100 text-brand-700' },
  paused:           { label: '已暂停',  cls: 'bg-amber-50 text-amber-700' },
  packaging:        { label: '打包中',  cls: 'bg-brand-100 text-brand-700' },
  finished:         { label: '已完成',  cls: 'bg-emerald-50 text-emerald-700' },
  failed:           { label: '失败',    cls: 'bg-red-50 text-red-700' },
  cancelled:        { label: '已取消',  cls: 'bg-ink-100 text-ink-600' },
  early_terminated: { label: '提前终止', cls: 'bg-amber-50 text-amber-800' },
}

function dayBucket(iso) {
  const d = new Date(iso)
  const today = new Date()
  const startOf = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime()
  const diffDays = Math.round((startOf(today) - startOf(d)) / 86_400_000)
  if (diffDays === 0) return '今天'
  if (diffDays === 1) return '昨天'
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

function formatTime(iso) {
  const d = new Date(iso)
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
}

export default function HistoryPage() {
  const [tasks, setTasks] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedTaskId, setSelectedTaskId] = useState(null)
  const [deletingId, setDeletingId] = useState(null)
  const [clearingAll, setClearingAll] = useState(false)

  const fetchHistory = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setTasks(await getTaskHistory({ limit: 200 }))
    } catch (err) {
      setError(err?.message || '加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchHistory()
  }, [fetchHistory])

  const grouped = useMemo(() => {
    const out = new Map()
    for (const t of tasks) {
      const key = dayBucket(t.created_at)
      if (!out.has(key)) out.set(key, [])
      out.get(key).push(t)
    }
    return Array.from(out.entries())
  }, [tasks])

  // ── Delete one row ──────────────────────────────────────────────────
  const handleDelete = useCallback(async (task) => {
    const label = task.video_filename || task.task_id.slice(0, 8)
    const ok = window.confirm(
      `确定删除任务「${label}」吗？\n\n` +
      `此操作不可逆，将同时删除：\n` +
      `  · 该任务的历史记录（数据库）\n` +
      `  · 已保存的检测帧与 ZIP 归档\n` +
      `  · 上传的原始视频（若无其他任务引用）`
    )
    if (!ok) return
    setDeletingId(task.task_id)
    setError(null)
    try {
      await deleteHistoryTask(task.task_id)
      setTasks((prev) => prev.filter((t) => t.task_id !== task.task_id))
      fetchHistory()
    } catch (err) {
      setError(err?.message || '删除失败')
    } finally {
      setDeletingId(null)
    }
  }, [fetchHistory])

  // ── Wipe everything ─────────────────────────────────────────────────
  const handleDeleteAll = useCallback(async () => {
    if (tasks.length === 0) return
    const ok = window.confirm(
      `确定清空全部 ${tasks.length} 条历史任务吗？\n\n` +
      `此操作不可逆，将同时删除：\n` +
      `  · 全部历史记录（数据库）\n` +
      `  · 所有任务的检测帧与 ZIP 归档\n` +
      `  · 所有上传的原始视频文件`
    )
    if (!ok) return
    setClearingAll(true)
    setError(null)
    try {
      await deleteAllHistory()
      setTasks([])
      fetchHistory()
    } catch (err) {
      setError(err?.message || '清空失败')
    } finally {
      setClearingAll(false)
    }
  }, [tasks.length, fetchHistory])

  return (
    <>
      <PageHeader
        title="历史记录"
        subtitle="回到每一次发现，查看检测结果、运行记录与归档文件。"
        icon={<History size={18} />}
        right={
          <>
            <button
              type="button"
              onClick={fetchHistory}
              disabled={loading}
              title="刷新"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-ink-200 bg-surface text-sm text-ink-600 hover:text-brand-600 hover:border-brand-300 disabled:opacity-50 transition-colors"
            >
              <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
              刷新
            </button>
            <button
              type="button"
              onClick={handleDeleteAll}
              disabled={clearingAll || tasks.length === 0}
              title="清空全部历史"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-red-200 bg-surface text-sm text-red-600 hover:bg-red-50 disabled:opacity-30 disabled:hover:bg-surface transition-colors"
            >
              <Trash2 size={14} className={clearingAll ? 'animate-pulse' : ''} />
              清空全部
            </button>
          </>
        }
      />

      {tasks.length > 0 && (
        <div className="workspace-history-metadata">
          <span>已加载 <strong>{tasks.length}</strong> 条任务</span>
          <span>已完成 <strong>{tasks.filter((task) => task.status === 'finished').length}</strong></span>
          <span>可下载归档 <strong>{tasks.filter((task) => task.zip_ready).length}</strong></span>
        </div>
      )}

      {error && (
        <div className="mb-4 p-3 rounded-lg border border-red-200 bg-red-50 text-red-700 text-sm">
          ⚠ {error}
        </div>
      )}

      {!error && !loading && tasks.length === 0 ? (
        <WorkspaceEmpty
          label="检测档案 / 历史记录"
          stateLabel="暂无记录"
          title="每一次发现，都有迹可循"
          description="目前还没有检测记录。创建视频检测任务后，可在这里回看任务详情，并下载已生成的归档文件。"
          tags={['任务回溯', '结果查看', '归档下载']}
          compact
        />
      ) : (
        <div className="flex flex-col gap-6">
          {grouped.map(([day, items]) => (
            <section key={day}>
              <h3 className="px-1 pb-2 text-[11px] font-medium text-ink-500 uppercase tracking-wider">
                {day} <span className="text-ink-400">· {items.length}</span>
              </h3>
              <div className="card divide-y divide-ink-100 overflow-hidden">
                {items.map((t) => {
                  const meta = STATUS_META[t.status] ?? { label: t.status, cls: 'bg-ink-100 text-ink-600' }
                  const isDeleting = deletingId === t.task_id
                  return (
                    <div key={t.task_id} className="group relative flex items-center">
                      <div
                        role="button"
                        tabIndex={0}
                        onClick={() => !isDeleting && setSelectedTaskId(t.task_id)}
                        onKeyDown={(e) => {
                          if (isDeleting) return
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault()
                            setSelectedTaskId(t.task_id)
                          }
                        }}
                        className={[
                          'flex-1 min-w-0 text-left px-4 py-3 pr-12 transition-colors cursor-pointer',
                          isDeleting ? 'opacity-50 pointer-events-none' : 'hover:bg-ink-50',
                        ].join(' ')}
                      >
                        <div className="flex items-center gap-2 mb-1">
                          <span className="flex-1 truncate font-medium text-ink-800 text-sm">
                            {t.video_filename || t.task_id.slice(0, 8)}
                          </span>
                          <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${meta.cls}`}>
                            {meta.label}
                          </span>
                        </div>
                        {t.prompt && <div className="text-xs text-ink-500 truncate">{t.prompt}</div>}
                        <div className="mt-1 flex items-center gap-3 text-[11px] text-ink-400">
                          <span>{formatTime(t.created_at)}</span>
                          {t.processed_frames > 0 && (
                            <span>{t.processed_frames}/{t.total_frames || '?'} 帧</span>
                          )}
                          {t.zip_ready && <span className="text-emerald-600">ZIP 可下载</span>}
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); handleDelete(t) }}
                        disabled={isDeleting}
                        title="删除该任务及所有缓存"
                        className="absolute right-3 p-1.5 rounded-lg text-ink-400 hover:text-red-600 hover:bg-red-50 transition-colors disabled:cursor-wait"
                      >
                        <Trash2 size={14} className={isDeleting ? 'animate-pulse' : ''} />
                      </button>
                    </div>
                  )
                })}
              </div>
            </section>
          ))}
        </div>
      )}

      {selectedTaskId && (
        <HistoryDetailModal taskId={selectedTaskId} onClose={() => setSelectedTaskId(null)} />
      )}
    </>
  )
}
