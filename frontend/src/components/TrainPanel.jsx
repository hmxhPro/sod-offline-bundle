/**
 * src/components/TrainPanel.jsx
 * -----------------------------
 * Configure + launch a training run for the selected category, and show live
 * progress (epoch / total + mAP) via useTrainingJobs polling (REQ2).
 */

import React, { useState } from 'react'
import { Rocket, Loader2, CheckCircle2, XCircle, Ban } from 'lucide-react'
import { useTrainingJobs } from '../hooks/useTrainingJobs'
import { cancelTraining } from '../services/api'

const JOB_META = {
  pending: { label: '排队中', cls: 'text-brand-600', icon: Loader2, spin: true },
  running: { label: '训练中', cls: 'text-brand-600', icon: Loader2, spin: true },
  finished: { label: '已完成', cls: 'text-emerald-600', icon: CheckCircle2 },
  needs_review: { label: '指标过低·需复核', cls: 'text-amber-600', icon: XCircle },
  failed: { label: '失败', cls: 'text-red-600', icon: XCircle },
  cancelled: { label: '已取消', cls: 'text-ink-500', icon: Ban },
}

function NumField({ label, value, onChange, min, max, step = 1, disabled }) {
  const handleChange = (e) => {
    const val = e.target.value
    // 允许空字符串（用户正在删除），否则转换为数字
    if (val === '') {
      onChange('')
    } else {
      const num = Number(val)
      onChange(isNaN(num) ? value : num)
    }
  }

  return (
    <label className="flex flex-col gap-1 flex-1">
      <span className="text-xs text-ink-500">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={handleChange}
        className="px-2.5 py-1.5 rounded-lg border border-ink-200 focus:border-brand-400 focus:ring-2 focus:ring-brand-100 outline-none text-sm disabled:bg-ink-50"
      />
    </label>
  )
}

export default function TrainPanel({ category, onTrained }) {
  const [epochs, setEpochs] = useState(100)
  const [imgsz, setImgsz] = useState(640)
  const [batch, setBatch] = useState(16)
  const { job, error, starting, start } = useTrainingJobs(onTrained)

  const annotated = category?.annotated_count || 0
  const active = job && ['pending', 'running'].includes(job.status)
  const canTrain = !!category && annotated > 0 && !starting && !active

  const handleTrain = async () => {
    if (!canTrain) return
    try {
      await start(category.id, { epochs, imgsz, batch })
    } catch {
      /* error surfaced via hook */
    }
  }

  const pct = job
    ? Math.round((job.total_epochs ? job.current_epoch / job.total_epochs : job.progress || 0) * 100)
    : 0
  const meta = job ? JOB_META[job.status] || JOB_META.pending : null
  const StatusIcon = meta?.icon

  return (
    <div className="card p-5 flex flex-col gap-4">
      <h3 className="font-semibold text-ink-900 flex items-center gap-2">
        <span className="p-1.5 rounded-lg bg-brand-50 text-brand-500">
          <Rocket size={14} />
        </span>
        训练模型
      </h3>

      {!category ? (
        <p className="text-ink-500 text-sm">请先选择一个类别。</p>
      ) : (
        <>
          <p className="text-sm text-ink-600">
            类别：<span className="font-medium text-ink-900">{category.name}</span> · 可训练图片{' '}
            <span className="font-mono">{annotated}</span> 张 / 总图片 {category.image_count} 张
          </p>

          <div className="flex gap-2">
            <NumField label="轮数 epochs" value={epochs} onChange={setEpochs} min={1} max={1000} disabled={active} />
            <NumField label="图像尺寸 imgsz" value={imgsz} onChange={setImgsz} min={64} max={2048} step={32} disabled={active} />
            <NumField label="批大小 batch" value={batch} onChange={setBatch} min={-1} max={256} disabled={active} />
          </div>
          <p className="text-[11px] text-ink-400">
            快速验证可设 epochs=2、imgsz=320。batch 设 -1 可让系统自动选择合适的值。
          </p>

          <button
            type="button"
            onClick={handleTrain}
            disabled={!canTrain}
            className={[
              'w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl font-semibold transition-all',
              canTrain ? 'bg-brand-500 text-white hover:bg-brand-600 shadow-brand' : 'bg-ink-100 text-ink-400 cursor-not-allowed',
            ].join(' ')}
          >
            {starting || active ? <Loader2 size={16} className="animate-spin" /> : <Rocket size={16} />}
            {active ? '训练进行中…' : annotated === 0 ? '请先标注图片' : '开始训练'}
          </button>

          {error && (
            <div className="p-2.5 rounded-lg border border-red-200 bg-red-50 text-red-700 text-xs">⚠ {error}</div>
          )}

          {job && (
            <div className="flex flex-col gap-2 pt-1 border-t border-ink-100">
              <div className="flex items-center justify-between text-sm">
                <span className={`flex items-center gap-1.5 ${meta?.cls}`}>
                  {StatusIcon && <StatusIcon size={14} className={meta?.spin ? 'animate-spin' : ''} />}
                  {meta?.label}
                  {job.status === 'running' && job.total_epochs > 0 && (
                    <span className="text-ink-500 font-mono ml-1">
                      epoch {job.current_epoch}/{job.total_epochs}
                    </span>
                  )}
                </span>
                <span className="font-mono text-brand-600">{pct}%</span>
              </div>
              <div className="h-1.5 bg-ink-100 rounded-full overflow-hidden">
                <div
                  className={[
                    'h-full rounded-full transition-all duration-500',
                    job.status === 'finished' ? 'bg-emerald-500'
                      : job.status === 'needs_review' ? 'bg-amber-500'
                      : job.status === 'failed' || job.status === 'cancelled' ? 'bg-ink-400'
                      : 'bg-brand-500 progress-glow',
                  ].join(' ')}
                  style={{ width: `${Math.min(pct, 100)}%` }}
                />
              </div>
              {(job.metric_map50 != null || job.metric_map50_95 != null) && (
                <div className="text-xs flex flex-col gap-0.5">
                  <div className="flex gap-4 text-ink-500">
                    {job.metric_map50 != null && <span>mAP50: <span className={`font-mono ${job.val_is_train ? 'text-ink-400' : 'text-ink-800'}`}>{job.metric_map50.toFixed(3)}</span></span>}
                    {job.metric_map50_95 != null && <span>mAP50-95: <span className={`font-mono ${job.val_is_train ? 'text-ink-400' : 'text-ink-800'}`}>{job.metric_map50_95.toFixed(3)}</span></span>}
                  </div>
                  {job.val_is_train && (
                    <span className="text-red-600 font-medium">⚠ 指标在训练集自测（标注样本过少），不可作为泛化依据</span>
                  )}
                </div>
              )}
              {active && (
                <button
                  type="button"
                  onClick={() => cancelTraining(job.id).catch(() => {})}
                  className="self-start text-xs text-ink-500 hover:text-red-600 flex items-center gap-1"
                >
                  <Ban size={12} /> 取消训练
                </button>
              )}
              {job.status === 'failed' && job.error && (
                <pre className="text-[11px] text-red-600 bg-red-50 rounded-lg p-2 max-h-32 overflow-auto whitespace-pre-wrap">
                  {job.error}
                </pre>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
