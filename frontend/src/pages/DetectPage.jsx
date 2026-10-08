/**
 * src/pages/DetectPage.jsx
 * ------------------------
 * The video-detection workspace: left config column (upload + model/prompt +
 * advanced settings + start) and a right task grid. All state comes from the
 * ConsoleLayout via useOutletContext so tasks keep streaming across navigation.
 */

import React, { useMemo } from 'react'
import { useOutletContext } from 'react-router-dom'
import {
  Play, Settings, Trash2, Activity, Film,
} from 'lucide-react'

import VideoUploader from '../components/VideoUploader'
import PromptInput from '../components/PromptInput'
import DetectModelSelect from '../components/DetectModelSelect'
import TaskCard from '../components/TaskCard'
import PageHeader from '../components/PageHeader'
import WorkspaceEmpty, { WorkspaceSteps } from '../components/WorkspaceEmpty'
import { StatRow } from '../components/ui'

export default function DetectPage() {
  const {
    tasks, addFiles, removeTask, clearAll, toggleCollapse,
    cancel, pause, resume, terminate,
    prompt, setPrompt, detInterval, setDetInterval, enableVlm, setEnableVlm,
    showAdvanced, setShowAdvanced, selectedModelId, modelReloadToken,
    detectionMode, setDetectionMode, llmPrompt, setLlmPrompt, llmModel, setLlmModel,
    llmModels, loadingLlmModels,
    handleSelectModel, handleStartAll, handleRetry,
  } = useOutletContext()

  const counts = useMemo(() => {
    const c = {
      queued: 0, uploading: 0, running: 0, paused: 0, packaging: 0,
      finished: 0, failed: 0, cancelled: 0, pending: 0, early_terminated: 0,
    }
    for (const t of tasks) c[t.taskStatus] = (c[t.taskStatus] || 0) + 1
    return c
  }, [tasks])

  const hasWork = tasks.length > 0
  const anyActive = ['uploading', 'pending', 'running', 'paused', 'packaging'].some(
    (s) => (counts[s] || 0) > 0
  )
  const queuedOrFailed = (counts.queued || 0) + (counts.failed || 0) + (counts.cancelled || 0)
  const canStart = hasWork &&
    (detectionMode === 'yolo' ? (selectedModelId || prompt.trim().length > 0) : llmPrompt.trim().length > 0) &&
    queuedOrFailed > 0 && !anyActive

  return (
    <>
      <PageHeader
        title="视频检测"
        subtitle="让语言定义目标，在连续画面中追踪每一次发现。"
        icon={<Film size={18} />}
        right={
          <span className="hidden md:inline-flex items-center gap-2 text-xs text-brand-600 bg-brand-50 border border-brand-100 px-3 py-1 rounded-full">
            <Film size={12} />
            批量视频检测
          </span>
        }
      />

      <WorkspaceSteps steps={['导入视频', '定义检测目标', '查看追踪结果', '导出检测记录']} />
      <div className="workspace-grid">
        {/* ── Left: config ──────────────────────────────────────── */}
        <aside className="flex flex-col gap-5">
          <div className="card p-5 flex flex-col gap-5">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-ink-900 flex items-center gap-2">
                <span className="p-1.5 rounded-lg bg-brand-50 text-brand-500">
                  <Film size={14} />
                </span>
                <span className="workspace-section-index">01</span> 任务配置
              </h3>
              {hasWork && (
                <button
                  type="button"
                  onClick={clearAll}
                  disabled={anyActive}
                  className={[
                    'text-xs flex items-center gap-1',
                    anyActive ? 'text-ink-300 cursor-not-allowed' : 'text-ink-500 hover:text-red-500',
                  ].join(' ')}
                >
                  <Trash2 size={12} />
                  清空全部
                </button>
              )}
            </div>

            <VideoUploader onFilesSelected={addFiles} disabled={false} hasTasks={hasWork} />

            {/* 检测方式选择 */}
            <div className="flex flex-col gap-2">
              <label className="text-ink-700 text-sm font-medium">检测方式</label>
              <div className="flex gap-3">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    value="yolo"
                    checked={detectionMode === 'yolo'}
                    onChange={(e) => setDetectionMode(e.target.value)}
                    disabled={anyActive}
                    className="w-4 h-4 text-brand-500 focus:ring-brand-500"
                  />
                  <span className="text-sm text-ink-700">YOLO 检测</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    value="llm"
                    checked={detectionMode === 'llm'}
                    onChange={(e) => setDetectionMode(e.target.value)}
                    disabled={anyActive}
                    className="w-4 h-4 text-brand-500 focus:ring-brand-500"
                  />
                  <span className="text-sm text-ink-700">自然语言检测</span>
                </label>
              </div>
            </div>

            {/* YOLO 模式 */}
            {detectionMode === 'yolo' && (
              <>
                <DetectModelSelect
                  value={selectedModelId}
                  onChange={handleSelectModel}
                  disabled={anyActive}
                  reloadToken={modelReloadToken}
                />

                {!selectedModelId && (
                  <PromptInput value={prompt} onChange={setPrompt} disabled={anyActive} />
                )}
              </>
            )}

            {/* LLM 模式 */}
            {detectionMode === 'llm' && (
              <>
                <div className="flex flex-col gap-2">
                  <label className="text-ink-700 text-sm font-medium">检测目标</label>
                  <textarea
                    value={llmPrompt}
                    onChange={(e) => setLlmPrompt(e.target.value)}
                    disabled={anyActive}
                    rows={2}
                    placeholder="例如：画面中的红色安全帽"
                    className={[
                      'w-full rounded-xl px-4 py-2.5 text-ink-800 placeholder-ink-400 resize-none',
                      'bg-surface border border-ink-200',
                      'focus:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-100',
                      'transition-colors duration-200',
                      anyActive ? 'opacity-60 cursor-not-allowed' : '',
                    ].join(' ')}
                  />
                  <p className="text-xs text-ink-500">
                    用自然语言描述要检测的目标
                  </p>
                </div>

                <div className="flex flex-col gap-2">
                  <label className="text-ink-700 text-sm font-medium">LLM 模型</label>
                  <select
                    value={llmModel}
                    onChange={(e) => setLlmModel(e.target.value)}
                    disabled={anyActive || loadingLlmModels}
                    className={[
                      'w-full rounded-xl px-4 py-2.5 text-ink-800',
                      'bg-surface border border-ink-200',
                      'focus:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-100',
                      'transition-colors duration-200',
                      anyActive ? 'opacity-60 cursor-not-allowed' : '',
                    ].join(' ')}
                  >
                    {loadingLlmModels ? (
                      <option>加载中...</option>
                    ) : llmModels.length === 0 ? (
                      <option>暂无可用模型</option>
                    ) : (
                      llmModels.map((model) => (
                        <option key={model} value={model}>{model}</option>
                      ))
                    )}
                  </select>
                  <p className="text-xs text-ink-500">
                    选择用于检测的多模态大模型
                  </p>
                </div>
              </>
            )}

            {/* Advanced toggle */}
            <button
              type="button"
              onClick={() => setShowAdvanced((v) => !v)}
              className="flex items-center gap-2 text-ink-500 hover:text-ink-800 text-sm transition-colors"
            >
              <Settings size={14} />
              高级设置
              <span className="ml-auto">{showAdvanced ? '▲' : '▼'}</span>
            </button>

            {showAdvanced && (
              <div className="flex flex-col gap-3 pl-3 border-l-2 border-brand-100">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableVlm}
                    onChange={(e) => setEnableVlm(e.target.checked)}
                    disabled={anyActive}
                    className="accent-brand-500 w-4 h-4"
                  />
                  <div className="flex flex-col">
                    <span className="text-ink-700 text-sm">视觉大模型语义复核</span>
                    <span className="text-ink-400 text-xs">
                      对候选目标补充语义判断，会增加处理时间
                    </span>
                  </div>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-ink-700 text-sm">检测间隔（每 N 帧全量检测）</span>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min={1}
                      max={30}
                      value={detInterval}
                      onChange={(e) => setDetInterval(Number(e.target.value))}
                      disabled={anyActive}
                      className="flex-1 accent-brand-500"
                    />
                    <span className="text-brand-600 font-mono w-8 text-center">{detInterval}</span>
                  </div>
                  <span className="text-ink-400 text-xs">值越大速度越快，精度略降。推荐 3 ~ 10</span>
                </label>
              </div>
            )}

            <button
              type="button"
              onClick={handleStartAll}
              disabled={!canStart}
              className={[
                'w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-semibold transition-all duration-200',
                canStart
                  ? 'bg-brand-500 text-white hover:bg-brand-600 shadow-brand'
                  : 'bg-ink-100 text-ink-400 cursor-not-allowed',
              ].join(' ')}
            >
              {anyActive ? <Activity size={16} className="animate-pulse" /> : <Play size={16} />}
              {anyActive
                ? `处理中…（${counts.running + counts.uploading + counts.pending + counts.paused + counts.packaging} 个任务）`
                : queuedOrFailed > 0
                ? `开始检测（${queuedOrFailed} 个视频）`
                : hasWork
                ? '全部已处理'
                : '开始检测'}
            </button>
          </div>

          {/* Stats card */}
          {hasWork && (
            <div className="card p-4 text-sm">
              <h4 className="text-ink-500 font-medium mb-3 text-xs uppercase tracking-wider">任务统计</h4>
              <div className="grid grid-cols-2 gap-y-2 gap-x-4">
                <StatRow label="视频总数" value={tasks.length} />
                <StatRow label="等待中" value={counts.queued || 0} tone="ink" />
                <StatRow
                  label="进行中"
                  value={(counts.uploading || 0) + (counts.pending || 0) + (counts.running || 0) + (counts.packaging || 0)}
                  tone="brand"
                />
                <StatRow label="已完成" value={counts.finished || 0} tone="emerald" />
                {counts.failed > 0 && <StatRow label="失败" value={counts.failed} tone="red" />}
              </div>
            </div>
          )}
        </aside>

        {/* ── Right: task grid ──────────────────────────────────── */}
        <section className="min-w-0">
          {!hasWork ? (
            <WorkspaceEmpty
              label="视频感知 / 结果预览"
              title="从一段影像，发现目标"
              description="导入视频并描述需要识别的目标。检测开始后，标注画面与任务进度将在这里呈现。"
              tags={['批量上传', '跨帧追踪', '任务进度']}
            />
          ) : (
            <div className="grid grid-cols-1 gap-4">
              {tasks.map((t) => (
                <TaskCard
                  key={t.id}
                  task={t}
                  onRemove={removeTask}
                  onRetry={handleRetry}
                  onCancel={cancel}
                  onPause={pause}
                  onResume={resume}
                  onTerminate={terminate}
                  onToggleCollapse={toggleCollapse}
                />
              ))}
            </div>
          )}
        </section>
      </div>
    </>
  )
}
