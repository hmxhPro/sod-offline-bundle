/**
 * src/pages/LLMAnnotationPage.jsx
 * -------------------------------
 * AI 自动标注（一站式）：用户只需 ①说出要标注的目标（自然语言或直接名词）②上传
 * 图片或视频。系统自动提取目标名词、建一个带时间戳的数据集，（视频则先抽帧）逐张
 * 标注，完成后可一键去训练该数据集。
 *
 * 模型与访问凭据由后端统一管理，前端展示配置状态与任务进度。
 */

import React, { useCallback, useEffect, useState } from 'react'
import { useNavigate, useOutletContext } from 'react-router-dom'
import {
  ScanLine, Crosshair, RefreshCw, XCircle, AlertCircle, CheckCircle2, Cpu,
  Rocket, Loader2, Image as ImageIcon, Film, Trash2,
} from 'lucide-react'

import PageHeader from '../components/PageHeader'
import WorkspaceEmpty, { WorkspaceSteps } from '../components/WorkspaceEmpty'
import ImageUploader from '../components/ImageUploader'
import VideoUploader from '../components/VideoUploader'
import { TabButton } from '../components/ui'
import {
  getCategories,
  getCategory,
  getAnnotationConfig,
  getAnnotationModels,
  setDefaultAnnotationModel,
  getAnnotationTasks,
  createAnnotationIteration,
  cancelAnnotationTask,
  deleteAnnotationTask,
  autoAnnotate,
  startTraining,
  getModels,
} from '../services/api'
import { splitTimestampName, formatTs } from '../utils/displayName'

const STATUS_META = {
  pending:   { label: '等待中', cls: 'bg-ink-100 text-ink-600' },
  running:   { label: '标注中', cls: 'bg-brand-100 text-brand-700' },
  finished:  { label: '已完成', cls: 'bg-emerald-50 text-emerald-700' },
  failed:    { label: '失败',   cls: 'bg-red-50 text-red-700' },
  cancelled: { label: '已取消', cls: 'bg-amber-50 text-amber-700' },
}

const DENSITY_OPTS = [
  { key: 'sparse', label: '稀疏', desc: '约每 3 秒 1 帧' },
  { key: 'medium', label: '适中', desc: '约每秒 1 帧' },
  { key: 'dense',  label: '密集', desc: '约每秒 3 帧' },
]

const ITERATION_SELECTIONS = [
  { key: 'pending_only', label: '只补标未标注图片' },
  { key: 'no_target_only', label: '重标未检测到目标图片' },
  { key: 'failed_only', label: '只重试检测失败图片' },
  { key: 'all', label: '全部图片重新标注' },
]

const WRITE_POLICIES = [
  { key: 'skip_existing', label: '跳过已有标注' },
  { key: 'overwrite', label: '覆盖已有标注' },
]

function StatusBadge({ status, extracting }) {
  if (extracting) {
    return (
      <span className="px-2 py-0.5 text-xs rounded-full font-medium bg-brand-100 text-brand-700">
        抽帧中
      </span>
    )
  }
  const meta = STATUS_META[status] || { label: status, cls: 'bg-ink-100 text-ink-600' }
  return (
    <span className={`px-2 py-0.5 text-xs rounded-full font-medium ${meta.cls}`}>
      {meta.label}
    </span>
  )
}

export default function LLMAnnotationPage() {
  const navigate = useNavigate()
  // 由 ConsoleLayout 经 Outlet 注入：用于「去训练该数据集」时预选训练页的当前类别。
  const { setSelectedCategory: setSharedCategory, setCatReloadToken } =
    useOutletContext() || {}

  const [config, setConfig] = useState(null)
  const [configError, setConfigError] = useState(false)
  const [categories, setCategories] = useState([])
  const [tasks, setTasks] = useState([])

  const [target, setTarget] = useState('')
  const [mode, setMode] = useState('images')         // images | video
  const [imageFiles, setImageFiles] = useState([])   // File[]
  const [videoFile, setVideoFile] = useState(null)   // File | null
  const [videoDensity, setVideoDensity] = useState('medium')

  const [submitting, setSubmitting] = useState(false)
  const [uploadPct, setUploadPct] = useState(0)
  const [error, setError] = useState(null)
  const [created, setCreated] = useState(null)       // { display, ts } 成功提示
  const [notice, setNotice] = useState(null)         // 删除等操作的结果回显
  const [modelList, setModelList] = useState(null)   // 后端模型列表查询结果
  const [loadingModels, setLoadingModels] = useState(false)
  const [selectedModel, setSelectedModel] = useState(null)  // 当前选中的模型（为 null 时使用后端默认）
  const [settingDefault, setSettingDefault] = useState(false) // 正在设置默认模型
  const [trainedModels, setTrainedModels] = useState([])  // 已训练模型列表（用于辅助标注）
  const [iterationForm, setIterationForm] = useState(null) // { categoryId, prompt, selectionMode, writePolicy, assistModelId }
  const [iterationSubmitting, setIterationSubmitting] = useState(false)
  const [trainConfirm, setTrainConfirm] = useState(null) // { categoryId, categoryName }
  const [trainParams, setTrainParams] = useState({ epochs: 100, imgsz: 640, batch: 16 })
  const [trainingStarting, setTrainingStarting] = useState(false)
  const [retryingFailedTaskId, setRetryingFailedTaskId] = useState(null)

  const loadConfig = useCallback(async () => {
    try {
      setConfig(await getAnnotationConfig())
      setConfigError(false)
    } catch {
      setConfig(null)
      setConfigError(true)
    }
  }, [])

  const loadCategories = useCallback(async () => {
    try {
      setCategories(await getCategories())
    } catch (err) {
      console.error('加载数据集失败:', err)
    }
  }, [])

  const loadTasks = useCallback(async () => {
    try {
      setTasks(await getAnnotationTasks())
    } catch (err) {
      console.error('加载任务失败:', err)
    }
  }, [])

  const loadTrainedModels = useCallback(async () => {
    try {
      setTrainedModels(await getModels())
    } catch (err) {
      console.error('加载训练模型失败:', err)
    }
  }, [])

  useEffect(() => {
    loadConfig()
    loadCategories()
    loadTasks()
    loadTrainedModels()
  }, [loadConfig, loadCategories, loadTasks, loadTrainedModels])

  // 有运行中 / 等待中的任务时，每 3 秒轮询刷新进度（同时刷新数据集列表以拿到新建的）。
  const hasActiveTask = tasks.some((t) => t.status === 'running' || t.status === 'pending')
  useEffect(() => {
    if (!hasActiveTask) return
    const timer = setInterval(() => { loadTasks(); loadCategories() }, 3000)
    return () => clearInterval(timer)
  }, [hasActiveTask, loadTasks, loadCategories])

  const handleImagesSelected = (files) => {
    setImageFiles((prev) => [...prev, ...files])
    setError(null)
  }
  const handleVideoSelected = (files) => {
    if (!files?.length) return
    if (files.length > 1) window.alert('一次只能处理一个视频，已选用第一个。')
    setVideoFile(files[0])  // 单个视频
    setError(null)
  }

  const switchMode = (m) => {
    setMode(m)
    setError(null)
  }

  const filesForSubmit = mode === 'images' ? imageFiles : (videoFile ? [videoFile] : [])
  const notConfigured = config && !config.configured
  const canSubmit = Boolean(
    !submitting && config?.configured && target.trim() && filesForSubmit.length > 0
  )

  // 调试：监听 canSubmit 状态变化
  useEffect(() => {
    console.log('canSubmit 状态:', {
      canSubmit,
      submitting,
      notConfigured,
      targetLength: target.trim().length,
      filesCount: filesForSubmit.length,
      mode,
      videoFileName: videoFile?.name,
      selectedModel
    })
  }, [canSubmit, submitting, notConfigured, target, filesForSubmit.length, mode, videoFile, selectedModel])

  const handleFetchModels = async () => {
    setLoadingModels(true)
    setError(null)
    setNotice(null)
    try {
      const res = await getAnnotationModels()
      setModelList(res)
      if (!res.models?.length) {
        setNotice('模型列表接口返回为空，请检查模型服务配置及可用权限。')
      }
    } catch (err) {
      setError(err?.message || '获取模型列表失败')
    } finally {
      setLoadingModels(false)
    }
  }

  const handleSetDefaultModel = async () => {
    if (!selectedModel) return
    setSettingDefault(true)
    setError(null)
    setNotice(null)
    try {
      const cfg = await setDefaultAnnotationModel(selectedModel)
      setConfig(cfg)
      setModelList((ml) => (ml ? { ...ml, current_model: cfg.model, current_model_available: true } : ml))
      setNotice(`默认模型已设置为 ${cfg.model}（已探测可用并写入配置，重启后仍生效）`)
      setSelectedModel(null)
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || '设置默认模型失败')
    } finally {
      setSettingDefault(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setCreated(null)
    setNotice(null)
    if (!target.trim()) { setError('请先填写要标注的目标'); return }
    if (filesForSubmit.length === 0) {
      setError(mode === 'images' ? '请上传至少一张图片' : '请上传一个视频'); return
    }

    setSubmitting(true)
    setUploadPct(0)
    try {
      const res = await autoAnnotate(
        { target: target.trim(), files: filesForSubmit, videoDensity, model: selectedModel },
        setUploadPct,
      )
      const { display, ts } = splitTimestampName(res.category_name)
      setCreated({ display: display || res.noun, ts })
      // 重置输入，准备下一次
      setTarget('')
      setImageFiles([])
      setVideoFile(null)
      await Promise.all([loadTasks(), loadCategories()])
    } catch (err) {
      setError(err?.message || '创建失败，请重试')
    } finally {
      setSubmitting(false)
      setUploadPct(0)
    }
  }

  const handleCancel = async (taskId) => {
    if (!window.confirm('确定取消该标注任务？')) return
    try {
      await cancelAnnotationTask(taskId)
      loadTasks()
    } catch (err) {
      setError(err?.message || '取消任务失败')
    }
  }

  // 删除已结束（失败 / 已取消 / 未识别到目标）的任务，并清理其中未成功标注的数据。
  // 文案不预判后端结果（是否清空数据集、是否受保护由后端整库判定），删除后据实回显。
  const handleDelete = async (task) => {
    const cat = categories.find((c) => c.id === task.category_id)
    const { display: catName } = splitTimestampName(cat?.name || task.category_id)
    const okMsg =
      `删除标注任务「${catName}」？\n` +
      '系统会清理其中未成功标注的图片，已成功标注的图片会保留；' +
      '若该数据集已无可用图片，则一并删除。'
    if (!window.confirm(okMsg)) return
    setError(null)
    setNotice(null)
    try {
      const res = await deleteAnnotationTask(task.id)
      const name = res?.dataset_name
        ? splitTimestampName(res.dataset_name).display
        : catName
      let msg
      if (res?.protected) {
        msg = `已删除该任务记录。数据集「${name}」已用于训练，相关数据已保留。`
      } else if (res?.dataset_deleted) {
        msg = `已删除该任务及数据集「${name}」。`
      } else if ((res?.purged_images || 0) > 0) {
        msg = `已删除该任务，并清理 ${res.purged_images} 张未成功标注的图片；数据集「${name}」中已标注的图片已保留。`
      } else {
        msg = '已删除该任务。'
      }
      setNotice(msg)
      await Promise.all([loadTasks(), loadCategories()])
    } catch (err) {
      setError(err?.message || '删除任务失败')
    }
  }

  // 跳转「模型训练」页并预选该数据集，标注好的数据可直接训练。
  const handleGoTrain = async (categoryId) => {
    let cat = categories.find((c) => c.id === categoryId) || null
    try {
      cat = await getCategory(categoryId)
    } catch {
      /* 网络异常时退回到列表里的副本 */
    }
    if (cat) setSharedCategory?.(cat)
    setCatReloadToken?.((t) => t + 1)
    navigate('/training')
  }

  const openIterationForm = (task, overrides = {}) => {
    setError(null)
    setNotice(null)
    setTrainConfirm(null)
    setIterationForm({
      categoryId: task.category_id,
      prompt: task.prompt || '',
      selectionMode: overrides.selectionMode || 'pending_only',
      writePolicy: overrides.writePolicy || 'skip_existing',
      assistModelId: '',
    })
  }

  const handleRetryFailedImages = async (task) => {
    if (!task?.category_id || !task?.prompt) return
    setRetryingFailedTaskId(task.id)
    setError(null)
    setNotice(null)
    try {
      const res = await createAnnotationIteration(task.category_id, {
        prompt: task.prompt.trim(),
        selection_mode: 'failed_only',
        write_policy: 'skip_existing',
        model: selectedModel,
      })
      setNotice(`已创建第 ${res?.iteration?.iteration_no || ''} 轮失败图片重检任务，请在右侧查看进度。`)
      setIterationForm(null)
      await Promise.all([loadTasks(), loadCategories()])
    } catch (err) {
      setError(err?.message || '重检失败图片失败')
    } finally {
      setRetryingFailedTaskId(null)
    }
  }

  const handleCreateIteration = async (e) => {
    e.preventDefault()
    if (!iterationForm?.categoryId || !iterationForm.prompt.trim()) return
    setIterationSubmitting(true)
    setError(null)
    setNotice(null)
    try {
      const res = await createAnnotationIteration(iterationForm.categoryId, {
        prompt: iterationForm.prompt.trim(),
        selection_mode: iterationForm.selectionMode,
        write_policy: iterationForm.writePolicy,
        model: selectedModel,
        assist_model_id: iterationForm.assistModelId || undefined,
      })
      setNotice(`已创建第 ${res?.iteration?.iteration_no || ''} 轮标注任务，请在右侧查看进度。`)
      setIterationForm(null)
      await Promise.all([loadTasks(), loadCategories()])
    } catch (err) {
      setError(err?.message || '创建迭代标注失败')
    } finally {
      setIterationSubmitting(false)
    }
  }

  const openTrainConfirm = (task) => {
    const cat = categories.find((c) => c.id === task.category_id)
    const { display } = splitTimestampName(cat?.name || task.category_id)
    setIterationForm(null)
    setTrainConfirm({ categoryId: task.category_id, categoryName: display })
    setError(null)
  }

  const handleStartTraining = async (e) => {
    e.preventDefault()
    if (!trainConfirm?.categoryId) return
    setTrainingStarting(true)
    setError(null)
    try {
      const cat = await getCategory(trainConfirm.categoryId).catch(() => null)
      if (cat) setSharedCategory?.(cat)
      await startTraining(trainConfirm.categoryId, trainParams)
      setCatReloadToken?.((t) => t + 1)
      navigate('/training')
    } catch (err) {
      setError(err?.message || '启动训练失败')
    } finally {
      setTrainingStarting(false)
    }
  }

  const inputCls = [
    'w-full rounded-xl px-4 py-2.5 text-ink-800 placeholder-ink-400',
    'bg-surface border border-ink-200',
    'focus:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-100',
    'transition-colors duration-200',
  ].join(' ')

  return (
    <>
      <PageHeader
        title="智能标注"
        subtitle="用语言描述目标，让视觉理解转化为可复用的训练样本。"
        icon={<ScanLine size={18} />}
        right={
          config && (
            <span className="hidden md:inline-flex items-center gap-2 text-xs text-brand-600 bg-brand-50 border border-brand-100 px-3 py-1 rounded-full">
              <Cpu size={12} />
              视觉模型 · {config.configured ? '访问密钥已配置' : '访问密钥未配置'}
            </span>
          )
        }
      />

      <WorkspaceSteps steps={['描述目标', '导入素材', '生成与修正标注', '进入模型训练']} />
      <div className="workspace-grid items-start">
        {/* ── 左侧：创建任务 ─────────────────────────────────────── */}
        <aside className="flex flex-col gap-5">
          <form onSubmit={handleSubmit} className="card p-5 flex flex-col gap-5">
            <h3 className="font-semibold text-ink-900 flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-brand-50 text-brand-500">
                <Crosshair size={14} />
              </span>
              <span className="workspace-section-index">01</span> 新建标注
            </h3>

            {notConfigured && (
              <div className="rounded-xl bg-amber-50 border border-amber-200 p-3 flex items-start gap-2">
                <AlertCircle size={16} className="text-amber-500 shrink-0 mt-0.5" />
                <p className="text-xs text-amber-700 leading-relaxed">
                  尚未配置视觉模型访问密钥，请联系管理员配置后再使用。
                </p>
              </div>
            )}

            {configError && (
              <div className="rounded-xl bg-amber-50 border border-amber-200 p-3 flex items-start gap-2">
                <AlertCircle size={16} className="text-amber-500 shrink-0 mt-0.5" />
                <p className="text-xs text-amber-700 leading-relaxed">
                  无法读取模型配置，请检查后端连接后刷新页面。
                </p>
              </div>
            )}

            {config && (
              <div className="rounded-xl bg-ink-50 border border-ink-100 p-3 flex flex-col gap-2">
                <div className="flex items-start justify-between gap-3">
                  <div className="text-xs text-ink-600 leading-relaxed">
                    <div>
                      当前标注模型：
                      <span className="font-mono text-ink-900">
                        {selectedModel || config.model}
                      </span>
                      {selectedModel && (
                        <span className="ml-1.5 text-emerald-600">✓ 已切换</span>
                      )}
                    </div>
                    <div>标注请求并发上限：{config.max_concurrent}</div>
                  </div>
                  <button
                    type="button"
                    onClick={handleFetchModels}
                    disabled={loadingModels || notConfigured}
                    className="shrink-0 inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-brand-200 text-xs font-medium text-brand-600 hover:bg-brand-50 disabled:opacity-50"
                  >
                    {loadingModels ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                    获取模型列表
                  </button>
                </div>
                {modelList && (
                  <div className="max-h-32 overflow-auto rounded-lg bg-surface border border-ink-100 p-2 text-[11px] text-ink-600">
                    <div className="mb-1 text-ink-500">
                      可用多模态模型 {modelList.models?.length || 0} 个（已过滤纯文本模型，共 {modelList.all_models_count || 0} 个）
                    </div>
                    {modelList.current_model_available === false && (
                      <div className="mb-1 text-amber-600">
                        默认模型 {modelList.current_model} 不在当前服务的可用清单中；创建任务时后端会自动探测并切换为可用的视觉模型，也可在下方手动选择。
                      </div>
                    )}
                    {modelList.models?.length ? (
                      <div className="flex flex-wrap gap-1.5">
                        {modelList.models.map((model) => (
                          <button
                            key={model}
                            type="button"
                            onClick={() => setSelectedModel(selectedModel === model ? null : model)}
                            className={`px-1.5 py-0.5 rounded border font-mono transition-colors ${
                              selectedModel === model
                                ? 'border-emerald-300 bg-emerald-100 text-emerald-800'
                                : model === modelList.current_model
                                ? 'border-brand-200 bg-brand-50 text-brand-700 hover:bg-brand-100'
                                : 'border-ink-100 bg-ink-50 text-ink-600 hover:bg-ink-100'
                            }`}
                            title={
                              selectedModel === model
                                ? '已选中 · 点击取消使用后端默认'
                                : model === modelList.current_model
                                ? '后端默认模型 · 点击选中'
                                : '点击选中此模型'
                            }
                          >
                            {model}
                            {selectedModel === model ? ' ✓' : model === modelList.current_model ? ' · 默认' : ''}
                          </button>
                        ))}
                      </div>
                    ) : (
                      <span className="text-ink-400">暂无可用多模态模型（请检查 API Key 权限）</span>
                    )}
                    {selectedModel && selectedModel !== modelList.current_model && (
                      <button
                        type="button"
                        onClick={handleSetDefaultModel}
                        disabled={settingDefault}
                        className="mt-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-emerald-300 bg-emerald-50 text-emerald-700 text-[11px] font-medium hover:bg-emerald-100 disabled:opacity-50"
                      >
                        {settingDefault ? <Loader2 size={12} className="animate-spin" /> : null}
                        {settingDefault ? '探测并设置中…' : `把 ${selectedModel} 设为默认模型`}
                      </button>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* ① 要标注的目标 */}
            <div className="flex flex-col gap-2">
              <label className="text-ink-700 text-sm font-medium">
                <span className="text-brand-500 font-semibold">①</span> 要标注的目标
                <span className="text-red-500"> *</span>
              </label>
              <textarea
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                disabled={submitting}
                rows={2}
                placeholder="如：帮我标注出河流中的船只；或直接填 船只"
                className={`${inputCls} resize-none`}
              />
              <p className="text-xs text-ink-500">
                系统会自动据此命名数据集（如"船只"），并加上时间，方便区分。
              </p>
            </div>

            {/* ② 上传素材 */}
            <div className="flex flex-col gap-2">
              <label className="text-ink-700 text-sm font-medium">
                <span className="text-brand-500 font-semibold">②</span> 上传素材
                <span className="text-red-500"> *</span>
              </label>
              <div className="flex items-center gap-2">
                <TabButton
                  active={mode === 'images'}
                  onClick={() => switchMode('images')}
                  icon={<ImageIcon size={15} />}
                >
                  图片
                </TabButton>
                <TabButton
                  active={mode === 'video'}
                  onClick={() => switchMode('video')}
                  icon={<Film size={15} />}
                >
                  视频
                </TabButton>
              </div>

              {mode === 'images' ? (
                <>
                  <ImageUploader
                    onFilesSelected={handleImagesSelected}
                    disabled={submitting}
                    hasItems={imageFiles.length > 0}
                    hint="支持多图同时上传 · JPG / PNG / BMP / WebP"
                  />
                  {imageFiles.length > 0 && (
                    <div className="flex items-center justify-between text-sm text-ink-600">
                      <span>已选 {imageFiles.length} 张图片</span>
                      <button
                        type="button"
                        onClick={() => setImageFiles([])}
                        disabled={submitting}
                        className="text-ink-400 hover:text-red-500 flex items-center gap-1"
                      >
                        <Trash2 size={13} /> 清空
                      </button>
                    </div>
                  )}
                </>
              ) : (
                <>
                  <VideoUploader
                    onFilesSelected={handleVideoSelected}
                    disabled={submitting}
                  />
                  {videoFile && (
                    <div className="flex items-center justify-between text-sm text-ink-600">
                      <span className="truncate" title={videoFile.name}>
                        已选视频：{videoFile.name}
                      </span>
                      <button
                        type="button"
                        onClick={() => setVideoFile(null)}
                        disabled={submitting}
                        className="shrink-0 ml-2 text-ink-400 hover:text-red-500 flex items-center gap-1"
                      >
                        <Trash2 size={13} /> 清空
                      </button>
                    </div>
                  )}
                  {/* 抽帧密度 */}
                  <div className="flex flex-col gap-1.5 mt-1">
                    <span className="text-xs text-ink-500">抽帧密度（每帧都会标注，越密越慢越贵）</span>
                    <div className="grid grid-cols-3 gap-2">
                      {DENSITY_OPTS.map((d) => (
                        <button
                          key={d.key}
                          type="button"
                          onClick={() => setVideoDensity(d.key)}
                          disabled={submitting}
                          className={[
                            'flex flex-col items-center gap-0.5 rounded-xl border px-2 py-2 text-center transition-colors',
                            videoDensity === d.key
                              ? 'border-brand-400 bg-brand-50/70 text-brand-700'
                              : 'border-ink-200 text-ink-600 hover:bg-ink-50',
                          ].join(' ')}
                        >
                          <span className="text-sm font-medium">{d.label}</span>
                          <span className="text-[10px] text-ink-400">{d.desc}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* 上传进度 */}
            {submitting && uploadPct > 0 && uploadPct < 100 && (
              <div>
                <div className="flex items-center justify-between text-xs text-ink-500 mb-1">
                  <span>上传中…</span>
                  <span>{uploadPct}%</span>
                </div>
                <div className="w-full bg-ink-100 rounded-full h-1.5 overflow-hidden">
                  <div className="bg-brand-500 h-1.5 rounded-full transition-all" style={{ width: `${uploadPct}%` }} />
                </div>
              </div>
            )}

            {/* 成功提示 */}
            {created && (
              <div className="rounded-xl bg-emerald-50 border border-emerald-200 px-3 py-2.5 text-sm text-emerald-800 flex items-start gap-2">
                <CheckCircle2 size={16} className="text-emerald-500 shrink-0 mt-0.5" />
                <span>
                  已创建数据集「<b>{created.display}</b>」
                  {created.ts && <span className="text-emerald-600">（{formatTs(created.ts)}）</span>}
                  ，正在自动标注，右侧可查看进度。
                </span>
              </div>
            )}

            {/* 错误 */}
            {error && (
              <div className="rounded-xl bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">
                {error}
              </div>
            )}

            {/* 提交 */}
            <button
              type="submit"
              disabled={!canSubmit}
              className={[
                'w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-semibold transition-all duration-200',
                canSubmit
                  ? 'bg-brand-500 text-white hover:bg-brand-600 shadow-brand'
                  : 'bg-ink-100 text-ink-400 cursor-not-allowed',
              ].join(' ')}
            >
              {submitting ? <Loader2 size={16} className="animate-spin" /> : <ScanLine size={16} />}
              {submitting ? '提交中…' : '开始标注'}
            </button>
          </form>

          {iterationForm && (
            <form onSubmit={handleCreateIteration} className="card p-5 flex flex-col gap-4 border-brand-100">
              <h3 className="font-semibold text-ink-900 flex items-center gap-2">
                <span className="p-1.5 rounded-lg bg-brand-50 text-brand-500">
                  <RefreshCw size={14} />
                </span>
                继续迭代标注
              </h3>
              <textarea
                value={iterationForm.prompt}
                onChange={(e) => setIterationForm((f) => ({ ...f, prompt: e.target.value }))}
                disabled={iterationSubmitting}
                rows={2}
                className={`${inputCls} resize-none`}
              />
              <div className="flex flex-col gap-2">
                <span className="text-xs text-ink-500">本轮处理范围</span>
                {ITERATION_SELECTIONS.map((opt) => (
                  <label key={opt.key} className="flex items-center gap-2 text-sm text-ink-700">
                    <input
                      type="radio"
                      checked={iterationForm.selectionMode === opt.key}
                      onChange={() => setIterationForm((f) => ({ ...f, selectionMode: opt.key }))}
                      disabled={iterationSubmitting}
                    />
                    {opt.label}
                  </label>
                ))}
              </div>
              <div className="flex flex-col gap-2">
                <span className="text-xs text-ink-500">写入策略</span>
                {WRITE_POLICIES.map((opt) => (
                  <label key={opt.key} className="flex items-center gap-2 text-sm text-ink-700">
                    <input
                      type="radio"
                      checked={iterationForm.writePolicy === opt.key}
                      onChange={() => setIterationForm((f) => ({ ...f, writePolicy: opt.key }))}
                      disabled={iterationSubmitting}
                    />
                    {opt.label}
                  </label>
                ))}
              </div>
              <div className="flex flex-col gap-2">
                <span className="text-xs text-ink-500">辅助模型（可选）</span>
                <select
                  value={iterationForm.assistModelId || ''}
                  onChange={(e) => setIterationForm((f) => ({ ...f, assistModelId: e.target.value }))}
                  disabled={iterationSubmitting}
                  className="px-3 py-2 border border-ink-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
                >
                  <option value="">不使用辅助模型（纯LLM标注）</option>
                  {trainedModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} {m.version > 1 ? `v${m.version}` : ''}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-ink-500">
                  选择已训练模型可先用YOLO预标注，再由LLM修正和补充，提升效率和准确度。
                </p>
              </div>
              <div className="flex gap-2">
                <button
                  type="submit"
                  disabled={iterationSubmitting || !iterationForm.prompt.trim()}
                  className="flex-1 inline-flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-brand-500 text-white font-medium disabled:bg-ink-100 disabled:text-ink-400"
                >
                  {iterationSubmitting ? <Loader2 size={15} className="animate-spin" /> : <ScanLine size={15} />}
                  开始下一轮
                </button>
                <button
                  type="button"
                  onClick={() => setIterationForm(null)}
                  disabled={iterationSubmitting}
                  className="px-3 py-2 rounded-xl border border-ink-200 text-ink-500 hover:bg-ink-50"
                >
                  取消
                </button>
              </div>
            </form>
          )}

          {trainConfirm && (
            <form onSubmit={handleStartTraining} className="card p-5 flex flex-col gap-4 border-emerald-100">
              <h3 className="font-semibold text-ink-900 flex items-center gap-2">
                <span className="p-1.5 rounded-lg bg-emerald-50 text-emerald-600">
                  <Rocket size={14} />
                </span>
                确认训练「{trainConfirm.categoryName}」
              </h3>
              <p className="text-xs text-ink-500">训练会占用 GPU；确认参数后将使用当前最新标注快照启动 YOLO11 训练。</p>
              <div className="grid grid-cols-3 gap-2">
                {[
                  ['epochs', '轮数', 1, 1000, 1],
                  ['imgsz', '尺寸', 64, 2048, 32],
                  ['batch', '批量', -1, 256, 1],
                ].map(([key, label, min, max, step]) => (
                  <label key={key} className="flex flex-col gap-1 text-xs text-ink-500">
                    {label}
                    <input
                      type="number"
                      value={trainParams[key]}
                      min={min}
                      max={max}
                      step={step}
                      disabled={trainingStarting}
                      onChange={(e) => setTrainParams((p) => ({ ...p, [key]: Number(e.target.value) }))}
                      className="px-2 py-1.5 rounded-lg border border-ink-200 text-sm text-ink-800"
                    />
                  </label>
                ))}
              </div>
              <div className="flex gap-2">
                <button
                  type="submit"
                  disabled={trainingStarting}
                  className="flex-1 inline-flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-emerald-600 text-white font-medium disabled:bg-ink-100 disabled:text-ink-400"
                >
                  {trainingStarting ? <Loader2 size={15} className="animate-spin" /> : <Rocket size={15} />}
                  开始训练
                </button>
                <button
                  type="button"
                  onClick={() => setTrainConfirm(null)}
                  disabled={trainingStarting}
                  className="px-3 py-2 rounded-xl border border-ink-200 text-ink-500 hover:bg-ink-50"
                >
                  取消
                </button>
              </div>
            </form>
          )}

          {/* 后端配置摘要；密钥配置不代表模型服务已通过健康检查。 */}
          {config && (
            <div className="card p-4 text-sm">
              <h4 className="text-ink-500 font-medium mb-2 text-xs uppercase tracking-wider">
                标注服务
              </h4>
              <div className="space-y-1.5 text-ink-700">
                <div className="flex justify-between">
                  <span className="text-ink-500">当前模型</span>
                  <span className="font-mono text-xs text-ink-700 truncate max-w-[12rem]" title={config.model}>{config.model}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-ink-500">配置状态</span>
                  <span className={`font-medium ${config.configured ? 'text-emerald-600' : 'text-amber-600'}`}>
                    {config.configured ? '访问密钥已配置' : '访问密钥未配置'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </aside>

        {/* ── 右侧：任务列表 ─────────────────────────────────────── */}
        <section className="min-w-0">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-ink-900">标注任务</h3>
            <button
              type="button"
              onClick={loadTasks}
              className="text-sm text-brand-600 hover:text-brand-700 flex items-center gap-1"
            >
              <RefreshCw size={14} />
              刷新
            </button>
          </div>

          {notice && (
            <div className="mb-4 rounded-xl bg-emerald-50 border border-emerald-200 px-3 py-2.5 text-sm text-emerald-800 flex items-start gap-2">
              <CheckCircle2 size={16} className="text-emerald-500 shrink-0 mt-0.5" />
              <span>{notice}</span>
            </div>
          )}

          {tasks.length === 0 ? (
            <WorkspaceEmpty
              label="样本工场 / 标注任务"
              title="从一句描述，积累训练样本"
              description="描述需要标注的目标，导入图片或视频。标注完成后可人工检查、继续迭代，或进入专用模型训练。"
              tags={['语义标注', '人工校验', '数据持续沉淀']}
            />
          ) : (
            <div className="flex flex-col gap-4">
              {tasks.map((task) => {
                const cat = categories.find((c) => c.id === task.category_id)
                const { display: catName, ts: catTs } = splitTimestampName(
                  cat?.name || task.category_id,
                )
                const pct = Math.round((task.progress || 0) * 100)
                const cancellable = task.status === 'running' || task.status === 'pending'
                // 失败 / 已取消 / 完成但未识别到任何目标 → 可删除并清理未成功标注的数据。
                const deletable =
                  task.status === 'failed' ||
                  task.status === 'cancelled' ||
                  (task.status === 'finished' && (task.success_count || 0) === 0)
                // 视频抽帧阶段：任务还是 pending 且尚无图片数。
                const extracting = task.status === 'pending' && (task.total_images || 0) === 0
                return (
                  <div key={task.id} className="card p-4">
                    <div className="flex items-start justify-between gap-3 mb-2">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <h4 className="font-medium text-ink-900 truncate">{catName}</h4>
                          {catTs && (
                            <span className="text-[11px] text-ink-400">{formatTs(catTs)}</span>
                          )}
                          <StatusBadge status={task.status} extracting={extracting} />
                        </div>
                        <p className="text-sm text-ink-600 truncate" title={task.prompt}>
                          标注要求：{task.prompt}
                        </p>
                        <p className="text-xs text-ink-400 mt-1">视觉模型标注</p>
                      </div>
                      {cancellable ? (
                        <button
                          type="button"
                          onClick={() => handleCancel(task.id)}
                          className="shrink-0 text-xs text-red-500 hover:text-red-600 flex items-center gap-1 px-2 py-1 border border-red-200 rounded-lg"
                        >
                          <XCircle size={12} />
                          取消
                        </button>
                      ) : deletable ? (
                        <button
                          type="button"
                          onClick={() => handleDelete(task)}
                          className="shrink-0 text-xs text-ink-400 hover:text-red-600 flex items-center gap-1 px-2 py-1 border border-ink-200 hover:border-red-200 rounded-lg"
                        >
                          <Trash2 size={12} />
                          删除
                        </button>
                      ) : null}
                    </div>

                    {/* 准备/抽帧中（进度未知） */}
                    {task.status === 'pending' && (
                      <div className="mt-3 flex items-center gap-2 text-sm text-brand-600">
                        <Loader2 size={14} className="animate-spin" />
                        {extracting ? '正在从视频抽取画面…' : '准备中…'}
                      </div>
                    )}

                    {/* 标注进度 */}
                    {task.status === 'running' && (
                      <div className="mt-3">
                        <div className="flex items-center justify-between text-xs text-ink-500 mb-1">
                          <span>{task.processed_images} / {task.total_images}</span>
                          <span>{pct}%</span>
                        </div>
                        <div className="w-full bg-ink-100 rounded-full h-2 overflow-hidden">
                          <div
                            className="bg-brand-500 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {/* 完成统计 + 去训练 */}
                    {task.status === 'finished' && (
                      <div className="mt-3 flex flex-col gap-3">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm bg-emerald-50 rounded-xl p-3 text-ink-700">
                          <div className="flex items-center gap-1">
                            <CheckCircle2 size={14} className="text-emerald-500" />
                            检测成功 {task.success_count}
                          </div>
                          <div>未检测到目标 {task.no_target_count || 0}</div>
                          <div>检测失败 {task.failed_count}</div>
                          <div>标注框 {task.total_boxes}</div>
                        </div>
                        {task.success_count > 0 ? (
                          <div className="flex flex-col gap-2">
                            <p className="text-xs text-ink-500">
                              标注已并入「{catName}」，可以继续迭代、人工检查，或确认后训练。
                            </p>
                            <div className="flex items-center gap-2 flex-wrap">
                              <button
                                type="button"
                                onClick={() => openIterationForm(task)}
                                className="inline-flex items-center gap-1.5 text-sm font-medium text-brand-600 hover:text-white hover:bg-brand-500 border border-brand-200 hover:border-brand-500 rounded-lg px-3 py-1.5 transition-colors"
                              >
                                <RefreshCw size={14} />
                                继续迭代标注
                              </button>
                              {(task.failed_count || 0) > 0 && (
                                <button
                                  type="button"
                                  onClick={() => handleRetryFailedImages(task)}
                                  disabled={retryingFailedTaskId === task.id}
                                  className="inline-flex items-center gap-1.5 text-sm font-medium text-red-600 hover:text-white hover:bg-red-500 border border-red-200 hover:border-red-500 rounded-lg px-3 py-1.5 transition-colors disabled:opacity-60"
                                >
                                  {retryingFailedTaskId === task.id ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                                  重新检测失败图片
                                </button>
                              )}
                              <button
                                type="button"
                                onClick={() => handleGoTrain(task.category_id)}
                                className="inline-flex items-center gap-1.5 text-sm font-medium text-ink-600 hover:text-ink-900 border border-ink-200 hover:bg-ink-50 rounded-lg px-3 py-1.5 transition-colors"
                              >
                                <ImageIcon size={14} />
                                人工检查/修正
                              </button>
                              <button
                                type="button"
                                onClick={() => openTrainConfirm(task)}
                                className="inline-flex items-center gap-1.5 text-sm font-medium text-emerald-700 hover:text-white hover:bg-emerald-600 border border-emerald-200 hover:border-emerald-600 rounded-lg px-3 py-1.5 transition-colors"
                              >
                                <Rocket size={14} />
                                完成标注，开始训练
                              </button>
                            </div>
                          </div>
                        ) : (
                          <div className="flex flex-col gap-2">
                            <p className="text-xs text-amber-600">
                              未能从素材中识别到目标，请调整描述或更换素材后重试。
                            </p>
                            {(task.failed_count || 0) > 0 && (
                              <button
                                type="button"
                                onClick={() => handleRetryFailedImages(task)}
                                disabled={retryingFailedTaskId === task.id}
                                className="self-start inline-flex items-center gap-1.5 text-sm font-medium text-red-600 hover:text-white hover:bg-red-500 border border-red-200 hover:border-red-500 rounded-lg px-3 py-1.5 transition-colors disabled:opacity-60"
                              >
                                {retryingFailedTaskId === task.id ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                                重新检测失败图片
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    )}

                    {/* 错误信息 */}
                    {task.error && (
                      <div className="mt-3 rounded-xl bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700 break-all">
                        {task.error}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </section>
      </div>
    </>
  )
}
