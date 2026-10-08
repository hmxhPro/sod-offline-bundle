/**
 * src/layout/ConsoleLayout.jsx
 * ----------------------------
 * The application shell AND the single host for cross-page state.
 *
 * It stays mounted for the whole session, so detection SSE streams and
 * uploaded-task state survive navigation between pages (react-router only swaps
 * the <Outlet> child, never this layout). All detection/training state lives
 * here and is handed to pages via <Outlet context={...}/>; pages read it with
 * useOutletContext().
 */

import React, { useState, useEffect } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { Menu, ChevronRight, Command } from 'lucide-react'
import Sidebar from './Sidebar'
import { useServiceStatus } from '../hooks/useServiceStatus'
import { useDetectionTasks } from '../hooks/useDetectionTasks'
import { getCategory, getAnnotationModels } from '../services/api'

export default function ConsoleLayout() {
  const [navigationOpen, setNavigationOpen] = useState(false)
  const location = useLocation()
  const service = useServiceStatus()
  const pageNames = { '/': '感知总览', '/detect': '视频检测', '/image-detect': '图片检测', '/training': '模型训练', '/llm-annotation': '智能标注', '/history': '任务档案' }
  useEffect(() => { setNavigationOpen(false) }, [location.pathname])
  useEffect(() => {
    if (!navigationOpen) return
    const closeOnEscape = (event) => { if (event.key === 'Escape') setNavigationOpen(false) }
    document.addEventListener('keydown', closeOnEscape)
    return () => document.removeEventListener('keydown', closeOnEscape)
  }, [navigationOpen])
  // ── Detection workspace config ─────────────────────────────────────────
  const [prompt, setPrompt] = useState('')
  const [detInterval, setDetInterval] = useState(5)
  const [enableVlm, setEnableVlm] = useState(true)
  const [showAdvanced, setShowAdvanced] = useState(false)
  // Trained-model selection for the detection workspace ('' = open-vocab).
  const [selectedModelId, setSelectedModelId] = useState('')
  // Detection mode: 'yolo' or 'llm'
  const [detectionMode, setDetectionMode] = useState('yolo')
  // LLM detection state
  const [llmPrompt, setLlmPrompt] = useState('')
  const [llmModel, setLlmModel] = useState('')
  const [llmModels, setLlmModels] = useState([])
  const [loadingLlmModels, setLoadingLlmModels] = useState(false)

  // ── Training workspace state ───────────────────────────────────────────
  const [selectedCategory, setSelectedCategory] = useState(null)
  const [catReloadToken, setCatReloadToken] = useState(0)
  // Shared across pages: bumped after a training run AND read by
  // DetectModelSelect, so a model trained on /training shows up on /detect.
  const [modelReloadToken, setModelReloadToken] = useState(0)
  // Dataset source for the selected category: online annotation vs. import.
  const [trainTab, setTrainTab] = useState('annotate')

  // 加载可用的 LLM 模型列表
  useEffect(() => {
    const loadLlmModels = async () => {
      setLoadingLlmModels(true)
      try {
        const response = await getAnnotationModels()
        // API 返回 {provider, current_model, models: string[], all_models_count}
        const modelList = Array.isArray(response?.models) ? response.models : []
        setLlmModels(modelList)
        if (modelList.length > 0) {
          setLlmModel(response.current_model || modelList[0])
        }
      } catch (err) {
        console.error('加载 LLM 模型列表失败:', err)
        setLlmModels([])
      } finally {
        setLoadingLlmModels(false)
      }
    }
    loadLlmModels()
  }, [])

  // Instantiated exactly ONCE here (never unmounts on nav) so in-flight SSE
  // streams keep running while the user is on another page.
  const detection = useDetectionTasks()

  // Selecting a trained model defaults VLM off (the model is self-sufficient);
  // it stays user-toggleable in advanced settings. Clearing restores default.
  const handleSelectModel = (id) => {
    setSelectedModelId(id)
    setEnableVlm(!id)
  }

  const handleStartAll = async () => {
    await detection.startAll(prompt.trim(), detInterval, enableVlm, selectedModelId || undefined)
  }

  const handleRetry = (id) => {
    if (!selectedModelId && !prompt.trim()) {
      alert('请先填写检测目标或选择已训练模型')
      return
    }
    detection.resetOne(id)
    // Give state a tick before restarting.
    setTimeout(
      () => detection.startOne(id, prompt.trim(), detInterval, enableVlm, selectedModelId || undefined),
      30
    )
  }

  // Re-fetch the selected category so TrainPanel sees fresh annotated/image
  // counts, and nudge the category list to refresh its own counts.
  const refreshSelectedCategory = async () => {
    if (!selectedCategory) return
    try {
      setSelectedCategory(await getCategory(selectedCategory.id))
    } catch {
      /* keep stale copy on transient failure */
    }
    setCatReloadToken((t) => t + 1)
  }

  // After a training run finishes: reload the model list + refresh counts.
  const handleTrained = () => {
    setModelReloadToken((t) => t + 1)
    refreshSelectedCategory()
  }

  // Single context object handed down to every routed page. Rebuilt each render
  // (useDetectionTasks returns a fresh object anyway) — page re-render scope is
  // the same as the old single-component App.
  const ctx = {
    // detection — state + hook actions
    ...detection,
    prompt, setPrompt, detInterval, setDetInterval, enableVlm, setEnableVlm,
    showAdvanced, setShowAdvanced, selectedModelId, setSelectedModelId,
    detectionMode, setDetectionMode, llmPrompt, setLlmPrompt, llmModel, setLlmModel,
    llmModels, loadingLlmModels,
    handleSelectModel, handleStartAll, handleRetry,
    // training — state + actions
    selectedCategory, setSelectedCategory, catReloadToken, setCatReloadToken,
    modelReloadToken, setModelReloadToken, trainTab, setTrainTab,
    refreshSelectedCategory, handleTrained,
  }

  return (
    <div className="workspace-shell">
      <a className="skip-link" href="#workspace-main">跳转到主要内容</a>
      <Sidebar open={navigationOpen} onClose={() => setNavigationOpen(false)} service={service} />
      <main id="workspace-main" className="workspace-main" tabIndex={-1}>
        <header className="workspace-topbar">
          <div className="workspace-breadcrumb">
            <button type="button" className="mobile-nav-toggle" aria-label="打开导航菜单" aria-expanded={navigationOpen} aria-controls="workspace-navigation" onClick={() => setNavigationOpen(true)}><Menu size={20} /></button>
            <Command size={15} className="breadcrumb-icon" /><span className="breadcrumb-root">工作空间</span><ChevronRight size={13} /><strong>{pageNames[location.pathname] || '工作空间'}</strong>
          </div>
          <div className="topbar-status"><span className={`service-dot ${service.kind}`} /><span>{service.label}</span><span className="topbar-divider" /><span className="topbar-wordmark">OPEN WORLD VISION</span></div>
        </header>
        <div className="workspace-content">
          <Outlet context={ctx} />
        </div>
        <footer className="workspace-footer"><span>智瞰万象 <span className="footer-separator">/</span> 开放世界智能视觉感知</span><span>感知 · 理解 · 进化</span></footer>
      </main>
    </div>
  )
}
