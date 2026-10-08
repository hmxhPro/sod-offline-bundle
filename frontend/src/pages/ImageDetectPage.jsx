/**
 * src/pages/ImageDetectPage.jsx
 * ------------------------------
 * 图片检测页面：直接上传图片并检测，无需视频转换。
 * 支持两种模式：
 *   1. 使用已训练模型（model_id）
 *   2. 零样本检测（class_names）
 */

import React, { useState, useCallback, useEffect } from 'react'
import { Image as ImageIcon, Play, Trash2, Download, AlertCircle, Loader2 } from 'lucide-react'

import ImageUploader from '../components/ImageUploader'
import DetectModelSelect from '../components/DetectModelSelect'
import PageHeader from '../components/PageHeader'
import WorkspaceEmpty, { WorkspaceSteps } from '../components/WorkspaceEmpty'
import { detectImages, mediaUrl, getAnnotationModels } from '../services/api'

export default function ImageDetectPage() {
  // 上传的图片文件列表
  const [files, setFiles] = useState([])
  // 检测方式：'yolo' 或 'llm'
  const [detectionMode, setDetectionMode] = useState('yolo')
  // YOLO 模式状态
  const [selectedModelId, setSelectedModelId] = useState('')
  const [selectedModel, setSelectedModel] = useState(null)
  // 零样本检测的类别名称（逗号分隔）- 已弃用，用 LLM 替代
  const [classNames, setClassNames] = useState('')
  // LLM 模式状态
  const [llmPrompt, setLlmPrompt] = useState('')
  const [llmModel, setLlmModel] = useState('')
  const [llmModels, setLlmModels] = useState([])
  const [loadingLlmModels, setLoadingLlmModels] = useState(false)
  // 置信度阈值
  const [confidence, setConfidence] = useState(0.25)
  // 检测状态
  const [detecting, setDetecting] = useState(false)
  const [progress, setProgress] = useState(0)
  // 检测结果
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

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

  const handleFilesSelected = useCallback((newFiles) => {
    setFiles((prev) => [...prev, ...newFiles])
    // 清除之前的结果
    setResults(null)
    setError(null)
  }, [])

  const handleClearAll = useCallback(() => {
    setFiles([])
    setResults(null)
    setError(null)
  }, [])

  const handleSelectModel = useCallback((id, model) => {
    setSelectedModelId(id)
    setSelectedModel(model)
    // 清除结果
    setResults(null)
    setError(null)
  }, [])

  const handleDetect = async () => {
    if (files.length === 0) {
      alert('请先上传图片')
      return
    }

    // 验证参数
    if (detectionMode === 'yolo') {
      if (!selectedModelId) {
        alert('请选择已训练模型')
        return
      }
    } else if (detectionMode === 'llm') {
      if (!llmPrompt.trim()) {
        alert('请输入检测目标描述')
        return
      }
    }

    setDetecting(true)
    setProgress(0)
    setError(null)
    setResults(null)

    try {
      const opts = detectionMode === 'llm'
        ? {
            detectionMode: 'llm',
            llmPrompt: llmPrompt.trim(),
            llmModel: llmModel,
            conf: confidence
          }
        : {
            detectionMode: 'yolo',
            modelId: selectedModelId,
            conf: confidence
          }

      const data = await detectImages(files, opts, (pct) => setProgress(pct))
      setResults(data)
    } catch (err) {
      setError(err?.message || '检测失败')
    } finally {
      setDetecting(false)
      setProgress(0)
    }
  }

  const canDetect = files.length > 0 &&
    (detectionMode === 'yolo' ? selectedModelId : llmPrompt.trim()) &&
    !detecting
  const hasResults = results && results.results && results.results.length > 0

  return (
    <>
      <PageHeader
        title="图片检测"
        subtitle="用一句描述定义目标，让静态画面成为可分析的数据。"
        icon={<ImageIcon size={18} />}
        right={
          <span className="hidden md:inline-flex items-center gap-2 text-xs text-brand-600 bg-brand-50 border border-brand-100 px-3 py-1 rounded-full">
            <ImageIcon size={12} />
            批量检测
          </span>
        }
      />

      <WorkspaceSteps steps={['导入图片', '选择模型与目标', '分析检测结果', '保存标注图片']} />
      <div className="workspace-grid">
        {/* ── 左侧：配置区 ──────────────────────────────────────── */}
        <aside className="flex flex-col gap-5">
          <div className="card p-5 flex flex-col gap-5">
            {/* 标题 */}
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-ink-900 flex items-center gap-2">
                <span className="p-1.5 rounded-lg bg-brand-50 text-brand-500">
                  <ImageIcon size={14} />
                </span>
                <span className="workspace-section-index">01</span> 任务配置
              </h3>
              {files.length > 0 && (
                <button
                  type="button"
                  onClick={handleClearAll}
                  disabled={detecting}
                  className={[
                    'text-xs flex items-center gap-1',
                    detecting ? 'text-ink-300 cursor-not-allowed' : 'text-ink-500 hover:text-red-500',
                  ].join(' ')}
                >
                  <Trash2 size={12} />
                  清空
                </button>
              )}
            </div>

            {/* 图片上传器 */}
            <ImageUploader
              onFilesSelected={handleFilesSelected}
              disabled={detecting}
              hasItems={files.length > 0}
              hint={files.length > 0 ? `已选择 ${files.length} 张图片` : undefined}
            />

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
                    disabled={detecting}
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
                    disabled={detecting}
                    className="w-4 h-4 text-brand-500 focus:ring-brand-500"
                  />
                  <span className="text-sm text-ink-700">自然语言检测</span>
                </label>
              </div>
            </div>

            {/* YOLO 模式：模型选择 */}
            {detectionMode === 'yolo' && (
              <DetectModelSelect
                value={selectedModelId}
                onChange={handleSelectModel}
                disabled={detecting}
              />
            )}

            {/* LLM 模式：提示词和模型选择 */}
            {detectionMode === 'llm' && (
              <>
                <div className="flex flex-col gap-2">
                  <label className="text-ink-700 text-sm font-medium">检测目标</label>
                  <textarea
                    value={llmPrompt}
                    onChange={(e) => setLlmPrompt(e.target.value)}
                    disabled={detecting}
                    rows={2}
                    placeholder="例如：画面中的红色安全帽"
                    className={[
                      'w-full rounded-xl px-4 py-2.5 text-ink-800 placeholder-ink-400 resize-none',
                      'bg-surface border border-ink-200',
                      'focus:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-100',
                      'transition-colors duration-200',
                      detecting ? 'opacity-60 cursor-not-allowed' : '',
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
                    disabled={detecting || loadingLlmModels}
                    className={[
                      'w-full rounded-xl px-4 py-2.5 text-ink-800',
                      'bg-surface border border-ink-200',
                      'focus:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-100',
                      'transition-colors duration-200',
                      detecting ? 'opacity-60 cursor-not-allowed' : '',
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

            {/* 置信度阈值 */}
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <label className="text-ink-700 text-sm font-medium">置信度阈值</label>
                <span className="text-brand-600 font-mono text-sm">{confidence.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                disabled={detecting}
                className="w-full accent-brand-500"
              />
              <p className="text-xs text-ink-500">
                值越高过滤越严格，漏检越多；值越低误检越多
              </p>
            </div>

            {/* 开始检测按钮 */}
            <button
              type="button"
              onClick={handleDetect}
              disabled={!canDetect}
              className={[
                'w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-semibold transition-all duration-200',
                canDetect
                  ? 'bg-brand-500 text-white hover:bg-brand-600 shadow-brand'
                  : 'bg-ink-100 text-ink-400 cursor-not-allowed',
              ].join(' ')}
            >
              <Play size={16} />
              {detecting ? (progress < 100 ? `上传中 ${progress}%` : '正在检测…') : `开始检测（${files.length} 张）`}
            </button>
          </div>

          {/* 配置摘要 */}
          {(selectedModel || classNames.trim()) && (
            <div className="card p-4 text-sm">
              <h4 className="text-ink-500 font-medium mb-2 text-xs uppercase tracking-wider">当前配置</h4>
              <div className="space-y-1.5 text-ink-700">
                <div className="flex justify-between">
                  <span className="text-ink-500">检测模式</span>
                  <span className="font-medium">{selectedModel ? '已训练模型' : '零样本'}</span>
                </div>
                {selectedModel ? (
                  <>
                    <div className="flex justify-between">
                      <span className="text-ink-500">模型名称</span>
                      <span className="font-medium">{selectedModel.name}</span>
                    </div>
                    {selectedModel.class_names && (
                      <div className="flex flex-col gap-1">
                        <span className="text-ink-500">检测类别</span>
                        <span className="font-medium text-xs">
                          {Object.values(selectedModel.class_names).join('、')}
                        </span>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="flex flex-col gap-1">
                    <span className="text-ink-500">检测类别</span>
                    <span className="font-medium text-xs">{classNames}</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-ink-500">置信度</span>
                  <span className="font-medium">{confidence.toFixed(2)}</span>
                </div>
              </div>
            </div>
          )}
        </aside>

        {/* ── 右侧：结果展示区 ──────────────────────────────────── */}
        <section className="min-w-0">
          {/* 错误提示 */}
          {error && (
            <div className="card p-4 mb-4 bg-red-50 border-red-200 flex items-start gap-3">
              <AlertCircle size={18} className="text-red-500 shrink-0 mt-0.5" />
              <div>
                <p className="text-red-800 font-medium">检测失败</p>
                <p className="text-red-600 text-sm mt-1">{error}</p>
              </div>
            </div>
          )}

          {/* 空状态 */}
          {!hasResults && !detecting && !error && (
            <WorkspaceEmpty
              label="图像感知 / 结果预览"
              stateLabel={files.length > 0 ? '等待开始' : '等待输入'}
              title={files.length > 0 ? '素材已选定，等待开始' : '让目标，从画面中浮现'}
              description={files.length > 0
                ? `已选择 ${files.length} 张图片。确认检测方式与参数后，点击开始检测。`
                : '导入图片，选择专用模型或用自然语言描述目标，在这里查看目标位置与置信度。'}
              tags={['批量图片', '语义定位', '标注导出']}
            />
          )}

          {detecting && (
            <div className="card p-8 flex items-center gap-4" role="status" aria-live="polite">
              <Loader2 size={22} className="text-brand-500 animate-spin shrink-0" />
              <div>
                <h3 className="font-medium text-ink-800">正在处理图片</h3>
                <p className="text-sm text-ink-500 mt-1">{progress < 100 ? `素材上传进度 ${progress}%` : '素材已上传，正在等待检测结果…'}</p>
              </div>
            </div>
          )}

          {/* 检测结果网格 */}
          {hasResults && (
            <div className="space-y-4">
              {/* 结果摘要 */}
              <div className="card p-4 bg-emerald-50 border-emerald-200">
                <p className="text-emerald-800 font-medium">
                  ✓ 检测完成：共处理 {results.results.length} 张图片
                  {results.class_names && results.class_names.length > 0 && (
                    <span className="text-emerald-600 ml-2">
                      · 类别：{results.class_names.join('、')}
                    </span>
                  )}
                </p>
              </div>

              {/* 图片结果卡片 */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {results.results.map((item) => (
                  <div key={item.image_index} className="card overflow-hidden">
                    {/* 图片 */}
                    <div className="relative bg-ink-900">
                      <img
                        src={mediaUrl(item.annotated_url)}
                        alt={item.filename}
                        className="w-full h-auto"
                      />
                      <div className="absolute top-2 right-2">
                        <a
                          href={mediaUrl(item.annotated_url)}
                          download={item.filename}
                          className="p-2 rounded-lg bg-black/50 hover:bg-black/70 text-white transition-colors"
                          title="下载标注图片"
                        >
                          <Download size={16} />
                        </a>
                      </div>
                    </div>

                    {/* 信息 */}
                    <div className="p-4">
                      <h4 className="font-medium text-ink-800 mb-2 truncate" title={item.filename}>
                        {item.filename}
                      </h4>
                      <div className="text-xs text-ink-500 mb-3">
                        {item.width} × {item.height} · {item.detections.length} 个目标
                      </div>

                      {/* 检测目标列表 */}
                      {item.detections.length > 0 ? (
                        <div className="space-y-1.5 max-h-32 overflow-y-auto">
                          {item.detections.map((det, idx) => (
                            <div
                              key={idx}
                              className="flex items-center justify-between text-sm bg-ink-50 rounded-lg px-3 py-1.5"
                            >
                              <span className="text-ink-700 font-medium">{det.class}</span>
                              <span className="text-ink-500 font-mono text-xs">
                                {(det.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-ink-400 text-sm">未检测到目标</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </>
  )
}
