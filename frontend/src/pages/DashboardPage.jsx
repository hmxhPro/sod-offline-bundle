import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useOutletContext } from 'react-router-dom'
import {
  ArrowRight, ArrowUpRight, ChevronDown, Crosshair,
  Film, Image, Layers3, MoveUpRight, RefreshCw, ScanLine, Terminal,
} from 'lucide-react'
import { getModels, getCategories, getTaskHistory } from '../services/api'
import './dashboard.css'

const WORKSPACES = [
  { to: '/detect', number: '01', icon: Film, title: '视频感知', en: 'VIDEO PERCEPTION', description: '从连续画面中定位目标，关联每一次出现。', detail: '开放词汇检测 / 跨帧跟踪' },
  { to: '/image-detect', number: '02', icon: Image, title: '图像检测', en: 'IMAGE DETECTION', description: '把一句自然语言，变成图像中的目标位置。', detail: '批量图像 / 自然语言目标' },
  { to: '/llm-annotation', number: '03', icon: ScanLine, title: '智能标注', en: 'INTELLIGENT ANNOTATION', description: '让视觉模型辅助标注，为下一次训练积累样本。', detail: '语义理解 / 人工修正' },
  { to: '/training', number: '04', icon: Layers3, title: '专用训练', en: 'SPECIALIZED TRAINING', description: '将场景数据转化为专用模型，回到检测现场。', detail: '数据集管理 / YOLO11 训练' },
]

const WORKFLOW = [
  { title: '定义意图', description: '自然语言描述目标', detail: 'LANGUAGE', to: '/detect' },
  { title: '发现目标', description: '开放词汇快速定位', detail: 'YOLOE · SAHI', to: '/detect' },
  { title: '语义复核', description: '重点复核模糊目标', detail: 'VISION LANGUAGE', to: '/detect' },
  { title: '沉淀数据', description: '标注修正与样本积累', detail: 'ANNOTATION', to: '/llm-annotation' },
  { title: '专用训练', description: '适配自己的业务场景', detail: 'YOLO11', to: '/training' },
  { title: '在线应用', description: '模型回到检测工作台', detail: 'DEPLOY & DETECT', to: '/detect' },
]

// A drawn terrain study, deliberately separate from real detection results.
function PerceptionField() {
  const geometry = useMemo(() => {
    const project = (u, v) => {
      const height =
        1.08 * Math.exp(-((u - 0.18) ** 2 * 10 + (v + 0.12) ** 2 * 7)) +
        0.55 * Math.exp(-((u + 0.62) ** 2 * 15 + (v - 0.24) ** 2 * 9)) +
        0.34 * Math.exp(-((u - 0.7) ** 2 * 17 + (v - 0.55) ** 2 * 12))
      return [310 + u * (215 + v * 52) + v * 25, 255 + v * 127 - height * 130]
    }
    const path = (points) => points.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`).join(' ')
    const rows = Array.from({ length: 36 }, (_, row) => path(Array.from({ length: 65 }, (_, col) => project(-1 + col / 32, -1 + row / 17.5))))
    const cols = Array.from({ length: 39 }, (_, col) => path(Array.from({ length: 65 }, (_, row) => project(-1 + col / 19, -1 + row / 32))))
    return { rows, cols }
  }, [])

  return (
    <div className="zk-field">
      <div className="zk-field-top"><span><Crosshair size={13} /> PERCEPTION FIELD</span><span className="zk-field-demo">感知流程示意</span></div>
      <svg viewBox="0 0 620 440" role="img" aria-labelledby="zk-terrain-title" className="zk-terrain">
        <title id="zk-terrain-title">开放世界视觉感知概念图：地形网格、目标框选与语义关联。非实时检测结果。</title>
        <defs>
          <linearGradient id="zk-terrain-stroke" x1="0" y1="0" x2="0.3" y2="1">
            <stop offset="0%" stopColor="#9ac4b6" stopOpacity="0.15" />
            <stop offset="48%" stopColor="#5ce5bd" stopOpacity="0.66" />
            <stop offset="100%" stopColor="#84b4a5" stopOpacity="0.1" />
          </linearGradient>
          <linearGradient id="zk-scan-glow" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#5ce5bd" stopOpacity="0" />
            <stop offset="100%" stopColor="#5ce5bd" stopOpacity="0.1" />
          </linearGradient>
          <radialGradient id="zk-field-light"><stop offset="0%" stopColor="#285246" stopOpacity="0.22" /><stop offset="100%" stopColor="#0d1515" stopOpacity="0" /></radialGradient>
          <clipPath id="zk-field-clip"><rect x="26" y="52" width="568" height="330" /></clipPath>
        </defs>
        <ellipse cx="328" cy="246" rx="275" ry="180" fill="url(#zk-field-light)" />
        <g fill="none" stroke="#547469" strokeOpacity="0.16">
          <path d="M28 110H592M28 184H592M28 258H592M28 332H592M100 58V382M205 58V382M310 58V382M415 58V382M520 58V382" />
          <path d="M28 60V46H42M578 46H592V60M28 370V384H42M578 384H592V370" stroke="#75988d" strokeOpacity="0.8" />
        </g>
        <g stroke="url(#zk-terrain-stroke)" strokeWidth="0.7" fill="none">{geometry.rows.map((d, i) => <path key={`r${i}`} d={d} />)}{geometry.cols.map((d, i) => <path key={`c${i}`} d={d} />)}</g>
        <g clipPath="url(#zk-field-clip)"><g className="zk-scan-sweep"><rect x="-86" y="55" width="86" height="324" fill="url(#zk-scan-glow)" /><path d="M0 55V379" stroke="#5ce5bd" strokeOpacity="0.48" /></g></g>
        <g className="zk-field-target zk-field-target-primary">
          <rect x="287" y="94" width="105" height="141" fill="#5ce5bd" fillOpacity="0.025" stroke="#5ce5bd" strokeOpacity="0.3" />
          <path d="M287 112V94H305M374 94H392V112M287 217V235H305M374 235H392V217" fill="none" stroke="#72f2cc" strokeWidth="1.7" />
          <path d="M339 158V170M333 164H345" stroke="#9defd7" strokeWidth="1" />
          <path d="M392 100L424 76H500" fill="none" stroke="#79ac9b" strokeWidth="0.8" />
          <text x="428" y="66" fill="#bce6d8" fontSize="10" letterSpacing="1.5">语义定位</text>
          <rect x="287" y="78" width="52" height="16" fill="#5ce5bd" /><text x="294" y="89" fill="#09271d" fontSize="8" letterSpacing="0.8">TARGET A</text>
        </g>
        <g className="zk-field-target zk-field-target-secondary">
          <rect x="121" y="205" width="81" height="78" fill="#dde2d9" fillOpacity="0.02" stroke="#b6c3b9" strokeOpacity="0.27" />
          <path d="M121 219V205H135M188 205H202V219M121 269V283H135M188 283H202V269" fill="none" stroke="#c4d3c9" strokeWidth="1.2" />
          <path d="M122 283L95 305H51" stroke="#789184" strokeWidth="0.8" fill="none" />
          <text x="50" y="323" fill="#afc0b8" fontSize="10" letterSpacing="1">细粒度感知</text>
        </g>
        <g fill="none" stroke="#5ce5bd" strokeOpacity="0.28" strokeDasharray="3 5"><path d="M202 244C249 270 269 249 287 215" /><path d="M392 220C439 225 439 285 482 301" /></g>
        <g stroke="#b6c7bf" strokeOpacity="0.6" fill="none"><circle cx="485" cy="304" r="9" /><path d="M470 304H500M485 289V319" /></g>
        <text x="502" y="329" fill="#809b90" fontSize="9" letterSpacing="1.3">关联</text>
        <g fill="#71877d" fontSize="8" letterSpacing="1.5"><text x="30" y="413">LANGUAGE → VISION</text><text x="478" y="413">OPEN WORLD</text></g>
      </svg>
      <div className="zk-field-caption"><span className="zk-field-dot" /><span>看见目标，理解语义。</span><span>CONCEPT VIEW</span></div>
    </div>
  )
}

function requestWithDeadline(request) {
  let timer
  return Promise.race([request, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('timeout')), 10000) })]).finally(() => clearTimeout(timer))
}

export default function DashboardPage() {
  const { tasks = [], prompt = '', setPrompt, setDetectionMode, handleSelectModel } = useOutletContext()
  const navigate = useNavigate()
  const inputRef = useRef(null)
  const [intent, setIntent] = useState(prompt)
  const [intentError, setIntentError] = useState(false)
  const [counts, setCounts] = useState({ models: null, categories: null, history: null })
  const [loading, setLoading] = useState(true)
  const [reload, setReload] = useState(0)

  useEffect(() => {
    let alive = true
    setLoading(true)
    Promise.allSettled([getModels(), getCategories(), getTaskHistory({ limit: 200 })].map(requestWithDeadline)).then(([models, categories, history]) => {
      if (!alive) return
      const count = (result) => result.status === 'fulfilled' && Array.isArray(result.value) ? result.value.length : null
      setCounts({ models: count(models), categories: count(categories), history: count(history) })
      setLoading(false)
    })
    return () => { alive = false }
  }, [reload])

  const activeCount = tasks.filter((task) => ['uploading', 'pending', 'running', 'paused', 'packaging'].includes(task.taskStatus)).length
  const incomplete = Object.values(counts).some((value) => value === null)
  const display = (value) => loading ? '…' : value ?? '—'
  const beginDetection = (event) => {
    event.preventDefault()
    if (!intent.trim()) {
      setIntentError(true)
      inputRef.current?.focus()
      return
    }
    setPrompt(intent.trim())
    setDetectionMode('yolo')
    handleSelectModel('')
    navigate('/detect')
  }

  return (
    <div className="zk-dashboard">
      <div className="zk-page-intro"><span>工作空间 / 概览</span><span className="zk-mono">ZHIKAN · VISION WORKSPACE</span></div>

      <section className="zk-hero" aria-labelledby="zk-hero-title">
        <div className="zk-hero-copy">
          <div className="zk-eyebrow"><span /> ZHIKAN / 智瞰万象</div>
          <h1 id="zk-hero-title">让视野，<br /><span>不止于已知。</span></h1>
          <p className="zk-hero-platform">智瞰万象——基于大小模型协同的开放世界智能视觉感知平台</p>
          <p className="zk-hero-description">用自然语言定义目标，让大小模型协同理解。<br className="zk-desktop-break" />从一次发现，到属于你的专用视觉模型。</p>
          <form className={`zk-intent-form${intentError ? ' has-error' : ''}`} onSubmit={beginDetection}>
            <label htmlFor="zk-intent">这一次，你想看见什么？</label>
            <div className="zk-intent-control"><Crosshair size={18} aria-hidden="true" /><input ref={inputRef} id="zk-intent" value={intent} onChange={(event) => { setIntent(event.target.value); setIntentError(false) }} placeholder="描述目标，如：行人、车辆、船只" maxLength={500} aria-invalid={intentError} aria-describedby={intentError ? 'zk-intent-error' : 'zk-intent-hint'} /><button type="submit" aria-label="创建视频检测任务"><ArrowUpRight size={22} /></button></div>
            {intentError ? <p id="zk-intent-error" role="alert" className="zk-input-error">请输入要检测的目标，再进入工作台。</p> : <p id="zk-intent-hint" className="zk-intent-hint">输入目标后进入视频工作台，上传素材即可开始。</p>}
          </form>
          <div className="zk-hero-principles"><span>自然语言即任务</span><span>大小模型协同</span><span>检测训练一体化</span></div>
        </div>
        <PerceptionField />
      </section>

      <section className="zk-assets" aria-label="工作空间数据" aria-busy={loading}>
        <div className="zk-asset"><span>本次会话任务</span><strong>{activeCount}<small>进行中</small></strong></div>
        <Link className="zk-asset" to="/history"><span>最近检测任务 <ArrowUpRight size={12} /></span><strong>{display(counts.history)}<small>最多 200 条</small></strong></Link>
        <Link className="zk-asset" to="/training"><span>已注册模型 <ArrowUpRight size={12} /></span><strong>{display(counts.models)}<small>个</small></strong></Link>
        <Link className="zk-asset" to="/training"><span>训练类别 <ArrowUpRight size={12} /></span><strong>{display(counts.categories)}<small>类</small></strong></Link>
        <button className="zk-refresh" type="button" disabled={loading} onClick={() => setReload((value) => value + 1)} aria-label="刷新工作空间数据" title={loading ? '正在获取数据' : incomplete ? '部分数据暂不可用，点击重试' : '刷新工作空间数据'}><RefreshCw size={15} className={loading ? 'zk-is-loading' : ''} /></button>
      </section>
      {!loading && incomplete && <p className="zk-data-note" role="status">部分工作空间数据暂不可用，连接后端服务后可刷新查看。</p>}

      <section className="zk-workspaces" aria-labelledby="zk-workspace-title">
        <div className="zk-section-heading"><div><span className="zk-section-kicker">01 / WORKSPACE</span><h2 id="zk-workspace-title">从这里，开始感知。</h2></div><p>从素材到模型，每一步都在同一工作空间。</p></div>
        <div className="zk-entry-grid">{WORKSPACES.map(({ to, number, icon: Icon, title, en, description, detail }) => <Link className="zk-entry" to={to} key={to}><div className="zk-entry-top"><Icon size={23} strokeWidth={1.4} /><span className="zk-mono">{number}</span></div><div className="zk-entry-title"><h3>{title}</h3><ArrowUpRight size={19} /></div><span className="zk-entry-en">{en}</span><p>{description}</p><div className="zk-entry-footer">{detail}<ArrowRight size={13} /></div></Link>)}</div>
      </section>

      <section className="zk-workflow-section" aria-labelledby="zk-workflow-title">
        <div className="zk-section-heading"><div><span className="zk-section-kicker">02 / CONTINUOUS LEARNING</span><h2 id="zk-workflow-title">每一次发现，都成为下一次的能力。</h2></div><span className="zk-loop-label"><RefreshCw size={13} /> 感知与训练闭环</span></div>
        <ol className="zk-workflow">{WORKFLOW.map((step, index) => <li key={step.title}><Link to={step.to}><span className="zk-step-node">{String(index + 1).padStart(2, '0')}</span><h3>{step.title}</h3><p>{step.description}</p><span className="zk-step-detail">{step.detail}</span></Link></li>)}</ol>
        <div className="zk-workflow-return"><span /><MoveUpRight size={13} /><p>专用模型重新投入检测，场景数据持续积累。</p></div>
      </section>

      <section className="zk-project-notes" aria-label="项目理念与运行环境">
        <details className="zk-project-detail"><summary><span><Crosshair size={17} /><span>为什么是开放世界？<small>DESIGN ORIGIN</small></span></span><ChevronDown size={17} /></summary><div className="zk-detail-content"><p>真实世界里的新目标，不会等模型重新训练后才出现。智瞰万象从这个问题出发，让自然语言成为任务入口：开放词汇模型负责快速定位，视觉大模型对低置信度或语义模糊目标进行重点复核，使检测效率与语义理解相互补充。</p><p>结合 SAHI 小目标增强和 ByteTrack 跨帧关联，将检测数据用于人工修正、智能标注、数据集生成与 YOLO11 专用模型训练，逐步积累特定场景下的识别能力。</p></div></details>
        <details className="zk-project-detail"><summary><span><Terminal size={17} /><span>在你的环境中运行<small>LOCAL DEPLOYMENT</small></span></span><ChevronDown size={17} /></summary><div className="zk-detail-content"><p>采用 B/S 架构，支持 Windows 10/11 与 Ubuntu 22.04/24.04 部署。前端由 React、Vite 和 Tailwind CSS 构建；后端以 Python、FastAPI、PostgreSQL 和 SQLAlchemy 提供检测与数据管理，并通过 Nginx 托管页面、转发接口。</p><p>PyTorch 驱动 YOLOE、SAHI、ByteTrack 与 YOLO11，可使用 NVIDIA GPU 加速。视觉大模型可通过 vLLM 本地部署，也可接入兼容 OpenAI 协议的模型服务。提供离线安装、启停、健康检查与日志管理工具；具体能力取决于实际部署配置。</p></div></details>
      </section>
      <footer className="zk-dashboard-footer"><span>智瞰万象<span className="zk-footer-separator">/</span>开放世界智能视觉感知</span><span className="zk-mono">YOLOE · SAHI · ByteTrack · YOLO11</span></footer>
    </div>
  )
}
