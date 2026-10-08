import React from 'react'
import { Link } from 'react-router-dom'
import { LayoutDashboard, Film, Image, ScanLine, Layers3, History, Sun, Moon, ArrowUpRight, X } from 'lucide-react'
import NavItem from './NavItem'
import { useTheme } from '../hooks/useTheme'

export default function Sidebar({ open, onClose, service }) {
  const { theme, toggle } = useTheme()
  return (
    <>
      {open && <button className="sidebar-backdrop" aria-label="关闭导航菜单" onClick={onClose} />}
      <aside id="workspace-navigation" className={`workspace-sidebar ${open ? 'is-open' : ''}`}>
        <Link to="/" className="brand-lockup" aria-label="智瞰万象 · 感知总览" onClick={onClose}>
          <svg className="brand-symbol" viewBox="0 0 40 40" fill="none" aria-hidden="true">
            <path d="M3 14V5h9M28 5h9v9M37 26v9h-9M12 35H3v-9" stroke="currentColor" strokeWidth="2" />
            <path d="M8 20s5-8 12-8 12 8 12 8-5 8-12 8S8 20 8 20Z" stroke="currentColor" strokeWidth="1.5" />
            <circle cx="20" cy="20" r="4" fill="currentColor" />
          </svg>
          <span><strong>智瞰万象</strong><small>ZHIKAN · VISION</small></span>
        </Link>
        <button type="button" className="mobile-nav-close" aria-label="关闭导航菜单" onClick={onClose}><X size={19} /></button>
        <div className="workspace-tag"><span /> 开放世界智能视觉平台</div>
        <nav className="workspace-nav" aria-label="主导航">
          <span className="nav-section-label">WORKSPACE <span>工作空间</span></span>
          <NavItem to="/" end icon={<LayoutDashboard size={18} />}>感知总览</NavItem>
          <NavItem to="/detect" icon={<Film size={18} />}>视频检测</NavItem>
          <NavItem to="/image-detect" icon={<Image size={18} />}>图片检测</NavItem>
          <span className="nav-section-label nav-section-second">DATA & MODEL <span>数据与模型</span></span>
          <NavItem to="/llm-annotation" icon={<ScanLine size={18} />}>智能标注</NavItem>
          <NavItem to="/training" icon={<Layers3 size={18} />}>模型训练</NavItem>
          <NavItem to="/history" icon={<History size={18} />}>任务档案</NavItem>
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-note"><span className="mono-caption">PERCEPTION / TRAINING LOOP</span><p>语言定义目标，<br />检测数据反哺训练。</p><ArrowUpRight size={18} /></div>
          <div className="sidebar-service"><span className={`service-dot ${service.kind}`} /><span>{service.label}</span></div>
          <button type="button" onClick={toggle} className="theme-switch" aria-label={theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'}>
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}<span>{theme === 'dark' ? '浅色外观' : '深色外观'}</span><span className="theme-switch-track"><span /></span>
          </button>
          <span className="sidebar-version">智瞰万象 <span>VISION WORKSPACE / 01</span></span>
        </div>
      </aside>
    </>
  )
}
