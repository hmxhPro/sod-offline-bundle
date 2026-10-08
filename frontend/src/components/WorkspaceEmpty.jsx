import React, { useId } from 'react'
import '../styles/workspaces.css'

export function WorkspaceSteps({ steps }) {
  return (
    <ol className="workspace-steps" aria-label="操作流程">
      {steps.map((step, index) => (
        <li key={step}>
          <span className="workspace-step-number">{String(index + 1).padStart(2, '0')}</span>
          <span>{step}</span>
        </li>
      ))}
    </ol>
  )
}

export default function WorkspaceEmpty({ title, description, label = '感知画布', stateLabel = '等待输入', tags = [], compact = false, children }) {
  const id = useId().replace(/:/g, '')
  return (
    <div className={`card workspace-empty${compact ? ' workspace-empty--compact' : ''}`}>
      <div className="workspace-empty-topline">
        <span>{label}</span>
        <span className="workspace-empty-state"><i /> {stateLabel}</span>
      </div>
      <div className="workspace-empty-body">
        <svg className="workspace-scope" viewBox="0 0 380 240" fill="none" aria-hidden="true">
          <defs>
            <pattern id={`${id}-grid`} width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" stroke="currentColor" strokeOpacity=".09" strokeWidth=".7" />
            </pattern>
            <radialGradient id={`${id}-glow`}>
              <stop offset="0" stopColor="currentColor" stopOpacity=".12" />
              <stop offset="1" stopColor="currentColor" stopOpacity="0" />
            </radialGradient>
          </defs>
          <ellipse cx="190" cy="120" rx="177" ry="115" fill={`url(#${id}-glow)`} />
          <rect x="40" y="20" width="300" height="200" fill={`url(#${id}-grid)`} />
          <path d="M40 55V20h35M305 20h35v35M340 185v35h-35M75 220H40v-35" stroke="currentColor" strokeOpacity=".48" />
          <path d="M190 8v28m0 168v28M26 120h28m272 0h28" stroke="currentColor" strokeOpacity=".3" />
          <circle cx="190" cy="120" r="76" stroke="currentColor" strokeOpacity=".16" strokeDasharray="3 7" />
          <circle cx="190" cy="120" r="49" stroke="currentColor" strokeOpacity=".12" />
          <path d="M140 94v-24h24m52 0h24v24m0 52v24h-24m-52 0h-24v-24" stroke="currentColor" strokeWidth="1.6" />
          <path d="M176 120h28m-14-14v28" stroke="currentColor" strokeOpacity=".7" />
          <circle cx="190" cy="120" r="3" fill="currentColor" />
          <path className="workspace-scope-scan" d="M54 120h272" stroke="currentColor" strokeOpacity=".38" />
          <path d="M73 189h35m-35 5h22M272 47h35m-22 5h22" stroke="currentColor" strokeOpacity=".23" />
        </svg>
        <div className="workspace-empty-copy">
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
        {tags.length > 0 && (
          <div className="workspace-empty-tags">
            {tags.map((tag) => <span key={tag}>{tag}</span>)}
          </div>
        )}
        {children}
      </div>
      <div className="workspace-empty-baseline" aria-hidden="true"><span /> <span /> <span /></div>
    </div>
  )
}
