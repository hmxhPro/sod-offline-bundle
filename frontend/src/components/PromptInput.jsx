/**
 * src/components/PromptInput.jsx
 * -------------------------------
 * Text area + example chips for target noun input.
 * Uses the shared theme palette for the target prompt and examples.
 */

import React from 'react'
import { Search } from 'lucide-react'

const EXAMPLES = [
  '钓鱼台,菜地,水塘',
  '鱼排,增氧机,泵房',
  '太阳能板,大棚,违建',
  '小船,车辆,房屋',
  '围网,地笼,小型浮台',
]

export default function PromptInput({ value, onChange, disabled }) {
  return (
    <div className="flex flex-col gap-3">
      <label className="text-ink-800 font-semibold flex items-center gap-2">
        <span className="p-1.5 rounded-lg bg-brand-50 text-brand-500">
          <Search size={14} />
        </span>
        检测目标
      </label>

      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        rows={2}
        placeholder="用逗号分隔目标名词，例如：钓鱼台,菜地,水塘"
        className={[
          'w-full rounded-xl px-4 py-3 text-ink-800 placeholder-ink-400',
          'bg-surface border border-ink-200',
          'focus:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-100',
          'resize-none transition-colors duration-200',
          disabled ? 'opacity-60 cursor-not-allowed' : '',
        ].join(' ')}
      />

      <p className="text-xs text-ink-500 leading-relaxed">
        建议输入<span className="text-ink-700 font-medium">具体目标名词</span>，多个目标用逗号分隔。
        名词越短越稳定；过长的句子会先由系统自动抽取关键词，速度较慢且可能漏抽。
      </p>

      {/* Example chips */}
      <div className="flex flex-wrap gap-2">
        <span className="text-ink-500 text-xs self-center">示例：</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            onClick={() => !disabled && onChange(ex)}
            disabled={disabled}
            type="button"
            className={[
              'text-xs px-3 py-1 rounded-full border border-ink-200 bg-surface',
              'text-ink-600 hover:text-brand-600 hover:border-brand-300 hover:bg-brand-50',
              'transition-colors duration-150',
              disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer',
            ].join(' ')}
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  )
}
