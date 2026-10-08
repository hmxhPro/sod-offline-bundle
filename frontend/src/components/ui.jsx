/**
 * src/components/ui.jsx
 * ---------------------
 * Tiny shared presentational helpers extracted from the old App.jsx so the
 * routed pages can reuse them.
 */

import React from 'react'

/** A label/value pair rendered into a 2-col grid (used by the detect stats card). */
export function StatRow({ label, value, tone }) {
  const toneMap = {
    brand:   'text-brand-600',
    emerald: 'text-emerald-600',
    red:     'text-red-500',
    ink:     'text-ink-800',
  }
  return (
    <>
      <span className="text-ink-500">{label}</span>
      <span className={`font-mono text-right ${toneMap[tone] || 'text-ink-800'}`}>{value}</span>
    </>
  )
}

/** Pill-style tab toggle (used by the training annotate/import switch). */
export function TabButton({ active, onClick, icon, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-medium border transition-colors',
        active
          ? 'bg-brand-500 text-white border-brand-500 shadow-brand'
          : 'bg-surface text-ink-600 border-ink-200 hover:border-brand-300 hover:text-brand-600',
      ].join(' ')}
    >
      {icon}
      {children}
    </button>
  )
}
