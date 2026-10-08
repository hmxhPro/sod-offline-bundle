/**
 * src/components/PageHeader.jsx
 * ----------------------------
 * Shared page top-bar for every console page: an optional icon + title +
 * subtitle on the left, and a free-form `right` slot (status chip / actions)
 * on the right.
 */

import React from 'react'

export default function PageHeader({ title, subtitle, icon, right }) {
  return (
    <div className="page-heading">
      <div className="flex items-start gap-3 min-w-0">
        {icon && (
          <span className="page-heading-icon">{icon}</span>
        )}
        <div className="min-w-0">
          <h1 className="page-heading-title">{title}</h1>
          {subtitle && <p className="mt-1 text-sm text-ink-500">{subtitle}</p>}
        </div>
      </div>
      {right && <div className="page-heading-actions">{right}</div>}
    </div>
  )
}
