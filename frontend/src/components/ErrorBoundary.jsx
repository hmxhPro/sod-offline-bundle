import React from 'react'

/**
 * src/components/ErrorBoundary.jsx
 * --------------------------------
 * Top-level error boundary (F-1). Catches render-time exceptions anywhere in the
 * component tree and shows a reload affordance instead of unmounting to a blank
 * white page (which would also hide the history/download entry points).
 *
 * Caveat: React error boundaries do NOT catch errors thrown in event handlers,
 * async callbacks, or the SSE stream — those still require local try/catch.
 */
export default class ErrorBoundary extends React.Component {
  state = { err: null }

  static getDerivedStateFromError(err) {
    return { err }
  }

  componentDidCatch(err, info) {
    // Log for diagnosis; hook a real error reporter here later (F-9).
    console.error('UI crashed:', err, info)
  }

  render() {
    if (!this.state.err) return this.props.children
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4 p-6 text-center">
        <div className="text-lg font-semibold text-ink-800">页面出现错误</div>
        <p className="text-sm text-ink-500 max-w-md break-all">
          {this.state.err?.message || '渲染异常，请刷新重试。'}
        </p>
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="px-4 py-2 rounded-xl bg-brand-500 text-white font-medium hover:bg-brand-600"
        >
          刷新页面
        </button>
      </div>
    )
  }
}
