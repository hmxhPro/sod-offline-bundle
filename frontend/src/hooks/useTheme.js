/**
 * src/hooks/useTheme.js
 * ---------------------
 * Light/dark theme state. Persists the choice to localStorage and toggles the
 * `.dark` class on <html> (the CSS-variable token system in index.css does the
 * rest). Defaults to dark. A matching no-FOUC script in index.html applies the
 * stored class before React mounts so there's no flash on load.
 */

import { useCallback, useEffect, useState } from 'react'

const KEY = 'sod_theme'

export function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem(KEY) === 'light' ? 'light' : 'dark'
    } catch {
      return 'dark'
    }
  })

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'dark') root.classList.add('dark')
    else root.classList.remove('dark')
    try {
      localStorage.setItem(KEY, theme)
    } catch {
      /* ignore storage failures (private mode etc.) */
    }
  }, [theme])

  const toggle = useCallback(() => {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
  }, [])

  return { theme, toggle }
}
