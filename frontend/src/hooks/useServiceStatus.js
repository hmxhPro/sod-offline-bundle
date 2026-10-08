import { useEffect, useState } from 'react'

// This indicator checks the data API, not GPU readiness or model health.
export function useServiceStatus() {
  const [service, setService] = useState({ kind: 'checking', label: '正在连接数据服务' })
  useEffect(() => {
    let alive = true
    let controller
    let timeout
    let next
    const check = async () => {
      controller = new AbortController()
      timeout = setTimeout(() => controller.abort(), 5000)
      try {
        const base = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
        const response = await fetch(`${base}/api/tasks?limit=1`, { signal: controller.signal, cache: 'no-store' })
        if (!response.ok) throw new Error('Service unavailable')
        const data = await response.json()
        if (!Array.isArray(data)) throw new Error('Unexpected service response')
        if (alive) setService({ kind: 'connected', label: '数据服务已连接' })
      } catch {
        if (alive) setService({ kind: 'disconnected', label: '数据服务未连接' })
      } finally {
        clearTimeout(timeout)
        if (alive) next = setTimeout(check, 30000)
      }
    }
    check()
    return () => { alive = false; clearTimeout(timeout); clearTimeout(next); controller?.abort() }
  }, [])
  return service
}
