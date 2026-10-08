/**
 * src/hooks/useTrainingJobs.js
 * ----------------------------
 * Drive a single YOLOE training job: start it, then poll
 * GET /api/training/jobs/{id} every ~3s until a terminal state. The active
 * job id is persisted to localStorage so a page refresh resumes polling
 * (mirrors the reconcile pattern in useDetectionTasks).
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { startTraining as apiStartTraining, getTrainingJob } from '../services/api'

const LS_KEY = 'sod_active_training_job_v1'
// `needs_review` is terminal too (M-2: a finished run whose mAP was too low to
// register as a model). Omitting it here made the poller loop forever.
const TERMINAL = ['finished', 'needs_review', 'failed', 'cancelled']

export function useTrainingJobs(onFinished) {
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)
  const [starting, setStarting] = useState(false)
  const timerRef = useRef(null)
  const onFinishedRef = useRef(onFinished)

  useEffect(() => {
    onFinishedRef.current = onFinished
  }, [onFinished])

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const pollOnce = useCallback(
    async (jobId) => {
      try {
        const j = await getTrainingJob(jobId)
        setJob(j)
        if (TERMINAL.includes(j.status)) {
          localStorage.removeItem(LS_KEY)
          clearTimer()
          // Refresh the model list / category on any successful completion,
          // including needs_review (the category status / job history changed
          // even though no deployable model was registered).
          if (j.status === 'finished' || j.status === 'needs_review') onFinishedRef.current?.(j)
          return
        }
        timerRef.current = setTimeout(() => pollOnce(jobId), 2000)
      } catch (e) {
        const msg = String(e?.message || '')
        if (msg.includes('不存在') || msg.toLowerCase().includes('not found')) {
          localStorage.removeItem(LS_KEY)
          clearTimer()
          setError('训练任务已不存在（服务可能已重启）')
          return
        }
        // transient error — keep retrying
        timerRef.current = setTimeout(() => pollOnce(jobId), 4000)
      }
    },
    [clearTimer]
  )

  const start = useCallback(
    async (categoryId, params = {}) => {
      setError(null)
      setStarting(true)
      try {
        const r = await apiStartTraining(categoryId, params)
        localStorage.setItem(LS_KEY, r.job_id)
        setJob({
          id: r.job_id,
          status: 'pending',
          progress: 0,
          current_epoch: 0,
          total_epochs: params.epochs || 0,
          model_name: r.model_name,
          category_id: categoryId,
        })
        clearTimer()
        pollOnce(r.job_id)
        return r
      } catch (e) {
        setError(e?.message || '启动训练失败')
        throw e
      } finally {
        setStarting(false)
      }
    },
    [clearTimer, pollOnce]
  )

  const reset = useCallback(() => {
    clearTimer()
    localStorage.removeItem(LS_KEY)
    setJob(null)
    setError(null)
  }, [clearTimer])

  // Resume polling a persisted active job on mount.
  useEffect(() => {
    const saved = localStorage.getItem(LS_KEY)
    if (saved) pollOnce(saved)
    return () => clearTimer()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return { job, error, starting, start, reset }
}
