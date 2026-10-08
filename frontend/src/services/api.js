/**
 * src/services/api.js
 * -------------------
 * Axios-based API client for the FastAPI backend.
 */

import axios from 'axios'

// Base URL: empty string means "same host" (works with Vite proxy in dev,
// and when frontend is served by the backend in production).
const BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

const api = axios.create({
  baseURL: BASE_URL,
  // No timeout by default — individual calls (upload) may take a long time
  // for large files. Non-upload calls are fast and SSE uses EventSource.
  timeout: 0,
  maxContentLength: Infinity,
  maxBodyLength: Infinity,
})

// ── Request / Response interceptors ──────────────────────────────────────────

// Normalize every backend error into a friendly Error. The backend wraps all
// non-2xx responses as { code, message, detail, request_id } (see
// app/core/errors.py); older/raw FastAPI bodies (string or array `detail`) are
// still handled. The thrown Error carries `.code`, `.requestId`, `.status` so
// callers/UI can branch or show the id. For 5xx we append the request id to the
// message so it surfaces everywhere errors are already displayed (TaskCard,
// alerts, ErrorBoundary) — letting users quote it for server-log lookup.
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const res = error.response
    const data = res?.data
    let message

    if (data && typeof data === 'object') {
      if (typeof data.message === 'string' && data.message) {
        message = data.message
      } else if (typeof data.detail === 'string' && data.detail) {
        message = data.detail
      } else if (Array.isArray(data.detail)) {
        // FastAPI default 422 body: [{ loc, msg, type }, ...]
        message = data.detail.map((d) => d?.msg || JSON.stringify(d)).join('；')
      }
    }
    if (!message) {
      message = res
        ? `请求失败（HTTP ${res.status}）`
        : '网络连接失败，请检查后端服务是否运行。'
    }

    const requestId = data?.request_id || res?.headers?.['x-request-id']
    if (requestId && res && res.status >= 500) {
      message = `${message}（错误码 ${requestId}）`
    }

    const err = new Error(message)
    err.code = data?.code
    err.requestId = requestId
    err.status = res?.status
    return Promise.reject(err)
  }
)

// ── API methods ───────────────────────────────────────────────────────────────

/**
 * Upload a video file.
 * @param {File} file
 * @param {(progressPercent: number) => void} [onProgress]
 * @returns {Promise<UploadResponse>}
 */
export async function uploadVideo(file, onProgress) {
  const formData = new FormData()
  formData.append('file', file)

  const { data } = await api.post('/api/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (evt) => {
      if (onProgress && evt.total) {
        onProgress(Math.round((evt.loaded * 100) / evt.total))
      }
    },
  })
  return data
}

/**
 * Start a detection task.
 * Provide exactly one of `prompt` (natural-language open-vocab) or `model_id`
 * (use a trained model's baked-in classes).
 * @param {{ video_id: string, video_filename?: string, prompt?: string, model_id?: string, detection_interval?: number, enable_vlm?: boolean }} params
 * @returns {Promise<DetectResponse>}
 */
export async function startDetection(params) {
  const { data } = await api.post('/api/detect', params)
  return data
}

/**
 * Get task state (polling fallback).
 * @param {string} taskId
 * @returns {Promise<TaskState>}
 */
export async function getTask(taskId) {
  const { data } = await api.get(`/api/task/${taskId}`)
  return data
}

/**
 * Request cancellation of a running detection task.
 * @param {string} taskId
 */
export async function cancelDetection(taskId) {
  const { data } = await api.post(`/api/task/${taskId}/cancel`)
  return data
}

/**
 * Pause a running detection task.
 * @param {string} taskId
 */
export async function pauseDetection(taskId) {
  const { data } = await api.post(`/api/task/${taskId}/pause`)
  return data
}

/**
 * Resume a paused detection task.
 * @param {string} taskId
 */
export async function resumeDetection(taskId) {
  const { data } = await api.post(`/api/task/${taskId}/resume`)
  return data
}

/**
 * Manually terminate a running task and package results.
 * @param {string} taskId
 */
export async function terminateDetection(taskId) {
  const { data } = await api.post(`/api/task/${taskId}/terminate`)
  return data
}

/**
 * Get the SSE stream URL for a task.
 * @param {string} taskId
 * @returns {string}
 */
export function getStreamUrl(taskId) {
  return `${BASE_URL}/api/stream/${taskId}`
}

/**
 * Get the URL for a single annotated frame image.
 * @param {string} taskId
 * @param {string} filename
 * @returns {string}
 */
export function getFrameUrl(taskId, filename) {
  return `${BASE_URL}/api/frame/${taskId}/${filename}`
}

/**
 * Get the download URL for a finished task's ZIP.
 * @param {string} taskId
 * @returns {string}
 */
export function getDownloadUrl(taskId) {
  return `${BASE_URL}/api/download/${taskId}`
}

// ── History (DB-backed) ─────────────────────────────────────────────────────

/**
 * List past detection tasks, newest first.
 * @param {{ limit?: number, offset?: number, date?: string }} [opts]
 * @returns {Promise<Array<TaskHistoryItem>>}
 */
export async function getTaskHistory({ limit = 100, offset = 0, date } = {}) {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
  if (date) params.set('date', date)
  const { data } = await api.get(`/api/tasks?${params.toString()}`)
  return data
}

/**
 * List saved annotated frame filenames for a past task.
 * @param {string} taskId
 * @returns {Promise<string[]>}
 */
export async function getTaskFrames(taskId) {
  const { data } = await api.get(`/api/task/${taskId}/frames`)
  return data
}

/**
 * Hard-delete one history task: DB row + frame dir + ZIP + uploaded video
 * (the upload is kept if other tasks still reference its video_id).
 * Backend returns 409 if the task is still running — caller should cancel
 * it first.
 * @param {string} taskId
 */
export async function deleteHistoryTask(taskId) {
  await api.delete(`/api/task/${taskId}`)
}

/**
 * Wipe ALL history: every DB row + each task's result dir + upload.
 * Requires the confirm token (server returns 409 without it). 409 also if any
 * task is still active in memory.
 */
export async function deleteAllHistory() {
  await api.delete(`/api/tasks?confirm=DELETE_ALL`)
}

// ════════════════════════════════════════════════════════════════════════════
// YOLOE custom-training workflow (REQ1/REQ2/REQ3)
// ════════════════════════════════════════════════════════════════════════════

/** Prefix a backend-relative media path (e.g. annotated_url) with the base URL. */
export function mediaUrl(path) {
  return `${BASE_URL}${path}`
}

// ── Categories ───────────────────────────────────────────────────────────────

export async function createCategory(name, description) {
  const { data } = await api.post('/api/categories', { name, description })
  return data
}

export async function getCategories() {
  const { data } = await api.get('/api/categories')
  return data
}

export async function getCategory(categoryId) {
  const { data } = await api.get(`/api/categories/${categoryId}`)
  return data
}

export async function deleteCategory(categoryId) {
  await api.delete(`/api/categories/${categoryId}`)
}

/**
 * Merge source category dataset into target category
 * @param {string} targetCategoryId - Target category ID
 * @param {string} sourceCategoryId - Source category ID
 * @param {boolean} deleteSource - Whether to delete source category after merge
 */
export async function mergeDatasets(targetCategoryId, sourceCategoryId, deleteSource = false) {
  const formData = new FormData()
  formData.append('target_category_id', targetCategoryId)
  formData.append('source_category_id', sourceCategoryId)
  formData.append('delete_source', deleteSource)

  const { data } = await api.post('/api/categories/merge', formData)
  return data
}

// ── Dataset images + annotation ──────────────────────────────────────────────

export async function uploadCategoryImages(categoryId, files, onProgress) {
  const formData = new FormData()
  for (const f of files) formData.append('files', f)
  const { data } = await api.post(`/api/categories/${categoryId}/images`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (evt) => {
      if (onProgress && evt.total) onProgress(Math.round((evt.loaded * 100) / evt.total))
    },
  })
  return data
}

export async function getCategoryImages(categoryId) {
  const { data } = await api.get(`/api/categories/${categoryId}/images`)
  return data
}

export function getImageFileUrl(categoryId, imageId) {
  return `${BASE_URL}/api/categories/${categoryId}/images/${imageId}/file`
}

export async function getImageAnnotation(categoryId, imageId) {
  const { data } = await api.get(`/api/categories/${categoryId}/images/${imageId}/annotation`)
  return data
}

export async function saveImageAnnotation(categoryId, imageId, boxes) {
  const { data } = await api.put(
    `/api/categories/${categoryId}/images/${imageId}/annotation`,
    { boxes }
  )
  return data
}

export async function deleteCategoryImage(categoryId, imageId) {
  await api.delete(`/api/categories/${categoryId}/images/${imageId}`)
}

/**
 * Import a pre-annotated YOLO dataset folder into a category (second data
 * source alongside in-browser annotation). All boxes are folded to the
 * category's single class on the backend.
 * @param {string} categoryId
 * @param {File[]} files            images + .txt label files
 * @param {string[]} relPaths       webkitRelativePath for each file (same order as files)
 * @param {(pct:number)=>void} [onProgress]
 * @returns {Promise<{imported_images:number, with_annotation:number, background:number, skipped_files:number, message:string}>}
 */
export async function importCategoryDataset(categoryId, files, relPaths, onProgress) {
  const formData = new FormData()
  files.forEach((f, i) => {
    formData.append('files', f)
    formData.append('rel_paths', relPaths[i] ?? f.name)
  })
  const { data } = await api.post(`/api/categories/${categoryId}/dataset/import`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (evt) => {
      if (onProgress && evt.total) onProgress(Math.round((evt.loaded * 100) / evt.total))
    },
  })
  return data
}

// ── Batched import (handles large folders) ───────────────────────────────────
// A whole dataset folder can be thousands of files. The backend (Starlette)
// caps one multipart request at 1000 files / 1000 fields, and the dev proxy
// chokes on multi-hundred-MB bodies — sending everything at once shows up as a
// "Network Error". So we split into several bounded requests. The backend
// import is additive (each call appends rows + recomputes counts), so batches
// accumulate naturally; we only must keep every image together with its label
// in the SAME request, because pairing happens per-request.

const IMPORT_MAX_FILES_PER_BATCH = 150              // ≤ ~300 multipart parts, safely under 1000
const IMPORT_MAX_BYTES_PER_BATCH = 64 * 1024 * 1024 // keep each request body modest for the proxy
const _LABEL_DIR_SEGMENTS = new Set(['images', 'labels', 'image', 'label'])

/** Mirror of backend `_match_key` (app/api/dataset.py): drop images/labels path
 *  segments + the extension so an image and its label collapse to one key. */
function datasetMatchKey(relPath) {
  const norm = (relPath || '').replace(/\\/g, '/')
  const parts = norm.split('/').filter((p) => p && !_LABEL_DIR_SEGMENTS.has(p.toLowerCase()))
  if (!parts.length) return ''
  const last = parts[parts.length - 1]
  const dot = last.lastIndexOf('.')
  parts[parts.length - 1] = dot > 0 ? last.slice(0, dot) : last
  return parts.join('/')
}

/**
 * Import a dataset folder in size-bounded batches (the safe path for large
 * folders). Groups files by match key so image↔label pairs never split across
 * requests, uploads batches sequentially, and aggregates the per-batch results.
 * @param {string} categoryId
 * @param {File[]} files
 * @param {string[]} relPaths           webkitRelativePath per file (same order)
 * @param {{ onProgress?: (overallPct:number, info:{index:number,count:number})=>void,
 *           onBatch?: (info:{index:number,count:number})=>void }} [hooks]
 * @returns {Promise<{imported_images:number, with_annotation:number, background:number,
 *           skipped_files:number, batches:number, message:string}>}
 *   On a mid-way batch failure, throws an Error whose `.partial` holds the
 *   counts already committed plus the failed batch index.
 */
export async function importCategoryDatasetBatched(categoryId, files, relPaths, { onProgress, onBatch } = {}) {
  // 1) Group whole image↔label sets by match key.
  const groups = new Map()
  files.forEach((f, i) => {
    const rel = relPaths[i] ?? f.name
    const key = datasetMatchKey(rel) || `__orphan_${i}` // unkeyed file → its own bucket
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key).push({ file: f, rel })
  })

  // 2) Greedily pack groups into byte/count-bounded batches (never split a group).
  const batches = []
  let cur = []
  let curFiles = 0
  let curBytes = 0
  for (const g of groups.values()) {
    const gFiles = g.length
    const gBytes = g.reduce((s, x) => s + (x.file.size || 0), 0)
    if (cur.length && (curFiles + gFiles > IMPORT_MAX_FILES_PER_BATCH ||
                       curBytes + gBytes > IMPORT_MAX_BYTES_PER_BATCH)) {
      batches.push(cur)
      cur = []; curFiles = 0; curBytes = 0
    }
    cur.push(...g); curFiles += gFiles; curBytes += gBytes
  }
  if (cur.length) batches.push(cur)

  // 3) Upload sequentially; track overall progress by bytes.
  const totalBytes = files.reduce((s, f) => s + (f.size || 0), 0) || 1
  const agg = { imported_images: 0, with_annotation: 0, background: 0, skipped_files: 0 }
  let bytesDone = 0

  for (let b = 0; b < batches.length; b++) {
    const items = batches[b]
    const batchBytes = items.reduce((s, x) => s + (x.file.size || 0), 0)
    const info = { index: b + 1, count: batches.length }
    onBatch?.(info)
    try {
      const res = await importCategoryDataset(
        categoryId,
        items.map((x) => x.file),
        items.map((x) => x.rel),
        (pct) => {
          const overall = ((bytesDone + (pct / 100) * batchBytes) / totalBytes) * 100
          onProgress?.(Math.min(99, Math.round(overall)), info)
        },
      )
      agg.imported_images += res.imported_images || 0
      agg.with_annotation += res.with_annotation || 0
      agg.background += res.background || 0
      agg.skipped_files += res.skipped_files || 0
    } catch (e) {
      const err = new Error(e?.message || '上传失败')
      err.partial = { ...agg, failedBatch: b + 1, totalBatches: batches.length }
      throw err
    }
    bytesDone += batchBytes
    onProgress?.(Math.round((bytesDone / totalBytes) * 100), info)
  }

  agg.batches = batches.length
  agg.message =
    `导入 ${agg.imported_images} 张图片（${agg.with_annotation} 张含标注，${agg.background} 张背景）` +
    (agg.skipped_files ? `，忽略 ${agg.skipped_files} 个无效文件` : '') +
    (batches.length > 1 ? `，分 ${batches.length} 批上传` : '')
  return agg
}

// ── Training ─────────────────────────────────────────────────────────────────

export async function startTraining(categoryId, params = {}) {
  const { data } = await api.post(`/api/categories/${categoryId}/train`, params)
  return data
}

export async function getTrainingJobs(categoryId) {
  const params = categoryId ? `?category_id=${encodeURIComponent(categoryId)}` : ''
  const { data } = await api.get(`/api/training/jobs${params}`)
  return data
}

export async function getTrainingJob(jobId) {
  const { data } = await api.get(`/api/training/jobs/${jobId}`)
  return data
}

export async function cancelTraining(jobId) {
  const { data } = await api.post(`/api/training/jobs/${jobId}/cancel`)
  return data
}

// ── Trained models (REQ3) ────────────────────────────────────────────────────

export async function getModels(categoryId) {
  const params = categoryId ? `?category_id=${encodeURIComponent(categoryId)}` : ''
  const { data } = await api.get(`/api/models${params}`)
  return data
}

export async function getModel(modelId) {
  const { data } = await api.get(`/api/models/${modelId}`)
  return data
}

export async function deleteModel(modelId) {
  await api.delete(`/api/models/${modelId}`)
}

// ── Image detection (REQ1) ───────────────────────────────────────────────────

/**
 * Detect on one or more images.
 * @param {File[]} files
 * @param {{ modelId?: string, classNames?: string, conf?: number, detectionMode?: string, llmModel?: string, llmPrompt?: string }} opts
 * @param {(pct:number)=>void} [onProgress]
 */
export async function detectImages(files, options = {}, onProgress) {
  const { modelId, classNames, conf, detectionMode, llmModel, llmPrompt } = options

  const formData = new FormData()
  for (const f of files) formData.append('files', f)

  if (detectionMode) formData.append('detection_mode', detectionMode)
  if (modelId) formData.append('model_id', modelId)
  if (classNames) formData.append('class_names', classNames)
  if (llmModel) formData.append('llm_model', llmModel)
  if (llmPrompt) formData.append('llm_prompt', llmPrompt)
  if (conf != null) formData.append('conf', String(conf))

  const { data } = await api.post('/api/image-detect', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (evt) => {
      if (onProgress && evt.total) onProgress(Math.round((evt.loaded * 100) / evt.total))
    },
  })
  return data
}

// ── LLM 自动标注（Qwen，配置全部在后端 .env）─────────────────────────────────
// 提供商/模型/密钥都固定在后端，前端只读取配置展示，并提交标注任务。

/**
 * Read the backend-fixed annotation config (provider/model/configured flag).
 * The API key is never returned — only whether one is configured.
 * @returns {Promise<{provider:string, model:string, configured:boolean, max_concurrent:number}>}
 */
export async function getAnnotationConfig() {
  const { data } = await api.get('/api/annotation-config')
  return data
}

/** Fetch model IDs available to the backend-configured API key. */
export async function getAnnotationModels() {
  const { data } = await api.get('/api/annotation-models')
  return data
}

/** Set the backend default annotation model (probed server-side, persisted to .env). */
export async function setDefaultAnnotationModel(model) {
  const { data } = await api.put('/api/annotation-config/model', { model })
  return data
}

/**
 * Create an LLM annotation task for a category.
 * @param {{ category_id: string, prompt: string, max_concurrent?: number, only_pending?: boolean }} params
 */
export async function createAnnotationTask(params) {
  const { data } = await api.post('/api/annotation-tasks', params)
  return data
}

/**
 * List annotation tasks, optionally filtered by category.
 * @param {string} [categoryId]
 */
export async function getAnnotationTasks(categoryId) {
  const params = categoryId ? `?category_id=${encodeURIComponent(categoryId)}` : ''
  const { data } = await api.get(`/api/annotation-tasks${params}`)
  return data
}

/** Create a new iterative annotation round for an existing category. */
export async function createAnnotationIteration(categoryId, params) {
  const { data } = await api.post(`/api/categories/${categoryId}/annotation-iterations`, params)
  return data
}

/** List annotation iterations for a category, newest first. */
export async function getAnnotationIterations(categoryId) {
  const { data } = await api.get(`/api/categories/${categoryId}/annotation-iterations`)
  return data
}

/** Cancel a running/pending annotation task. */
export async function cancelAnnotationTask(taskId) {
  await api.post(`/api/annotation-tasks/${taskId}/cancel`)
}

/**
 * 删除一个已结束（失败 / 已取消 / 已完成）的标注任务，并清理其中未成功标注的图片；
 * 已成功标注的图片予以保留。若数据集因此变空，则连同空数据集一并删除。进行中的任务
 * 需先取消（后端返回 409）。
 */
export async function deleteAnnotationTask(taskId) {
  const { data } = await api.delete(`/api/annotation-tasks/${taskId}`)
  return data
}

/**
 * 一站式自动标注：描述要标注的目标 + 上传图片或视频。后端自动提取目标名词、建带
 * 时间戳的数据集、（视频则先抽帧）逐张标注。复用标注任务表，进度仍走
 * getAnnotationTasks 轮询。
 * @param {{ target: string, files: File[], videoDensity?: 'sparse'|'medium'|'dense', model?: string }} params
 * @param {(pct:number)=>void} [onProgress] 上传进度 0-100
 * @returns {Promise<{task_id:string, iteration_id?:string, category_id:string, category_name:string, noun:string, mode:string, total_images:number}>}
 */
export async function autoAnnotate({ target, files, videoDensity = 'medium', model }, onProgress) {
  const formData = new FormData()
  formData.append('target', target)
  formData.append('video_density', videoDensity)
  if (model) formData.append('model', model)
  for (const f of files) formData.append('files', f)
  const { data } = await api.post('/api/auto-annotate', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (evt) => {
      if (onProgress && evt.total) onProgress(Math.round((evt.loaded * 100) / evt.total))
    },
  })
  return data
}
