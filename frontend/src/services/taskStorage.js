/**
 * src/services/taskStorage.js
 * ----------------------------
 * Persist the detection task list to localStorage so the workspace
 * survives page refreshes. Strips non-serializable / oversized fields
 * (File object, base64 image strings) before writing.
 *
 * Schema is versioned via STORAGE_KEY — bump the suffix when changing
 * the persisted shape so old payloads are ignored cleanly.
 */

export const STORAGE_KEY = 'sod_detection_tasks_v1'

function stripFrame(frame) {
  if (!frame || typeof frame !== 'object') return frame
  // image_b64 can be hundreds of KB per frame; image_filename + taskId
  // is enough for the UI to re-fetch via /api/frame/{taskId}/{filename}.
  const { image_b64: _omit, ...rest } = frame
  return rest
}

function sanitize(task) {
  // Drop the File handle (non-serializable) and any inline base64 frames.
  const { file: _file, latestFrame, allFrames, ...rest } = task
  // A frame whose only image source is image_b64 (no image_filename on
  // disk) can't be rendered after a refresh — the b64 is stripped above
  // and the disk URL fallback has nothing to point at, so the <img>
  // collapses to a white placeholder with just a frame number for alt
  // text. Filter those out at save time so the post-refresh grid only
  // contains frames we can actually display.
  const safeAllFrames = Array.isArray(allFrames)
    ? allFrames.filter((f) => f && f.image_filename).map(stripFrame)
    : []
  const safeLatest = latestFrame?.image_filename ? stripFrame(latestFrame) : null
  return {
    ...rest,
    latestFrame: safeLatest,
    allFrames: safeAllFrames,
  }
}

export function loadTasks() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch (err) {
    console.warn('[taskStorage] loadTasks failed, starting fresh:', err?.message)
    return []
  }
}

export function saveTasks(tasks) {
  try {
    const safe = (tasks || []).map(sanitize)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(safe))
  } catch (err) {
    // QuotaExceededError, private-mode storage disabled, etc.
    // Persistence is best-effort — never crash the UI.
    console.warn('[taskStorage] saveTasks failed:', err?.message)
  }
}

export function clearTasks() {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch (err) {
    console.warn('[taskStorage] clearTasks failed:', err?.message)
  }
}
