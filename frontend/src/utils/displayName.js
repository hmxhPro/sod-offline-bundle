/**
 * src/utils/displayName.js
 * ------------------------
 * 数据集/模型的底层唯一名带 14 位时间戳后缀（如 `船只_20260620153012`），用于区分
 * 同名目标的多次标注/训练。展示时把"干净名"和"时间"拆开 —— 列表里显示「名称 + 时间徽章」。
 */

/**
 * 拆出展示名与时间戳。
 * @param {string} name 形如 "船只_20260620153012"（或带撞名序号 "..._1"）
 * @returns {{ display: string, ts: Date|null }} display=去后缀的名称；ts=解析出的时间（无则 null）
 */
export function splitTimestampName(name) {
  if (!name) return { display: name || '', ts: null }
  const m = /^(.*)_(\d{14})(?:_\d+)?$/.exec(name)
  if (!m) return { display: name, ts: null }
  const [, base, d] = m
  const y = Number(d.slice(0, 4))
  const mo = Number(d.slice(4, 6))
  const day = Number(d.slice(6, 8))
  const h = Number(d.slice(8, 10))
  const mi = Number(d.slice(10, 12))
  const s = Number(d.slice(12, 14))
  const ts = new Date(y, mo - 1, day, h, mi, s)
  // 回填校验：JS Date 会把越界字段（13 月 / 40 日）溢出成另一个合法日期；逐字段比对，
  // 任一不符则视为"这不是时间戳"，原样保留名字、不剥离、不显示徽章。
  const valid =
    ts.getFullYear() === y && ts.getMonth() === mo - 1 && ts.getDate() === day &&
    ts.getHours() === h && ts.getMinutes() === mi && ts.getSeconds() === s
  if (!valid) return { display: name, ts: null }
  return { display: base || name, ts }
}

/** 时间徽章文本：2026-06-20 15:30 */
export function formatTs(ts) {
  if (!ts) return ''
  const p = (n) => String(n).padStart(2, '0')
  return `${ts.getFullYear()}-${p(ts.getMonth() + 1)}-${p(ts.getDate())} ${p(ts.getHours())}:${p(ts.getMinutes())}`
}

/**
 * 便捷：把一个带时间戳的名字 + 可选的回退时间字符串，渲染成「名称 · 时间」。
 * @param {string} name
 * @param {string} [fallbackIso] 当名字里没时间戳时用的 ISO 时间（如 created_at）
 * @returns {{ display: string, badge: string }}
 */
export function nameWithBadge(name, fallbackIso) {
  const { display, ts } = splitTimestampName(name)
  let badge = formatTs(ts)
  if (!badge && fallbackIso) {
    const d = new Date(fallbackIso)
    if (!Number.isNaN(d.getTime())) badge = formatTs(d)
  }
  return { display, badge }
}
