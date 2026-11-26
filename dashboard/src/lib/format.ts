// Number formats shared by every view. A value that is missing prints as a dash, never as zero.
export const DASH = "–"
export const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v)

export function fBytes(b: number | null | undefined): string {
  if (!fin(b)) return DASH
  const u = ["B", "KiB", "MiB", "GiB", "TiB"]
  let i = 0
  let v = Math.abs(b)
  while (v >= 1024 && i < u.length - 1) { v /= 1024; ++i }
  const s = v >= 100 || i === 0 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)
  return (b < 0 ? "-" : "") + s + " " + u[i]
}
/** [value, unit] for a tile that sets the unit apart. */
export function bytesParts(b: number | null | undefined): [string, string] {
  const s = fBytes(b)
  if (s === DASH) return [DASH, ""]
  const k = s.lastIndexOf(" ")
  return [s.slice(0, k), s.slice(k + 1)]
}
export const fRate = (b: number | null | undefined) => (fin(b) ? fBytes(b) + "/s" : DASH)
export function fNum(v: number | null | undefined, digits = 0): string {
  if (!fin(v)) return DASH
  return v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })
}
/** Three significant figures without trailing noise: 1,234 · 176 · 22.0 · 2.00 · 0.33 */
export function fSig(v: number | null | undefined): string {
  if (!fin(v)) return DASH
  const a = Math.abs(v)
  if (a >= 100) return fNum(v, 0)
  if (a >= 10) return fNum(v, 1)
  return fNum(v, 2)
}
export const fPct = (v: number | null | undefined, digits = 1) => (fin(v) ? (v * 100).toFixed(digits) : DASH)
export function fDur(s: number | null | undefined): string {
  if (!fin(s)) return DASH
  if (s < 1) return Math.round(s * 1000) + " ms"
  if (s < 60) return (s < 10 ? s.toFixed(1) : s.toFixed(0)) + " s"
  if (s < 3600) return Math.floor(s / 60) + " m " + String(Math.floor(s % 60)).padStart(2, "0") + " s"
  if (s < 86400) return Math.floor(s / 3600) + " h " + String(Math.floor((s % 3600) / 60)).padStart(2, "0") + " m"
  return Math.floor(s / 86400) + " d " + Math.floor((s % 86400) / 3600) + " h"
}
export function fClock(t: number, withSeconds = false): string {
  const d = new Date(t * 1000)
  const p = (n: number) => String(n).padStart(2, "0")
  return p(d.getHours()) + ":" + p(d.getMinutes()) + (withSeconds ? ":" + p(d.getSeconds()) : "")
}
export function fAgo(ms: number, now: number): string {
  return fDur(Math.max(0, (now - ms) / 1000)) + " ago"
}
