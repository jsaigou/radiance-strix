// What the viewer chose, kept across reloads where the browser lets the page keep anything: the
// theme, the charts' time window, and the view named by the address's fragment.
import { useSyncExternalStore } from "react"

function store<T>(initial: T) {
  let value = initial
  const subs = new Set<() => void>()
  return {
    get: () => value,
    set(v: T) { if (v !== value) { value = v; subs.forEach((f) => f()) } },
    subscribe(f: () => void) { subs.add(f); return () => { subs.delete(f) } },
  }
}
const read = (k: string): string | null => { try { return localStorage.getItem(k) } catch { return null } }
const write = (k: string, v: string) => { try { localStorage.setItem(k, v) } catch { /* storage may be blocked */ } }

export type Theme = "dark" | "light"
const theme = store<Theme>(document.documentElement.classList.contains("dark") ? "dark" : "light")
export function setTheme(t: Theme) {
  document.documentElement.classList.toggle("dark", t === "dark")
  write("radiance.theme", t)
  theme.set(t)
}
export const useTheme = () => useSyncExternalStore(theme.subscribe, theme.get)

export const RANGES = [[60, "1m"], [300, "5m"], [900, "15m"], [3600, "1h"]] as const
const stored = Number(read("radiance.range"))
const range = store<number>(RANGES.some(([s]) => s === stored) ? stored : 300)
export function setRange(s: number) { write("radiance.range", String(s)); range.set(s) }
export const useRange = () => useSyncExternalStore(range.subscribe, range.get)

export type Tab = "performance" | "requests" | "sessions" | "details"
export interface Route { model: string | null; tab: Tab }
const TABS: Tab[] = ["performance", "requests", "sessions", "details"]
function parse(): Route {
  const m = /^#\/m\/([^/]+)(?:\/([a-z]+))?/.exec(location.hash)
  if (!m) return { model: null, tab: "performance" }
  const tab = TABS.includes(m[2] as Tab) ? (m[2] as Tab) : "performance"
  return { model: decodeURIComponent(m[1]), tab }
}
let current = parse()
const routeSubs = new Set<() => void>()
window.addEventListener("hashchange", () => { current = parse(); routeSubs.forEach((f) => f()) })
export const useRoute = () => useSyncExternalStore((f) => { routeSubs.add(f); return () => { routeSubs.delete(f) } }, () => current)
export const hrefOf = (model: string | null, tab: Tab = "performance") => (model === null ? "#/" : "#/m/" + encodeURIComponent(model) + "/" + tab)
