// A body asked for when a view opens, and again every `refreshMs` while the view stays open and
// the tab is visible. Kept per path, so coming back to a view shows the last answer at once.
import { useEffect, useState } from "react"
import { api } from "./api"

const cache = new Map<string, unknown>()
export function useFetched<T>(path: string | null, refreshMs = 0): { data: T | null; error: string | null } {
  const [, bump] = useState(0)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    if (!path) return
    let alive = true
    const load = () => {
      if (document.hidden) return
      api<T>(path).then((d) => { if (alive) { cache.set(path, d); setError(null); bump((n) => n + 1) } })
        .catch((e: unknown) => { if (alive) setError(e instanceof Error ? e.message : String(e)) })
    }
    load()
    const timer = refreshMs ? window.setInterval(load, refreshMs) : undefined
    return () => { alive = false; window.clearInterval(timer) }
  }, [path, refreshMs])
  return { data: path ? ((cache.get(path) as T | undefined) ?? null) : null, error }
}
