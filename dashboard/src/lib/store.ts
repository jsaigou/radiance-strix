// The page's copy of what the process remembers. One Scope per subject (the fleet, or a model):
// it loads the server's hour once, then asks five times a second for what came after its last sample and
// appends it. Views read a scope through useScope and redraw when its version moves.
import { useSyncExternalStore } from "react"
import { HttpError, api, withModel, type FinishedRequest, type Fleet, type HistoryBody, type Stats } from "./api"

const KEEP_S = 3600
// The server samples five times a second while a page asks (once a second otherwise), so each
// answer carries the rows since the last one and the live object.
const POLL_MS = 200

export class Series {
  t: number[] = []
  c = new Map<string, (number | null)[]>()

  append(t: number[], cols: Record<string, (number | null)[]>) {
    // Only what is later than everything held: a sample the server sent before is not kept twice.
    let from = 0
    while (from < t.length && t[from] <= this.end) ++from
    if (from === t.length) return
    const n0 = this.t.length
    for (let i = from; i < t.length; ++i) this.t.push(t[i])
    const n = this.t.length
    for (const [k, a] of Object.entries(cols)) {
      let col = this.c.get(k)
      if (!col) { col = new Array(n0).fill(null); this.c.set(k, col) }
      for (let i = from; i < a.length; ++i) col.push(a[i])
    }
    for (const col of this.c.values()) while (col.length < n) col.push(null)
    const cut = this.firstAfter(this.t[n - 1] - KEEP_S)
    if (cut > 64) { this.t.splice(0, cut); for (const col of this.c.values()) col.splice(0, cut) }
  }
  /** The first index whose time is later than `t`. */
  firstAfter(t: number): number {
    let lo = 0, hi = this.t.length
    while (lo < hi) { const m = (lo + hi) >> 1; if (this.t[m] <= t) lo = m + 1; else hi = m }
    return lo
  }
  col(k: string): (number | null)[] | undefined { return this.c.get(k) }
  has(k: string): boolean { const a = this.c.get(k); return !!a && a.some((v) => v !== null) }
  /** The latest value of a column among the last `within` samples (five minutes of them). */
  last(k: string, within = 300): number | null {
    const a = this.c.get(k)
    if (a) for (let i = a.length - 1; i >= 0 && i >= a.length - within; --i) if (a[i] !== null) return a[i]
    return null
  }
  get end(): number { return this.t.length ? this.t[this.t.length - 1] : 0 }
}

type Listener = () => void

export class Scope<Live> {
  series = new Series()
  live: Live | null = null
  finished: FinishedRequest[] = []
  loaded = false
  error: string | null = null
  /** True once the server said this subject is not for this client (403): asking again is pointless. */
  refused = false
  version = 0
  private path: string
  private subs = new Set<Listener>()
  private timer: number | undefined
  private busy = false
  private users = 0

  constructor(path: string) { this.path = path }

  subscribe = (f: Listener) => {
    this.subs.add(f)
    if (++this.users === 1) this.start()
    return () => { this.subs.delete(f); if (--this.users === 0) this.stop() }
  }
  getVersion = () => this.version

  private start() {
    void this.poll()
    this.timer = window.setInterval(() => { if (!document.hidden) void this.poll() }, POLL_MS)
    document.addEventListener("visibilitychange", this.onVisible)
  }
  private stop() {
    window.clearInterval(this.timer)
    document.removeEventListener("visibilitychange", this.onVisible)
  }
  private onVisible = () => { if (!document.hidden) void this.poll() }

  private async poll() {
    if (this.busy || this.refused) return
    this.busy = true
    try {
      const since = this.series.end
      const sep = this.path.includes("?") ? "&" : "?"
      const b = await api<HistoryBody<Live>>(this.path + (since ? sep + "since=" + since : ""))
      this.series.append(b.t, b.c)
      if (b.requests?.length) {
        const have = new Set(this.finished.slice(-64).map((r) => r.id + ":" + r.at))
        for (const r of b.requests) if (!have.has(r.id + ":" + r.at)) this.finished.push(r)
        if (this.finished.length > 500) this.finished.splice(0, this.finished.length - 500)
      }
      this.live = b.live
      this.loaded = true
      this.error = null
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e)
      if (e instanceof HttpError && e.status === 403) this.refused = true
    } finally {
      this.busy = false
      ++this.version
      this.subs.forEach((f) => f())
    }
  }
}

export const fleet = new Scope<Fleet>("history")
const models = new Map<string, Scope<Stats>>()
export function modelScope(id: string): Scope<Stats> {
  let s = models.get(id)
  if (!s) { s = new Scope<Stats>(withModel("history", id)); models.set(id, s) }
  return s
}

/** Subscribes the component to a scope; the scope polls while anything is subscribed. */
export function useScope<L>(s: Scope<L>): Scope<L> {
  useSyncExternalStore(s.subscribe, s.getVersion)
  return s
}
