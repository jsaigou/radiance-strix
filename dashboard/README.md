# The dashboard's page

The page the engine serves at `/` (docs/GUIDE.md §6.1): React, Tailwind and shadcn/ui components
on Base UI, charts on uPlot, built by Vite into one document with its script, style and type
inlined. The engine compiles that document in from `core/server/dashboard_page.h`, so its own build
needs no Node.

```sh
npm ci
RADIANCE_ORIGIN=http://127.0.0.1:8000 npm run dev   # the page against a running engine
npm run embed                                      # build, then rewrite core/server/dashboard_page.h
```

Commit the header with the source change that produced it; the test `dashboard_page` fails while
they disagree. The page asks its own origin only: `/history` five times a second for everything it draws
(`src/lib/store.ts`), and `/sessions`, `/graph`, `/server_info` and `/admin/control-vector` when
the view that shows them is open. `logo/` holds the mark as standalone files.

- `src/lib`: what the page asks for and keeps (`api.ts`, `store.ts`), number formats, view state.
- `src/components`: the chart, the small shared pieces, and `ui/` (shadcn components, as source).
- `src/pages`: Overview, and a model's Performance, Requests, Sessions and Details.
