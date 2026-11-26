import fs from "node:fs"
import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig, type Plugin } from "vite"
import { viteSingleFile } from "vite-plugin-singlefile"

// The page carries other people's code and type inside the engine binary, so their licences go
// with it: every package that has code in the bundle, plus the ones that only contribute style or
// fonts, each with its licence text, into dist/THIRD-PARTY-LICENSES.txt.
function thirdPartyLicenses(): Plugin {
  const styleOnly = ["tailwindcss", "tw-animate-css", "shadcn", "@fontsource-variable/geist", "@fontsource-variable/geist-mono"]
  return {
    name: "third-party-licenses",
    apply: "build",
    generateBundle(_options, bundle) {
      const names = new Set<string>(styleOnly)
      for (const chunk of Object.values(bundle)) {
        if (chunk.type !== "chunk") continue
        for (const [id, m] of Object.entries(chunk.modules)) {
          const tail = id.replace(/\\/g, "/").split("node_modules/").pop()
          if (!tail || tail === id || m.renderedLength === 0) continue
          const hit = /^((?:@[^/]+\/)?[^/]+)\//.exec(tail)
          if (hit) names.add(hit[1])
        }
      }
      const parts = [...names].sort().map((name) => {
        const dir = path.resolve(__dirname, "node_modules", name)
        const pkg = JSON.parse(fs.readFileSync(path.join(dir, "package.json"), "utf8"))
        const file = fs.readdirSync(dir).find((f) => /^(licen[sc]e|copying)/i.test(f))
        const text = file ? fs.readFileSync(path.join(dir, file), "utf8").trim() : "(the package ships no licence file; its package.json names the licence above)"
        return `${"=".repeat(100)}\n${name} ${pkg.version}\nLicence: ${pkg.license ?? "see below"}\n${"=".repeat(100)}\n\n${text}\n`
      })
      const head = "Third-party software in the dashboard's page (core/server/dashboard_page.h), which the engine\n" +
        "binary carries: the packages with code, style or type in the built page, each with its licence.\n" +
        "Written by the page's build (dashboard/vite.config.ts); not edited by hand.\n\n"
      this.emitFile({ type: "asset", fileName: "THIRD-PARTY-LICENSES.txt", source: head + parts.join("\n") })
    },
  }
}

// The page ships inside the engine binary as one file: script and style inlined, no asset
// requests, nothing fetched from another origin.
export default defineConfig({
  plugins: [react(), tailwindcss(), viteSingleFile(), thirdPartyLicenses()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  build: { target: "es2022", cssCodeSplit: false, assetsInlineLimit: 100000000, chunkSizeWarningLimit: 4000 },
  server: { port: 5199, proxy: Object.fromEntries(["/stats", "/history", "/sessions", "/graph", "/server_info", "/admin", "/v1"].map((p) => [p, process.env.RADIANCE_ORIGIN ?? "http://127.0.0.1:8102"])) },
})
