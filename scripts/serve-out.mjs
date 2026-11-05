/**
 * Zero-dependency static server for the `out/` folder of a Next.js static export.
 * Mirrors Vercel's static serving: /a/b -> a/b.html or a/b/index.html, custom 404.html.
 *   node scripts/serve-out.mjs [port]      (PORT env also works; default 3000)
 */
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { join, extname, normalize, resolve } from "node:path";

const ROOT = resolve(process.env.OUT_DIR ?? "out");
const PORT = Number(process.argv[2] ?? process.env.PORT ?? 3000);
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8", ".xml": "application/xml; charset=utf-8",
  ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".ico": "image/x-icon",
  ".woff2": "font/woff2", ".woff": "font/woff", ".wasm": "application/wasm", ".map": "application/json",
  ".webp": "image/webp", ".mjs": "text/javascript; charset=utf-8",
};

async function isFile(p) {
  try { return (await stat(p)).isFile(); } catch { return false; }
}

async function resolveFile(pathname) {
  const clean = normalize(decodeURIComponent(pathname)).replace(/^(\.\.[/\\])+/, "");
  const base = join(ROOT, clean);
  if (!base.startsWith(ROOT)) return null;
  for (const c of [base, base + ".html", join(base, "index.html")]) if (await isFile(c)) return c;
  return null;
}

createServer(async (req, res) => {
  const { pathname } = new URL(req.url ?? "/", "http://x");
  let file = await resolveFile(pathname);
  let status = 200;
  if (!file) { file = join(ROOT, "404.html"); status = 404; }
  try {
    const body = await readFile(file);
    // the extensionless OG/Twitter cards are given image/png by vercel.json "headers" on Vercel
    const card = /(opengraph|twitter)-image$/.test(file);
    res.writeHead(status, { "content-type": card ? "image/png" : (TYPES[extname(file)] ?? "application/octet-stream") });
    res.end(body);
  } catch {
    res.writeHead(404, { "content-type": "text/plain" });
    res.end("Not found");
  }
}).listen(PORT, () => console.log(`serving ${ROOT} on http://localhost:${PORT}`));
