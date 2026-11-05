/**
 * Zero-dependency static server for the `out/` folder of a Next.js static export.
 * Mirrors Vercel's static serving: /a/b -> a/b.html or a/b/index.html, custom 404.html.
 *   node scripts/serve-out.mjs [port]      (PORT env also works; default 3000)
 */
import { createServer } from "node:http";
import { gzipSync } from "node:zlib";
import { readFile, stat } from "node:fs/promises";
import { readFileSync, existsSync } from "node:fs";
import { join, extname, normalize, resolve } from "node:path";

const ROOT = resolve(process.env.OUT_DIR ?? "out");
// Accept `[port]`, `-p <port>` and `--port <port>` (Lighthouse CI and Playwright call `pnpm start -p 3000`, as with `next start`).
const args = process.argv.slice(2);
const flag = args.findIndex((a) => a === "-p" || a === "--port");
const PORT = Number(
  (flag >= 0 ? args[flag + 1] : args.find((a) => /^\d+$/.test(a))) ??
    process.env.PORT ??
    3000,
);
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".xml": "application/xml; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".woff": "font/woff",
  ".wasm": "application/wasm",
  ".map": "application/json",
  ".webp": "image/webp",
  ".mjs": "text/javascript; charset=utf-8",
};

// Exact-path redirects from vercel.json, as Vercel applies them (permanent -> 308, otherwise 307).
const REDIRECTS = existsSync("vercel.json")
  ? (JSON.parse(readFileSync("vercel.json", "utf8")).redirects ?? [])
  : [];

async function isFile(p) {
  try {
    return (await stat(p)).isFile();
  } catch {
    return false;
  }
}

async function resolveFile(pathname) {
  const clean = normalize(decodeURIComponent(pathname)).replace(
    /^(\.\.[/\\])+/,
    "",
  );
  const base = join(ROOT, clean);
  if (!base.startsWith(ROOT)) return null;
  for (const c of [base, base + ".html", join(base, "index.html")])
    if (await isFile(c)) return c;
  return null;
}

createServer(async (req, res) => {
  const { pathname } = new URL(req.url ?? "/", "http://x");
  const r = REDIRECTS.find((x) => x.source === pathname);
  if (r) {
    res.writeHead(r.permanent === false ? 307 : 308, {
      location: r.destination,
    });
    res.end();
    return;
  }
  let file = await resolveFile(pathname);
  let status = 200;
  if (!file) {
    file = join(ROOT, "404.html");
    status = 404;
  }
  try {
    const body = await readFile(file);
    // the extensionless OG/Twitter cards are given image/png by vercel.json "headers" on Vercel
    const card = /(opengraph|twitter)-image$/.test(file);
    const type = card
      ? "image/png"
      : (TYPES[extname(file)] ?? "application/octet-stream");
    // compress text and cache hashed assets, as Vercel does, so Lighthouse measures the site, not this server
    const zip =
      /gzip/.test(req.headers["accept-encoding"] ?? "") &&
      /text|javascript|json|xml|svg|wasm/.test(type);
    const headers = { "content-type": type };
    if (pathname.startsWith("/_next/static/"))
      headers["cache-control"] = "public, max-age=31536000, immutable";
    if (zip) headers["content-encoding"] = "gzip";
    res.writeHead(status, headers);
    res.end(zip ? gzipSync(body) : body);
  } catch {
    res.writeHead(404, { "content-type": "text/plain" });
    res.end("Not found");
  }
}).listen(PORT, () =>
  console.log(`Ready: serving ${ROOT} on http://localhost:${PORT}`),
);
