/**
 * scripts/smoke-check.ts
 *
 * Post-deploy smoke check, adapted from the companion sites'. Fetches every
 * page and exits non-zero if any fails:
 *
 *     pnpm smoke https://gpu-kernels-explained.vercel.app
 *
 * With no argument it checks http://localhost:3000. For a protected
 * preview deployment, pass the bypass token as VERCEL_BYPASS (sent as the
 * `x-vercel-protection-bypass` header; never printed).
 *
 * Fails when a page is not a 200 (redirects count as failures) or lacks
 * the content that proves it rendered from the model: the landing page and
 * the presets page must print the model's own numbers (computed here with
 * the same code), every chapter must render its MDX (a layer, server-
 * rendered KaTeX and the animation's placeholder), and the header must
 * carry the two-group site switch with Kernels current.
 */
import { fmtRate } from "@/lib/format";
import { derived, preset } from "@/lib/gpu/model";
import { formatValue, lookup } from "@/lib/gpu/values";
import { SECTIONS } from "@/lib/mdx/sections";

type Result = { path: string; ok: boolean; detail: string };

const headers: Record<string, string> = process.env.VERCEL_BYPASS
  ? { "x-vercel-protection-bypass": process.env.VERCEL_BYPASS }
  : {};

const SWITCH = [
  'data-site-switch="full"',
  'data-site-switch="compact"',
  'href="https://gpu-kernels-explained.vercel.app"',
  'href="https://agent-harnesses-explained.vercel.app"',
  'href="https://agent-protocols-explained.vercel.app"',
  'href="https://agent-context-explained.vercel.app"',
];

async function checkPage(
  base: string,
  path: string,
  mustContain: string[],
): Promise<Result> {
  try {
    const res = await fetch(base + path, { redirect: "manual", headers });
    if (res.status !== 200)
      return { path, ok: false, detail: String(res.status) };
    const html = await res.text();
    const missing = mustContain.filter((s) => !html.includes(s));
    if (missing.length)
      return {
        path,
        ok: false,
        detail: `200 but missing ${missing.join(", ")}`,
      };
    return { path, ok: true, detail: "200" };
  } catch (err) {
    return { path, ok: false, detail: (err as Error).message };
  }
}

async function main(): Promise<void> {
  const base = (process.argv[2] ?? "http://localhost:3000").replace(/\/$/, "");
  const a100 = derived(preset("a100"));
  const h100 = derived(preset("h100"));
  const checks: Promise<Result>[] = [
    checkPage(base, "/", [
      "GPU Kernels Explained",
      formatValue(lookup("a100.ridge_fp32"), "num"),
      ...SWITCH,
    ]),
    checkPage(base, "/gpus", [
      'data-status="spec"',
      'data-status="approximation"',
      fmtRate(a100.bw.smem),
      fmtRate(h100.bw.l2),
    ]),
    checkPage(base, "/about", ["The execution model"]),
    checkPage(
      base,
      "/learn",
      SECTIONS.map((x) => x.slug),
    ),
    ...SECTIONS.map((x) =>
      checkPage(base, `/learn/${x.slug}`, [
        'data-layer="concept"',
        "data-pending-widget",
        'class="katex"',
        ...SWITCH,
      ]),
    ),
  ];
  const results = await Promise.all(checks);
  let failed = 0;
  for (const r of results) {
    if (!r.ok) failed++;
    console.log(`${r.ok ? "ok  " : "FAIL"} ${r.path} ${r.detail}`);
  }
  console.log(
    `${results.length - failed}/${results.length} checks passed against ${base}`,
  );
  if (failed) process.exit(1);
}

void main();
