/**
 * Captures the README screenshots. Manual run, output committed.
 *
 *   pnpm build && pnpm start   # in another shell
 *   pnpm screenshots           # headless Chromium writes docs/screenshots/*.png
 *
 * Light theme, fixed viewport, reduced motion (nothing plays by itself), and
 * each animation set to a chosen mid-animation frame by its scrub bar, so
 * the pictures are reproducible.
 */
import { mkdirSync } from "node:fs";
import path from "node:path";

import { chromium } from "@playwright/test";

const OUT = path.join(process.cwd(), "docs", "screenshots");
const BASE = process.env.SCREENSHOT_BASE_URL ?? "http://localhost:3000";

type Shot = { name: string; path: string; widget?: string; step?: number };

const SHOTS: Shot[] = [
  { name: "01-landing", path: "/" },
  {
    name: "02-memory-hierarchy",
    path: "/learn/01-memory-hierarchy",
    widget: "hierarchy-widget",
    step: 30,
  },
  {
    name: "03-roofline",
    path: "/learn/02-roofline",
    widget: "roofline-widget",
    step: 6,
  },
  {
    name: "04-divergence",
    path: "/learn/03-warps-and-divergence",
    widget: "simt-widget",
    step: 5,
  },
  {
    name: "05-coalescing",
    path: "/learn/04-coalescing",
    widget: "coalescing-widget",
    step: 12,
  },
  {
    name: "06-bank-conflicts",
    path: "/learn/05-bank-conflicts",
    widget: "bank-widget",
    step: 20,
  },
  {
    name: "07-occupancy",
    path: "/learn/06-occupancy",
    widget: "occupancy-widget",
    step: 3,
  },
  { name: "08-gpus", path: "/gpus" },
  { name: "09-gemm", path: "/learn/07-gemm", widget: "gemm-widget", step: 22 },
  {
    name: "10-reductions",
    path: "/learn/08-reductions",
    widget: "reduction-widget",
    step: 2,
  },
  {
    name: "11-flashattention",
    path: "/learn/09-softmax-and-flashattention",
    widget: "flash-widget",
    step: 30,
  },
  {
    name: "12-online-softmax",
    path: "/learn/09-softmax-and-flashattention",
    widget: "softmax-widget",
    step: 2,
  },
  {
    name: "13-overlap",
    path: "/learn/10-split-k-and-overlap",
    widget: "timeline-widget",
    step: 7,
  },
  {
    name: "14-split-k",
    path: "/learn/10-split-k-and-overlap",
    widget: "splitk-widget",
    step: 5,
  },
  {
    name: "15-dequantise",
    path: "/learn/11-quantised-kernels",
    widget: "dequant-widget",
    step: 12,
  },
  {
    name: "16-quantised-layer",
    path: "/learn/11-quantised-kernels",
    widget: "quant-widget",
    step: 6,
  },
];

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    colorScheme: "light",
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  for (const s of SHOTS) {
    await page.goto(BASE + s.path, { waitUntil: "networkidle" });
    const file = path.join(OUT, `${s.name}.png`);
    if (s.widget) {
      const fig = page.getByTestId(s.widget);
      await fig.waitFor();
      if (s.step !== undefined)
        await fig.getByTestId("scrub").fill(String(s.step));
      await fig.screenshot({ path: file });
    } else {
      await page.screenshot({ path: file });
    }
    console.log(`wrote ${path.relative(process.cwd(), file)}`);
  }
  await browser.close();
}

void main();
