/**
 * Frame tests on the page (visual standard §4): set key frames of every
 * animation and require the caption on screen to be the caption built from
 * the Python reference's state for that frame (tests/fixtures).
 */
import { expect, test, type Locator } from "@playwright/test";

import fx from "../fixtures/gpu_fixtures.json";

import {
  bankCaption,
  coalesceCaption,
  flowCaption,
  occupancyCaption,
  simtCaption,
  sweepCaption,
} from "@/lib/gpu/captions";
import {
  preset,
  type BankResult,
  type CoalesceResult,
  type FlowStep,
  type KernelTime,
  type Occupancy,
  type OccupancyStep,
  type SimtResult,
  type SweepStep,
} from "@/lib/gpu/model";

const PY = fx.presets.a100;

// No autoplay (reduced motion): the test sets each frame itself.
test.use({ contextOptions: { reducedMotion: "reduce" } });
const A100 = preset("a100");

async function show(fig: Locator, s: number): Promise<void> {
  // pause first: the animation starts by itself when it scrolls into view
  if ((await fig.getAttribute("data-playing")) === "true")
    await fig.getByTestId("play").click();
  await fig.getByTestId("scrub").fill(String(s));
  await expect(fig).toHaveAttribute("data-step", String(s));
  await expect(fig).toHaveAttribute("data-playing", "false");
}

test("memory hierarchy: vector add on the A100", async ({ page }) => {
  await page.goto("/learn/01-memory-hierarchy");
  const fig = page.getByTestId("hierarchy-widget");
  const kt = PY.kernels.vecadd as unknown as KernelTime;
  const flow = PY.flow.vecadd as unknown as FlowStep[];
  for (const s of [0, 17, 30, 60]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      flowCaption(kt, flow[s]!, 60),
    );
  }
  // switch kernel: the model re-runs and the animation restarts
  await fig.getByRole("radio", { name: "GEMM 128×128" }).click();
  await show(fig, 60);
  await expect(fig.getByTestId("caption")).toHaveText(
    flowCaption(
      PY.kernels.gemm128 as unknown as KernelTime,
      (PY.flow.gemm128 as unknown as FlowStep[])[60]!,
      60,
    ),
  );
});

test("roofline: the tile sweep on the A100", async ({ page }) => {
  await page.goto("/learn/02-roofline");
  const fig = page.getByTestId("roofline-widget");
  const sweep = PY.sweep as unknown as SweepStep[];
  for (const s of [0, 4, 6, 7]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      sweepCaption(sweep[s]!, PY.derived.ridge_fp32),
    );
  }
});

test("SIMT: i < 8 with paths of 3 and 2", async ({ page }) => {
  await page.goto("/learn/03-warps-and-divergence");
  const fig = page.getByTestId("simt-widget");
  const r = fx.simt.find(
    (x) => x.cond === "lt" && x.k === 8 && x.lens.join() === "3,2,2",
  )!.out as unknown as SimtResult;
  for (const s of [0, 1, 3, 5, 8]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      simtCaption(r, s, [3, 2, 2]),
    );
  }
  await expect(fig.locator("pre [data-current='true']")).toHaveText(
    "c = h1(a, b);",
  );
});

test("coalescing: stride 1, then stride 8", async ({ page }) => {
  await page.goto("/learn/04-coalescing");
  const fig = page.getByTestId("coalescing-widget");
  const r1 = fx.coalesce.find(
    (c) => c.eb === 4 && c.stride === 1 && c.offset === 0,
  )!.out as unknown as CoalesceResult;
  for (const s of [0, 7, 8, 31]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      coalesceCaption(r1, s, 4),
    );
  }
  await fig.getByRole("slider", { name: /Stride/ }).fill("8");
  const r8 = fx.coalesce.find(
    (c) => c.eb === 4 && c.stride === 8 && c.offset === 0,
  )!.out as unknown as CoalesceResult;
  await show(fig, 31);
  await expect(fig.getByTestId("caption")).toHaveText(
    coalesceCaption(r8, 31, 4),
  );
  // 32 sectors, one per lane, in 8 lines of 4
  await expect(fig.locator("[data-sector]")).toHaveCount(32);
  await expect(fig.locator("[data-state='untouched']")).toHaveCount(0);
});

test("bank conflicts: the transpose column, then padded", async ({ page }) => {
  await page.goto("/learn/05-bank-conflicts");
  const fig = page.getByTestId("bank-widget");
  const r0 = fx.banks.find((b) => b.pattern === "col" && b.pad === 0)!
    .out as unknown as BankResult;
  for (const s of [0, 12, 31, 40, 63]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(bankCaption(r0, s));
  }
  await expect(fig.locator("[data-bank='0']")).toHaveAttribute(
    "data-queued",
    "0",
  );
  await fig.getByRole("slider", { name: /Padding/ }).fill("1");
  const r1 = fx.banks.find((b) => b.pattern === "col" && b.pad === 1)!
    .out as unknown as BankResult;
  await show(fig, 32);
  await expect(fig.getByTestId("caption")).toHaveText(bankCaption(r1, 32));
});

test("occupancy: 256 threads, 64 registers, 48 KB on the A100", async ({
  page,
}) => {
  await page.goto("/learn/06-occupancy");
  const fig = page.getByTestId("occupancy-widget");
  const f = PY.occupancySteps.find(
    (o) => o.threads === 256 && o.regs === 64 && o.smem === 49152,
  )!.out as unknown as { result: Occupancy; steps: OccupancyStep[] };
  for (const s of [0, 1, 2, 3]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      occupancyCaption(A100, f.result, f.steps[s]!),
    );
  }
  await expect(fig.locator("[data-resource='smem']")).toHaveAttribute(
    "data-limit",
    "true",
  );
});
