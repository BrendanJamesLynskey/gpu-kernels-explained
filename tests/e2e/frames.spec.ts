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
  dequantCaption,
  flashCaption,
  flowCaption,
  marchCaption,
  occupancyCaption,
  quantCaption,
  reduceCaption,
  simtCaption,
  softmaxCaption,
  splitKCaption,
  sweepCaption,
  timelineCaption,
} from "@/lib/gpu/captions";
import {
  GEMM_MARCH,
  preset,
  type AttentionTraffic,
  type Dequant,
  type FlashStep,
  type MarchStep,
  type OnlineSoftmax,
  type QuantGemm,
  type ReduceResult,
  type SplitK,
  type TimelineStep,
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

/**
 * Change a parameter and wait until the animation has restarted for it
 * (its data-key changes in the reset itself), so a following scrub cannot
 * be undone by the reset.
 */
async function change(fig: Locator, act: () => Promise<void>): Promise<void> {
  const before = (await fig.getAttribute("data-key")) ?? "";
  await act();
  await expect(fig).not.toHaveAttribute("data-key", before);
}

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
  await change(fig, () =>
    fig.getByRole("radio", { name: "GEMM 128×128" }).click(),
  );
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
  await change(fig, () =>
    fig.getByRole("slider", { name: /Stride/ }).fill("8"),
  );
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
  await change(fig, () =>
    fig.getByRole("slider", { name: /Padding/ }).fill("1"),
  );
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

test("GEMM: the shared-memory march, then register blocking", async ({
  page,
}) => {
  await page.goto("/learn/07-gemm");
  const fig = page.getByTestId("gemm-widget");
  const m = fx.gemmMarch.smem as unknown as MarchStep[];
  for (const s of [0, 1, 4, 64]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      marchCaption("smem", GEMM_MARCH.smem, m[s]!, 16, 4),
    );
  }
  await change(fig, () =>
    fig.getByRole("radio", { name: "register blocking" }).click(),
  );
  const r = fx.gemmMarch.regs as unknown as MarchStep[];
  await show(fig, 16);
  await expect(fig.getByTestId("caption")).toHaveText(
    marchCaption("regs", GEMM_MARCH.regs, r[16]!, 16, 4),
  );
  await expect(fig.locator("[data-variant='regs']")).toHaveAttribute(
    "data-current",
    "true",
  );
});

test("reductions: sequential addressing, then shuffles", async ({ page }) => {
  await page.goto("/learn/08-reductions");
  const fig = page.getByTestId("reduction-widget");
  const seq = fx.reduce.sequential as unknown as ReduceResult;
  for (const s of [0, 1, 3, 6]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(reduceCaption(seq, s));
  }
  await change(fig, () =>
    fig.getByRole("radio", { name: "warp shuffle" }).click(),
  );
  const sh = fx.reduce.shuffle as unknown as ReduceResult;
  for (const s of [1, 6]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(reduceCaption(sh, s));
  }
});

test("FlashAttention: N = 1024, d = 64, blocks of 128", async ({ page }) => {
  await page.goto("/learn/09-softmax-and-flashattention");
  const fig = page.getByTestId("flash-widget");
  const f = fx.flash.find((x) => x.args.join() === "1024,64,128,128")!
    .out as unknown as { traffic: AttentionTraffic; steps: FlashStep[] };
  for (const s of [0, 1, 8, 64]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      flashCaption(f.traffic, f.steps[s]!),
    );
  }
  await expect(fig.locator("[data-tile='now']")).toHaveCount(1);
});

test("online softmax: blocks of 4, then 8", async ({ page }) => {
  await page.goto("/learn/09-softmax-and-flashattention");
  const fig = page.getByTestId("softmax-widget");
  const o4 = fx.softmax.find((x) => x.seed === 127 && x.block === 4)!
    .out as unknown as OnlineSoftmax;
  for (const s of [0, 1, 2, 5]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(softmaxCaption(o4, s));
  }
  await change(fig, () =>
    fig.getByRole("radio", { name: "8" }).first().click(),
  );
  const o8 = fx.softmax.find((x) => x.seed === 127 && x.block === 8)!
    .out as unknown as OnlineSoftmax;
  await show(fig, 3);
  await expect(fig.getByTestId("caption")).toHaveText(softmaxCaption(o8, 3));
  await expect(fig.locator("[data-ordinary]")).toHaveCount(16);
});

test("overlap: double buffering the compute-heavy tiles", async ({ page }) => {
  await page.goto("/learn/10-split-k-and-overlap");
  const fig = page.getByTestId("timeline-widget");
  const t = fx.timelineSteps.find((x) => x.args.join() === "6,2,3,1,2")!
    .steps as unknown as TimelineStep[];
  for (const s of [0, 1, 5, t.length - 1]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      timelineCaption(t, s, 6),
    );
  }
  await change(fig, () =>
    fig.getByRole("radio", { name: "1" }).first().click(),
  );
  const one = fx.timelineSteps.find((x) => x.args.join() === "6,2,3,1,1")!
    .steps as unknown as TimelineStep[];
  await show(fig, one.length - 1);
  await expect(fig.getByTestId("caption")).toHaveText(
    timelineCaption(one, one.length - 1, 6),
  );
});

test("split-K on the A100", async ({ page }) => {
  await page.goto("/learn/10-split-k-and-overlap");
  const fig = page.getByTestId("splitk-widget");
  const sw = PY.splitK as unknown as SplitK[];
  for (const s of [0, 5, 6, 9]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      splitKCaption(sw[s]!, sw[0]!, 108),
    );
  }
  await show(fig, 5);
  await expect(fig.locator("[data-sm][data-waves='0']")).toHaveCount(12);
});

test("dequantising in registers, then the quantised layer", async ({
  page,
}) => {
  await page.goto("/learn/11-quantised-kernels");
  const fig = page.getByTestId("dequant-widget");
  const d = fx.dequant.find((x) => x.seed === 11 && x.scale === 0.0625)!
    .out as unknown as Dequant;
  for (const s of [0, 1, 2, 3, 24]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(dequantCaption(d, s));
  }
  const q = page.getByTestId("quant-widget");
  const rows = (i: number) =>
    (["bf16", "int8", "int4", "w8a8"] as const).map(
      (f) => (PY.quant as unknown as Record<string, QuantGemm[]>)[f]![i]!,
    );
  for (const s of [0, 6, 10]) {
    await show(q, s);
    await expect(q.getByTestId("caption")).toHaveText(quantCaption(rows(s)));
  }
});
