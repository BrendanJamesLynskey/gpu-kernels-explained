/**
 * Frame tests (visual standard §4): for key frames of every animation, the
 * state the site draws equals the Python reference's state for that frame
 * (from the fixtures), and the caption built from the reference's state is
 * the caption the site shows. tests/e2e/frames.spec.ts then checks the page.
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/gpu_fixtures.json";

import {
  bankCaption,
  bankStepCount,
  coalesceCaption,
  flowCaption,
  laneRuns,
  occupancyCaption,
  simtCaption,
  sweepCaption,
} from "@/lib/gpu/captions";
import {
  KERNELS,
  bankSteps,
  coalesceSteps,
  derived,
  flowSteps,
  kernelTime,
  occupancySteps,
  preset,
  rooflineSweep,
  simtSteps,
  type BankResult,
  type CoalesceResult,
  type FlowStep,
  type KernelTime,
  type Occupancy,
  type OccupancyStep,
  type SimtResult,
  type SweepStep,
} from "@/lib/gpu/model";

const A100 = preset("a100");
const PY = fx.presets.a100;

describe("chapter 1: memory hierarchy frames", () => {
  for (const kid of ["vecadd", "gemm16", "gemm128"] as const) {
    const ktPy = (PY.kernels as Record<string, unknown>)[kid] as KernelTime;
    const flowPy = (PY.flow as Record<string, unknown>)[kid] as FlowStep[];
    const kt = kernelTime(A100, KERNELS[kid]!);
    const flow = flowSteps(A100, KERNELS[kid]!, 60);
    for (const i of [0, 17, 30, 60]) {
      it(`${kid} step ${i}`, () => {
        expect(flow[i]).toEqual(flowPy[i]);
        expect(flowCaption(kt, flow[i]!, 60)).toBe(
          flowCaption(ktPy, flowPy[i]!, 60),
        );
      });
    }
  }
  it("says what the reference says at the halfway frame", () => {
    const kt = kernelTime(A100, KERNELS.vecadd!);
    const cap = flowCaption(kt, flowSteps(A100, KERNELS.vecadd!, 60)[30]!, 60);
    expect(cap).toBe(
      "t = 1.04 ms of 2.07 ms: HBM has moved 1.61 GB of 3.22 GB. The bottleneck is HBM, busy 100% of the time; Registers 1%, Shared memory 0%, L2 cache 22%; the ALUs 1%.",
    );
    const k2 = kernelTime(A100, KERNELS.gemm128!);
    expect(
      flowCaption(k2, flowSteps(A100, KERNELS.gemm128!, 60)[60]!, 60),
    ).toMatch(/^Done at 7\.05 ms: HBM has moved .* The bottleneck is the ALUs/);
  });
});

describe("chapter 2: roofline sweep frames", () => {
  const sweep = rooflineSweep(A100);
  const py = PY.sweep as SweepStep[];
  const ridge = derived(A100).ridge_fp32;
  for (const i of [0, 4, 6, 7]) {
    it(`tile step ${i}`, () => {
      expect(sweep[i]).toEqual(py[i]);
      expect(sweepCaption(sweep[i]!, ridge)).toBe(sweepCaption(py[i]!, ridge));
    });
  }
  it("crosses the ridge at 64 x 64", () => {
    expect(sweepCaption(sweep[6]!, ridge)).toContain(
      "64×64 tiles (4×4 per thread)",
    );
    expect(sweepCaption(sweep[6]!, ridge)).toContain(
      "compute-bound at 19.5 TFLOP/s",
    );
    expect(sweepCaption(sweep[5]!, ridge)).toContain("memory-bound");
  });
});

function pySimt(cond: string, k: number, lens: number[]): SimtResult {
  const f = fx.simt.find(
    (s) => s.cond === cond && s.k === k && s.lens.join() === lens.join(),
  );
  if (!f) throw new Error("no fixture");
  return f.out as SimtResult;
}

describe("chapter 3: SIMT frames", () => {
  const lens: [number, number, number] = [3, 2, 2];
  for (const [cond, k] of [
    ["lt", 8],
    ["mod", 2],
    ["data", 50],
    ["uniform", 1],
  ] as const) {
    const r = simtSteps(cond, k, ...lens);
    const py = pySimt(cond, k, lens);
    for (const i of [0, 1, 3, r.issued - 1]) {
      it(`${cond} ${k} issue ${i}`, () => {
        expect(r.steps[i]).toEqual(py.steps[i]);
        expect(simtCaption(r, i, lens)).toBe(simtCaption(py, i, lens));
      });
    }
  }
  it("names the masked lanes", () => {
    const r = simtSteps("lt", 8, 3, 2, 2);
    expect(simtCaption(r, 3, [3, 2, 2])).toBe(
      "Issue 4 of 9: the if-path, instruction 2 of 3; 8 of 32 lanes active (0–7); the rest are masked off.",
    );
    expect(simtCaption(r, 1, [3, 2, 2])).toContain("so the warp diverges");
    expect(simtCaption(r, 5, [3, 2, 2])).toContain(
      "the else-path, instruction 1 of 2",
    );
    expect(simtCaption(r, 8, [3, 2, 2])).toContain("after reconvergence");
    expect(simtCaption(r, 0, [3, 2, 2])).toContain("all 32 lanes active");
    expect(simtCaption(r, 99, [3, 2, 2])).toBe("");
    const u = simtSteps("uniform", 1, 3, 2, 2);
    expect(simtCaption(u, 1, [3, 2, 2])).toContain("no divergence");
    expect(laneRuns(0)).toBe("none");
    expect(laneRuns(0x55)).toBe("0, 2, 4, 6");
    expect(laneRuns(0x80000000)).toBe("31");
  });
});

describe("chapter 4: coalescing frames", () => {
  for (const [eb, stride, off] of [
    [4, 1, 0],
    [4, 8, 0],
    [4, 1, 4],
    [16, 1, 0],
  ] as const) {
    const r = coalesceSteps(eb, stride, off);
    const py = fx.coalesce.find(
      (c) => c.eb === eb && c.stride === stride && c.offset === off,
    )!.out as CoalesceResult;
    for (const lane of [0, 7, 8, 31]) {
      it(`${eb} B stride ${stride} offset ${off}, lane ${lane}`, () => {
        expect(r.steps[lane]).toEqual(py.steps[lane]);
        expect(coalesceCaption(r, lane, eb)).toBe(
          coalesceCaption(py, lane, eb),
        );
      });
    }
  }
  it("reports the guide's numbers", () => {
    expect(coalesceCaption(coalesceSteps(4, 8, 0), 31, 4)).toContain(
      "32 sectors (1.02 kB) for 128 B used: 13% efficient",
    );
    expect(coalesceCaption(coalesceSteps(4, 1, 0), 31, 4)).toContain(
      "4 sectors (128 B) for 128 B used: 100% efficient",
    );
    expect(coalesceCaption(coalesceSteps(4, 1, 0), 3, 4)).toContain(
      "already fetched",
    );
    expect(coalesceCaption(coalesceSteps(4, 1, 4), 7, 4)).toContain(
      "sector 1, a new transaction",
    );
    expect(coalesceCaption(coalesceSteps(16, 1, 4), 1, 16)).toContain(
      "partly new",
    );
    expect(coalesceCaption(coalesceSteps(16, 1, 4), 1, 16)).toContain(
      "sectors 0–1",
    );
    expect(coalesceCaption(coalesceSteps(4, 0, 0), 31, 4)).toContain(
      "needs 1 sector (32 B)",
    );
    expect(coalesceCaption(coalesceSteps(4, 1, 0), 40, 4)).toBe("");
  });
});

describe("chapter 5: bank-conflict frames", () => {
  for (const [pattern, pad, stride] of [
    ["col", 0, 1],
    ["col", 1, 1],
    ["stride", 0, 2],
    ["broadcast", 0, 1],
  ] as const) {
    const r = bankSteps(pattern, pad, stride);
    const py = fx.banks.find(
      (b) => b.pattern === pattern && b.pad === pad && b.stride === stride,
    )!.out as BankResult;
    const n = bankStepCount(r);
    for (const i of [0, 12, 31, n - 1]) {
      it(`${pattern} pad ${pad} stride ${stride}, step ${i}`, () => {
        expect(bankStepCount(py)).toBe(n);
        expect(bankCaption(r, i)).toBe(bankCaption(py, i));
      });
    }
  }
  it("tells the transpose story", () => {
    const r = bankSteps("col", 0);
    expect(bankCaption(r, 12)).toBe(
      "Lane 12 asks for word 384: bank 0 (384 mod 32), which now holds 13 different words for this warp.",
    );
    expect(bankCaption(r, 63)).toContain("Pass 32 of 32");
    expect(bankCaption(r, 63)).toContain("A 32-way conflict");
    expect(bankCaption(bankSteps("col", 1), 32)).toContain(
      "32 lanes served (32 of 32 so far). No conflict",
    );
    expect(bankCaption(bankSteps("row", 0), 0)).toContain(
      "holds 1 different word ",
    );
    expect(bankCaption(r, 99)).toBe("");
    expect(bankCaption({ ...r, requests: [] }, 0)).toBe("");
  });
});

describe("chapter 6: occupancy frames", () => {
  for (const pid of ["a100", "h100"] as const) {
    const p = preset(pid);
    for (const f of fx.presets[pid].occupancySteps) {
      const ts = occupancySteps(p, f.threads, f.regs, f.smem);
      const py = f.out as { result: Occupancy; steps: OccupancyStep[] };
      const keys = [0, Math.floor(ts.steps.length / 2), ts.steps.length - 1];
      for (const i of keys) {
        it(`${pid} ${f.threads}/${f.regs}/${f.smem} step ${i}`, () => {
          expect(ts.steps[i]).toEqual(py.steps[i]);
          expect(occupancyCaption(p, ts.result, ts.steps[i]!)).toBe(
            occupancyCaption(p, py.result, py.steps[i]!),
          );
        });
      }
    }
  }
  it("explains the limit", () => {
    const s = occupancySteps(A100, 256, 64, 49152);
    expect(occupancyCaption(A100, s.result, s.steps[3]!)).toBe(
      "Block 4 does not fit (limited by shared memory). 3 blocks, 24 warps resident: occupancy 37.5%.",
    );
    expect(occupancyCaption(A100, s.result, s.steps[0]!)).toMatch(
      /^An empty SM/,
    );
    expect(occupancyCaption(A100, s.result, s.steps[1]!)).toBe(
      "Block 1 placed: 8 of 64 warps, 16,384 of 65,536 registers, 49 KB of 164 KB shared memory in use.",
    );
    const z = occupancySteps(A100, 1024, 255, 0);
    expect(occupancyCaption(A100, z.result, z.steps[0]!)).toBe(
      "No block fits on the SM: it is limited by registers. Occupancy 0%.",
    );
    const one = occupancySteps(A100, 1024, 32, 0);
    expect(occupancyCaption(A100, one.result, one.steps[2]!)).toContain(
      "limited by warp slots and registers",
    );
    const single = occupancySteps(A100, 256, 32, 163 * 1024);
    expect(occupancyCaption(A100, single.result, single.steps[1]!)).toContain(
      "1 block, 8 warps",
    );
  });
});
