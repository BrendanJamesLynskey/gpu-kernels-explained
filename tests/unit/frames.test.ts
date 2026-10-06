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
  dequantCaption,
  flashCaption,
  flowCaption,
  laneRuns,
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
  GEMM_ORDER,
  KERNELS,
  MARCH_BK,
  MARCH_SIZE,
  QUANT_BATCHES,
  QUANT_ORDER,
  REDUCE_KINDS,
  dequantSteps,
  flashSteps,
  gemmMarch,
  onlineSoftmax,
  quantSweep,
  reduceSteps,
  softmaxInputs,
  splitKSweep,
  tileTimeline,
  timelineSteps,
  type AttentionTraffic,
  type Dequant,
  type FlashStep,
  type MarchStep,
  type OnlineSoftmax,
  type QuantGemm,
  type ReduceResult,
  type SplitK,
  type TimelineStep,
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

// --- chapters 7-11 ------------------------------------------------------------

describe("chapter 7: GEMM march frames", () => {
  for (const vid of GEMM_ORDER) {
    const steps = gemmMarch(vid);
    const py = (fx.gemmMarch as Record<string, MarchStep[]>)[vid]!;
    const spec = GEMM_MARCH[vid];
    for (const i of [0, 1, Math.floor(steps.length / 2), steps.length - 1]) {
      it(`${vid} step ${i}`, () => {
        expect(steps[i]).toEqual(py[i]);
        expect(marchCaption(vid, spec, steps[i]!, MARCH_SIZE, MARCH_BK)).toBe(
          marchCaption(vid, spec, py[i]!, MARCH_SIZE, MARCH_BK),
        );
      });
    }
  }
  it("tells the tiling story", () => {
    const s = gemmMarch("smem");
    expect(marchCaption("smem", GEMM_MARCH.smem, s[0]!, 16, 4)).toBe(
      "Start: C = A·B with 16 × 16 matrices, 16 blocks of 4 × 4 outputs, k in tiles of 4.",
    );
    expect(marchCaption("smem", GEMM_MARCH.smem, s[4]!, 16, 4)).toContain(
      "Block (0, 0), k-tile 4 of 4: the block stages a 4 × 4 tile of A",
    );
    expect(marchCaption("smem", GEMM_MARCH.smem, s[4]!, 16, 4)).toContain(
      "The block's C tile is finished and written (64 B).",
    );
    const n = gemmMarch("naive");
    expect(marchCaption("naive", GEMM_MARCH.naive, n[1]!, 16, 4)).toContain(
      "each of its 16 threads reads 4 values of A and 4 of B straight from global memory",
    );
    expect(marchCaption("naive", GEMM_MARCH.naive, n[1]!, 16, 4)).not.toContain(
      "shared memory",
    );
    const t = gemmMarch("tensor");
    expect(marchCaption("tensor", GEMM_MARCH.tensor, t[16]!, 16, 4)).toContain(
      "2.67 flop/byte",
    );
    expect(
      marchCaption("regs", GEMM_MARCH.regs, gemmMarch("regs")[1]!, 16, 4),
    ).toContain("each thread loads 2 + 2 values into registers");
  });
});

describe("chapter 8: reduction frames", () => {
  for (const k of REDUCE_KINDS) {
    const r = reduceSteps(k);
    const py = (fx.reduce as Record<string, ReduceResult>)[k]!;
    for (const i of [0, 1, 3, r.steps.length - 1]) {
      it(`${k} step ${i}`, () => {
        expect(r.steps[i]).toEqual(py.steps[i]);
        expect(reduceCaption(r, i)).toBe(reduceCaption(py, i));
      });
    }
  }
  it("names divergence, conflicts and the sum", () => {
    expect(reduceCaption(reduceSteps("divergent"), 1)).toContain(
      "2 of 2 active warps diverge",
    );
    expect(reduceCaption(reduceSteps("strided"), 1)).toContain(
      "a 2-way bank conflict",
    );
    const seq = reduceSteps("sequential");
    expect(reduceCaption(seq, 1)).toContain(
      "threads 0–31 add x[tid + 32] (32 threads); 1 warp fully active, no divergence; no bank conflict.",
    );
    expect(reduceCaption(seq, 6)).toContain("Done: x[0] = 2,888, the sum.");
    expect(reduceCaption(seq, 0)).toContain("their sum is 2,888");
    const sh = reduceSteps("shuffle");
    expect(reduceCaption(sh, 1)).toContain("__shfl_down_sync by 16");
    expect(reduceCaption(sh, 6)).toContain("thread 0 adds the 2 partial sums");
    expect(reduceCaption(sh, 9)).toBe("");
  });
});

describe("chapter 9: softmax and FlashAttention frames", () => {
  for (const f of fx.softmax.filter((s) => s.seed === 127)) {
    const o = onlineSoftmax(softmaxInputs(16, f.seed), f.block);
    const py = f.out as OnlineSoftmax;
    const n = o.steps.length + 2;
    for (const i of [0, 1, n - 2, n - 1]) {
      it(`softmax block ${f.block}, step ${i}`, () => {
        if (i >= 1 && i <= o.steps.length) {
          const a = o.steps[i - 1]!;
          const b = py.steps[i - 1]!;
          expect(a.lo).toBe(b.lo);
          expect(a.m).toBe(b.m); // maxima are exact (no exp)
          expect(Math.abs(a.l - b.l)).toBeLessThan(1e-13);
        }
        expect(softmaxCaption(o, i)).toBe(softmaxCaption(py, i));
      });
    }
  }
  it("tells the rescaling story", () => {
    const o = onlineSoftmax(softmaxInputs(16, 127), 4);
    expect(softmaxCaption(o, 0)).toBe(
      "16 scores, read in 4 blocks of 4. Start with m = −∞ and ℓ = 0.",
    );
    expect(softmaxCaption(o, 1)).toMatch(
      /^Block 1 of 4 \(x0–x3\): the running max becomes 1\.8/,
    );
    expect(softmaxCaption(o, 2)).toContain(
      "is rescaled by e^(1.8 − 2.3) = 0.607",
    );
    expect(softmaxCaption(o, 5)).toMatch(/^Normalise: /);
    expect(softmaxCaption(o, 9)).toBe("");
    const flat = onlineSoftmax([3, 1, 2, 0], 2);
    expect(softmaxCaption(flat, 2)).toContain("does not beat 3");
  });
  for (const f of fx.flash) {
    const [n, d, br, bc] = f.args as [number, number, number, number];
    const r = flashSteps(n, d, br, bc);
    const py = f.out as { traffic: AttentionTraffic; steps: FlashStep[] };
    const last = r.steps.length - 1;
    for (const i of [0, 1, Math.floor(last / 2), last]) {
      it(`flash N=${n} d=${d} B=${br}, step ${i}`, () => {
        expect(r.steps[i]).toEqual(py.steps[i]);
        expect(flashCaption(r.traffic, r.steps[i]!)).toBe(
          flashCaption(py.traffic, py.steps[i]!),
        );
      });
    }
  }
  it("names the loads and the never-stored score tile", () => {
    const r = flashSteps(512, 64, 64, 64);
    expect(flashCaption(r.traffic, r.steps[1]!)).toContain(
      "Q block 1 is loaded (8.19 kB), then K and V block 1 come in (16.4 kB); the 64 × 64 score tile stays on chip.",
    );
    expect(flashCaption(r.traffic, r.steps[8]!)).toContain(
      "Row done: O block 1 is written once",
    );
    expect(flashCaption(r.traffic, r.steps[0]!)).toContain(
      "8 query blocks × 8 key blocks of 64",
    );
  });
});

describe("chapter 10: overlap and split-K frames", () => {
  for (const t of fx.timelineSteps) {
    const [n, tl, tc, ts, b] = t.args as [
      number,
      number,
      number,
      number,
      number,
    ];
    const steps = timelineSteps(tileTimeline(n, tl, tc, ts, b));
    const py = t.steps as TimelineStep[];
    for (const i of [0, 1, Math.floor(steps.length / 2), steps.length - 1]) {
      it(`timeline ${t.args.join(",")} event ${i}`, () => {
        expect(steps[i]).toEqual(py[i]);
        expect(timelineCaption(steps, i, n)).toBe(timelineCaption(py, i, n));
      });
    }
  }
  it("says what each engine does", () => {
    const s = timelineSteps(tileTimeline(6, 2, 3, 1, 2));
    expect(timelineCaption(s, 1, 6)).toBe(
      "t = 2: copy-in loads tile 2, compute works on tile 1; 0 of 6 tiles stored. Compute busy so far: 0%.",
    );
    expect(timelineCaption(s, s.length - 1, 6)).toBe(
      "t = 21: all 6 tiles stored. The compute engine was busy 86% of the time.",
    );
    expect(timelineCaption(s, 99, 6)).toBe("");
    const one = timelineSteps(tileTimeline(1, 1, 1, 1, 1));
    expect(
      one.some((x) =>
        timelineCaption(one, one.indexOf(x), 1).includes("every engine waits"),
      ),
    ).toBe(false);
  });
  for (const pid of ["a100", "h100"] as const) {
    const p = preset(pid);
    const sweep = splitKSweep(p);
    const py = fx.presets[pid].splitK as SplitK[];
    for (const i of [0, 5, 6, sweep.length - 1]) {
      it(`${pid} split-K step ${i}`, () => {
        expect(sweep[i]).toEqual(py[i]);
        expect(splitKCaption(sweep[i]!, sweep[0]!, p.sms)).toBe(
          splitKCaption(py[i]!, py[0]!, p.sms),
        );
      });
    }
  }
  it("tells the wave story", () => {
    const s = splitKSweep(A100);
    expect(splitKCaption(s[0]!, s[0]!, 108)).toBe(
      "No split: 16 output tiles make 16 blocks on 108 SMs: 1 wave, 15% of SMs busy. 186 µs computing = 186 µs.",
    );
    expect(splitKCaption(s[5]!, s[0]!, 108)).toContain(
      "6 splits: 16 tiles × 6 = 96 blocks on 108 SMs: 1 wave, 89% of SMs busy.",
    );
    expect(splitKCaption(s[6]!, s[0]!, 108)).toContain("2 waves");
  });
});

describe("chapter 11: dequantise and quantised-layer frames", () => {
  for (const f of fx.dequant) {
    const d = dequantSteps(f.seed, f.scale);
    const py = f.out as Dequant;
    for (const i of [0, 1, 2, 3, 13, d.steps.length - 1]) {
      it(`dequant seed ${f.seed} scale ${f.scale}, step ${i}`, () => {
        expect(d.steps[i]).toEqual(py.steps[i]);
        expect(dequantCaption(d, i)).toBe(dequantCaption(py, i));
      });
    }
  }
  it("walks one weight through the three instructions", () => {
    const d = dequantSteps();
    expect(dequantCaption(d, 0)).toBe(
      "One 32-bit load brings eight 4-bit weights into a register: 0x3FF73EBB. The group's scale is 0.0625.",
    );
    expect(dequantCaption(d, 1)).toBe(
      "Weight 1 of 8: shift right by 0 and mask with 0xF: q = 11 (bits 0–3).",
    );
    expect(dequantCaption(d, 2)).toBe(
      "Weight 1 of 8: dequantise, w = (11 − 8) × 0.0625 = 0.1875.",
    );
    expect(dequantCaption(d, 3)).toBe(
      "Weight 1 of 8: fused multiply-add with the activation −4: acc = 0 + 0.1875 × (−4) = −0.75.",
    );
    expect(dequantCaption(d, 99)).toBe("");
  });
  for (const pid of ["a100", "h100"] as const) {
    const p = preset(pid);
    const sweeps = QUANT_ORDER.map((f) => quantSweep(p, f));
    const py = QUANT_ORDER.map(
      (f) => (fx.presets[pid].quant as Record<string, QuantGemm[]>)[f]!,
    );
    for (const i of [0, 6, QUANT_BATCHES.length - 1]) {
      it(`${pid} quantised layer, batch index ${i}`, () => {
        const rows = sweeps.map((s) => s[i]!);
        const pyRows = py.map((s) => s[i]!);
        expect(rows).toEqual(pyRows);
        expect(quantCaption(rows)).toBe(quantCaption(pyRows));
      });
    }
  }
  it("compares the formats at batch 1", () => {
    const rows = QUANT_ORDER.map((f) => quantSweep(A100, f)[0]!);
    expect(quantCaption(rows)).toBe(
      "1 token at once: BF16 86.3 µs, memory-bound; INT8 (W8A16) 43.2 µs, memory-bound, 2× BF16's speed; INT4 (W4A16) 22.3 µs, memory-bound, 3.88× BF16's speed; W8A8 43.2 µs, memory-bound, 2× BF16's speed.",
    );
    expect(quantCaption([])).toBe("");
  });
});
