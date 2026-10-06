/**
 * The TypeScript model against the Python reference's fixtures
 * (scripts/make_fixtures.py): every value exactly, no tolerance.
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/gpu_fixtures.json";

import {
  GEMM_ORDER,
  KERNELS,
  attentionTraffic,
  bankSteps,
  dequantSteps,
  flashSteps,
  gemmMarch,
  gemmVariant,
  intValues,
  onlineSoftmax,
  quantSweep,
  reduceSteps,
  softmaxInputs,
  splitKSweep,
  timelineSteps,
  QUANT_ORDER,
  REDUCE_KINDS,
  coalesceSteps,
  derived,
  flowSteps,
  kernelTime,
  laneData,
  occupancy,
  occupancySteps,
  preset,
  rooflinePoint,
  rooflineSweep,
  simtSteps,
  tileTimeline,
  type BankPattern,
  type Cond,
  type Engine,
  type PresetId,
} from "@/lib/gpu/model";

type Fx = typeof fx;
const presets = fx.presets as unknown as Record<
  PresetId,
  Fx["presets"]["a100"]
>;

for (const pid of ["a100", "h100"] as const) {
  const p = preset(pid);
  const f = presets[pid];

  describe(`${pid}: model parity`, () => {
    it("flattens the preset identically", () => {
      expect(p).toEqual(f.flat);
    });
    it("derives the same bandwidths and ridges", () => {
      expect(derived(p)).toEqual(f.derived);
    });
    it("times every kernel identically", () => {
      for (const [kid, k] of Object.entries(KERNELS))
        expect(kernelTime(p, k), kid).toEqual(
          (f.kernels as Record<string, unknown>)[kid],
        );
    });
    it("produces the same flow animation states", () => {
      for (const [kid, k] of Object.entries(KERNELS))
        expect(flowSteps(p, k, 60), kid).toEqual(
          (f.flow as Record<string, unknown>)[kid],
        );
    });
    it("places every roofline point identically", () => {
      for (const r of f.roofline)
        expect(rooflinePoint(p, r.ai, r.engine as Engine)).toEqual(r.out);
    });
    it("sweeps the tile sizes identically", () => {
      expect(rooflineSweep(p)).toEqual(f.sweep);
    });
    it(`computes occupancy for all ${f.occupancy.length} configurations`, () => {
      for (const o of f.occupancy)
        expect(occupancy(p, o.threads, o.regs, o.smem)).toEqual(o.out);
    });
    it("costs the four GEMM variants identically", () => {
      for (const v of GEMM_ORDER)
        expect(gemmVariant(p, v), v).toEqual(
          (f.gemmVariants as Record<string, unknown>)[v],
        );
    });
    it("sweeps split-K identically", () => {
      expect(splitKSweep(p)).toEqual(f.splitK);
    });
    it("sweeps every quantised format identically", () => {
      for (const q of QUANT_ORDER)
        expect(quantSweep(p, q), q).toEqual(
          (f.quant as Record<string, unknown>)[q],
        );
    });
    it("steps the block placement identically", () => {
      for (const o of f.occupancySteps)
        expect(occupancySteps(p, o.threads, o.regs, o.smem)).toEqual(o.out);
    });
  });
}

describe("SIMT, coalescing, banks and timelines: parity", () => {
  it("draws the same lane data", () => {
    expect(laneData(2024)).toEqual(fx.laneData);
  });
  it(`traces all ${fx.simt.length} branch cases identically`, () => {
    for (const s of fx.simt) {
      const [a, b, c] = s.lens as [number, number, number];
      expect(simtSteps(s.cond as Cond, s.k, a, b, c)).toEqual(s.out);
    }
  });
  it(`coalesces all ${fx.coalesce.length} access patterns identically`, () => {
    for (const c of fx.coalesce)
      expect(coalesceSteps(c.eb, c.stride, c.offset)).toEqual(c.out);
  });
  it(`counts conflicts for all ${fx.banks.length} bank patterns identically`, () => {
    for (const b of fx.banks)
      expect(bankSteps(b.pattern as BankPattern, b.pad, b.stride)).toEqual(
        b.out,
      );
  });
  it("schedules the tile timelines identically", () => {
    for (const t of fx.timeline) {
      const [n, tl, tc, ts, buf] = t.args as [
        number,
        number,
        number,
        number,
        number,
      ];
      expect(tileTimeline(n, tl, tc, ts, buf)).toEqual(t.out);
    }
  });
});

/**
 * Deep comparison with a relative tolerance for numbers: the softmax uses
 * exp, and V8's Math.exp and the C library's exp may differ in the last
 * bit (house rule: transcendentals with a tolerance, everything else exact).
 */
function expectClose(a: unknown, b: unknown, path = "$"): void {
  if (typeof a === "number" && typeof b === "number") {
    const tol = 1e-14 * Math.max(1, Math.abs(a), Math.abs(b));
    if (Math.abs(a - b) > tol) expect(a, path).toBe(b);
    return;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    expect(a.length, path).toBe(b.length);
    a.forEach((x, i) => expectClose(x, b[i], `${path}[${i}]`));
    return;
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    expect(Object.keys(a).sort(), path).toEqual(Object.keys(b).sort());
    for (const k of Object.keys(a))
      expectClose(
        (a as Record<string, unknown>)[k],
        (b as Record<string, unknown>)[k],
        `${path}.${k}`,
      );
    return;
  }
  expect(a, path).toEqual(b);
}

describe("chapters 7-11: parity", () => {
  it("steps the overlap timelines identically", () => {
    for (const t of fx.timelineSteps) {
      const [n, tl, tc, ts, b] = t.args as [
        number,
        number,
        number,
        number,
        number,
      ];
      const out = tileTimeline(n, tl, tc, ts, b);
      expect(out).toEqual(t.timeline);
      expect(timelineSteps(out)).toEqual(t.steps);
    }
  });
  it("marches the GEMM tiles identically", () => {
    for (const v of GEMM_ORDER)
      expect(gemmMarch(v), v).toEqual(
        (fx.gemmMarch as Record<string, unknown>)[v],
      );
  });
  it("reduces identically, every way", () => {
    for (const k of REDUCE_KINDS)
      expect(reduceSteps(k), k).toEqual(
        (fx.reduce as Record<string, unknown>)[k],
      );
    expect(intValues(32, 2024)).toEqual(fx.laneData);
  });
  it("draws the same softmax scores (exactly)", () => {
    for (const s of fx.softmax)
      expect(softmaxInputs(16, s.seed)).toEqual(s.out.x);
  });
  it(`runs all ${fx.softmax.length} online softmaxes (within 1e-14 relative)`, () => {
    for (const s of fx.softmax)
      expectClose(onlineSoftmax(softmaxInputs(16, s.seed), s.block), s.out);
  });
  it("counts FlashAttention's traffic identically", () => {
    for (const f of fx.flash) {
      const [n, d, br, bc] = f.args as [number, number, number, number];
      expect(flashSteps(n, d, br, bc)).toEqual(f.out);
    }
    for (const a of fx.attention) {
      const [n, d, br, bc] = a.args as [number, number, number, number];
      expect(attentionTraffic(n, d, br, bc)).toEqual(a.out);
    }
  });
  it("dequantises identically", () => {
    for (const d of fx.dequant)
      expect(dequantSteps(d.seed, d.scale)).toEqual(d.out);
  });
});
