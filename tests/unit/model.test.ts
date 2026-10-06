/**
 * The TypeScript model against the Python reference's fixtures
 * (scripts/make_fixtures.py): every value exactly, no tolerance.
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/gpu_fixtures.json";

import {
  KERNELS,
  bankSteps,
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
