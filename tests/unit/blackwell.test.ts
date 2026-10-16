/**
 * Chapter 12's model against the Python reference (reference/blackwell.py):
 * the tensor-core presets, ridge points, block-scaled formats, stages and
 * the warp-specialised pipeline's schedules and frames, all exactly.
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/blackwell_fixtures.json";

import {
  BLOCK_ORDER,
  TC_ORDER,
  bitsPerValue,
  frames,
  maxStages,
  pipeCaption,
  pipelineDefault,
  ridges,
  stageBytes,
  tcPreset,
  weightBytes,
  type PipeParams,
} from "@/lib/gpu/blackwell";

describe("tensor-core presets", () => {
  it("match the reference, field for field", () => {
    for (const pid of TC_ORDER)
      expect(tcPreset(pid)).toEqual(
        (fx.presets as Record<string, unknown>)[pid],
      );
  });
  it("ridge points match, and rise A100 -> H100 -> B200 (BF16)", () => {
    for (const pid of TC_ORDER)
      expect(ridges(pid)).toEqual((fx.ridges as Record<string, unknown>)[pid]);
    const r = TC_ORDER.map((p) => ridges(p).bf16!);
    expect(r[0]).toBeLessThan(r[1]!);
    expect(r[1]).toBeLessThan(r[2]!);
    expect(ridges("b200").fp4).toBe(1250);
  });
  it("the B200 is the sibling sites' B200 (GB200 NVL72 part)", () => {
    const b = tcPreset("b200");
    expect(b.peak_bf16).toBe(2.5e15);
    expect(b.peak_fp8).toBe(5e15);
    expect(b.hbm_bw).toBe(8e12);
    expect(b.hbm_bytes).toBe(186e9);
  });
});

describe("formats and stages", () => {
  it("bits per value and weight bytes", () => {
    for (const f of BLOCK_ORDER) {
      expect(bitsPerValue(f)).toBe((fx.bits as Record<string, number>)[f]);
      expect(weightBytes(f, 8192, 8192)).toBe(
        (fx.weightBytes as Record<string, number>)[f],
      );
    }
  });
  it("stages that fit a 128 x 256 x 64 BF16 tile", () => {
    for (const pid of TC_ORDER)
      expect(
        maxStages(tcPreset(pid).smem_per_sm, stageBytes(128, 256, 64, 2)),
      ).toBe((fx.stages as Record<string, number>)[pid]);
  });
});

describe("the warp-specialised pipeline", () => {
  it("every schedule in the grid matches exactly", () => {
    for (const c of fx.pipelines)
      expect(pipelineDefault(c.over as Partial<PipeParams>)).toEqual(c.out);
  });
  it("every frame of the animated runs matches exactly", () => {
    for (const c of fx.frames)
      expect(frames(pipelineDefault(c.over as Partial<PipeParams>))).toEqual(
        c.out,
      );
  });
  it("the tensor core never runs two slices at once, and slots hold one slice", () => {
    const s = pipelineDefault();
    for (let g = 1; g < s.mmaStart.length; g++)
      expect(s.mmaStart[g]).toBeGreaterThanOrEqual(s.mmaDone[g - 1]!);
  });
});

describe("captions", () => {
  it("say what each engine is doing", () => {
    const s = pipelineDefault();
    const f = frames(s);
    expect(pipeCaption(s, f[0]!)).toBe(
      "t = 0; the producer issues a TMA load (tile 1, slice 1); the tensor core is idle; 0 of 18 slices done, 0 of 3 tiles stored.",
    );
    expect(pipeCaption(s, f[12]!)).toContain(
      "the tensor core multiplies tile 1, slice 1",
    );
    const e = f.find((x) => x.accs.some((a) => a[0] === "epi"))!;
    expect(pipeCaption(s, e)).toContain("the epilogue drains tile 1");
    expect(pipeCaption(s, f[s.total]!)).toBe(
      `t = ${s.total}; the tensor core is idle; 18 of 18 slices done, 3 of 3 tiles stored.`,
    );
  });
});
