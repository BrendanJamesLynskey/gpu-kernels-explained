/**
 * The numbers the chapters print: every <V of="…" /> path in the MDX
 * resolves in the model, and the formatters behave.
 */
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  fmtAi,
  fmtBytes,
  fmtFlops,
  fmtKiB,
  fmtRate,
  fmtTime,
  pct,
  trim,
} from "@/lib/format";
import {
  QUANT_SHAPE,
  preset,
  quantSweep,
  splitKSweep,
  tileTimeline,
} from "@/lib/gpu/model";
import { formatValue, lookup, type Fmt } from "@/lib/gpu/values";

const DIR = path.join(__dirname, "../../content/chapters");
const FILES = readdirSync(DIR).filter((f) => f.endsWith(".mdx"));

describe("model values quoted in the chapters", () => {
  const uses: { file: string; of: string; fmt: string }[] = [];
  for (const f of FILES) {
    const src = readFileSync(path.join(DIR, f), "utf8");
    for (const m of src.matchAll(/<V of="([^"]+)"(?: fmt="([^"]+)")? \/>/g))
      uses.push({ file: f, of: m[1]!, fmt: m[2] ?? "num" });
  }
  it("are used", () => expect(uses.length).toBeGreaterThan(30));
  it("all resolve to a number or string", () => {
    for (const u of uses) {
      const v = lookup(u.of);
      expect(["number", "string"], `${u.file}: ${u.of}`).toContain(typeof v);
      expect(formatValue(v, u.fmt as Fmt).length).toBeGreaterThan(0);
    }
  });
  it("matches the sources where they state a number", () => {
    expect(lookup("a100.bw.hbm")).toBe(1555e9);
    expect(formatValue(lookup("a100.bw.hbm"), "rate")).toBe("1.555 TB/s");
    expect(formatValue(lookup("a100.bw.l2"), "rate")).toBe("7.219 TB/s");
    expect(formatValue(lookup("a100.ridge_fp32"), "num")).toBe("12.5");
    expect(formatValue(lookup("a100.kernels.vecadd.total"), "time")).toBe(
      "2.07 ms",
    );
    expect(lookup("h100.kernels.gemm128.bound")).toBe("smem");
    expect(formatValue(lookup("a100.mma_shape"), "raw")).toContain("m16n8k16");
  });
  it("rejects bad paths", () => {
    expect(() => lookup("v100.bw.hbm")).toThrow();
    expect(() => lookup("a100.bw.nope")).toThrow();
    expect(() => lookup("a100.bw")).toThrow();
  });
});

describe("formatting", () => {
  it("formats every kind", () => {
    expect(formatValue(1.5e12, "flops")).toBe("1.5 TFLOP/s");
    expect(formatValue(2048, "bytes")).toBe("2.05 kB");
    expect(formatValue(2048, "kib")).toBe("2 KB");
    expect(formatValue(0.375, "pct")).toBe("38%");
    expect(formatValue(12.54, "ai")).toBe("12.5 flop/byte");
    expect(formatValue(65536, "int")).toBe("65,536");
    expect(formatValue(3.14159, "raw")).toBe("3.14159");
    expect(formatValue(3.14159, "time")).toBe("3.14 s");
    expect(formatValue(1e12, "rate")).toBe("1 TB/s");
    expect(formatValue("x", "num")).toBe("x");
  });
  it("covers the unit ranges", () => {
    expect(trim(0)).toBe("0");
    expect(fmtBytes(5)).toBe("5 B");
    expect(fmtBytes(0.5)).toBe("0.5 B");
    expect(fmtBytes(3e6)).toBe("3 MB");
    expect(fmtBytes(3e9)).toBe("3 GB");
    expect(fmtBytes(3e12)).toBe("3 TB");
    expect(fmtKiB(5)).toBe("5 B");
    expect(fmtKiB(40 * 1024 * 1024)).toBe("40 MB");
    expect(fmtRate(5e9)).toBe("5 GB/s");
    expect(fmtRate(5e6)).toBe("5 MB/s");
    expect(fmtRate(5)).toBe("5 B/s");
    expect(fmtFlops(5e9)).toBe("5 GFLOP/s");
    expect(fmtFlops(5e6)).toBe("5 MFLOP/s");
    expect(fmtFlops(5)).toBe("5 FLOP/s");
    expect(fmtTime(0)).toBe("0 s");
    expect(fmtTime(2.5e-6)).toBe("2.5 µs");
    expect(fmtTime(2.5e-9)).toBe("2.5 ns");
    expect(pct(0.3751, 1)).toBe("37.5%");
    expect(fmtAi(0.25)).toBe("0.25 flop/byte");
  });
});

/**
 * Numbers the chapters 7-11 state in words (not through <V />): each is
 * recomputed here from the model, so the prose cannot drift from it.
 */
describe("numbers written into the prose of chapters 7-11", () => {
  const A100 = preset("a100");
  const H100 = preset("h100");
  it("chapter 7: the tensor roof is 16 times the FP32 roof on the A100", () => {
    expect(A100.peak_tensor / A100.peak_fp32).toBe(16);
  });
  it("chapter 8: under 17% of threads busy on average for n = 64", () => {
    const n = 64;
    const share = (1 - 1 / n) / Math.log2(n);
    expect(share).toBeLessThan(0.17);
    expect(share).toBeGreaterThan(0.16);
  });
  it("chapter 10: the overlap example and the split-K optimum", () => {
    const t1 = tileTimeline(6, 2, 3, 1, 1).total;
    const t2 = tileTimeline(6, 2, 3, 1, 2).total;
    expect([t1, t2]).toEqual([31, 21]);
    expect(trim(t1 / t2, 3)).toBe("1.48");
    for (const [a, b, c] of [
      [2, 3, 1],
      [3, 1.5, 1],
      [1, 1, 1],
    ] as const)
      expect(tileTimeline(6, a, b, c, 3).total).toBe(
        tileTimeline(6, a, b, c, 2).total,
      );
    const best = (p: typeof A100) =>
      splitKSweep(p).reduce((x, y) => (y.total < x.total ? y : x));
    expect(best(A100).splits).toBe(6);
    expect(best(A100).blocks).toBe(96);
    expect(best(H100).splits).toBe(8);
    expect(best(H100).waves).toBe(1);
    expect(pct(1 - splitKSweep(A100)[0]!.sm_util)).toBe("85%");
  });
  it("chapter 11: INT4 with groups of 128 is 3.88 times smaller than BF16", () => {
    const b = quantSweep(A100, "bf16")[0]!;
    const q = quantSweep(A100, "int4")[0]!;
    expect(trim(b.weight_bytes / q.weight_bytes, 3)).toBe("3.88");
    expect(QUANT_SHAPE.rows * QUANT_SHAPE.cols).toBe(67108864);
    expect(A100.peak_int8_tensor / A100.peak_tensor).toBe(2);
    // the bends: BF16 between batch 128 and 256, INT4 between 32 and 64
    const bend = (f: "bf16" | "int4") =>
      quantSweep(A100, f).find((r) => r.bound === "compute")!.batch;
    expect(bend("bf16")).toBe(256);
    expect(bend("int4")).toBe(64);
    // at batch 1024 the weight-only formats equal BF16; W8A8 is twice as fast
    const at = (f: "bf16" | "int4" | "int8" | "w8a8") =>
      quantSweep(A100, f).at(-1)!.total;
    expect(at("int4")).toBe(at("bf16"));
    expect(at("int8")).toBe(at("bf16"));
    expect(at("bf16") / at("w8a8")).toBeCloseTo(2, 2);
  });
});
