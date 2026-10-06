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
