/**
 * The learning layer's numbers (brief 44) that are not already a path in
 * the chapters' value tree (`src/lib/gpu/values.ts`): each is computed here
 * by the tested model, at build time, from the same functions the widgets
 * call with the widgets' own settings. A self-check cites one as
 * `<Check v="…">`, prose as `<V v="…" />`; tests/unit/learn.test.ts checks
 * every one against the model.
 */
import {
  bankSteps,
  derived,
  coalesceSteps,
  gemmVariant,
  occupancy,
  preset,
  quantSweep,
  simtSteps,
  splitKSweep,
  tileTimeline,
  attentionTraffic,
} from "@/lib/gpu/model";
import { SCENARIOS } from "@/lib/gpu/scenarios";

const a100 = () => preset("a100");
const h100 = () => preset("h100");

/** Compute-heavy timeline (chapter 10's default scenario). */
const timeline = (buffers: number) => {
  const [n, tl, tc, ts] = SCENARIOS.compute.args;
  return tileTimeline(n, tl, tc, ts, buffers);
};

const quantAt = (fmt: "bf16" | "int8" | "int4" | "w8a8", batch: number) => {
  const r = quantSweep(a100(), fmt).find((q) => q.batch === batch);
  if (!r) throw new Error(`no quant point ${fmt} @ ${batch}`);
  return r;
};

const VALUES: Record<string, () => number> = {
  // chapter 1
  "ch01.smemOverHbm": () => {
    const d = derived(a100());
    return d.bw.smem / d.bw.hbm;
  },
  // chapter 3 (SimtWidget defaults: lenA 3, lenB 2, lenC 2)
  "ch03.lt8.eff": () => simtSteps("lt", 8, 3, 2, 2).efficiency,
  "ch03.mod2.eff": () => simtSteps("mod", 2, 3, 2, 2).efficiency,
  "ch03.lt64.eff": () => simtSteps("lt", 64, 3, 2, 2).efficiency,
  "ch03.lt8.issued": () => simtSteps("lt", 8, 3, 2, 2).issued,
  // chapter 4 (4-byte words)
  "ch04.stride8.eff": () => coalesceSteps(4, 8, 0).efficiency,
  "ch04.stride8.sectors": () => coalesceSteps(4, 8, 0).sectors,
  "ch04.offset4.sectors": () => coalesceSteps(4, 1, 4).sectors,
  "ch04.stride0.sectors": () => coalesceSteps(4, 0, 0).sectors,
  "ch04.stride2.eff": () => coalesceSteps(4, 2, 0).efficiency,
  // chapter 5
  "ch05.col.pad0": () => bankSteps("col", 0).degree,
  "ch05.col.pad2": () => bankSteps("col", 2).degree,
  "ch05.stride3": () => bankSteps("stride", 0, 3).degree,
  "ch05.stride4": () => bankSteps("stride", 0, 4).degree,
  // chapter 6 (A100; the guide's examples with 32 registers and no smem)
  "ch06.t768": () => occupancy(a100(), 768, 32, 0).occupancy,
  "ch06.t32": () => occupancy(a100(), 32, 32, 0).occupancy,
  "ch06.t256r64s48.a100": () => occupancy(a100(), 256, 64, 48 * 1024).occupancy,
  "ch06.t256r64s48.h100": () => occupancy(h100(), 256, 64, 48 * 1024).occupancy,
  // chapter 7
  "ch07.regsOverNaive": () =>
    gemmVariant(a100(), "naive").total / gemmVariant(a100(), "regs").total,
  "ch07.tensorOverRegs.h100": () =>
    gemmVariant(h100(), "regs").total / gemmVariant(h100(), "tensor").total,
  // chapter 9 (one head, 4096 x 64, blocks of 128)
  "ch09.stdOverFlash": () => {
    const t = attentionTraffic(4096, 64, 128, 128);
    return t.standard / t.flash;
  },
  // chapter 10
  "ch10.buf2.speedup": () => timeline(1).total / timeline(2).total,
  "ch10.buf3.minus.buf2": () => timeline(3).total - timeline(2).total,
  "ch10.a100.bestSplit": () => {
    const s = splitKSweep(a100());
    return s.reduce((b, r) => (r.total < b.total ? r : b)).splits;
  },
  "ch10.h100.bestSplit": () => {
    const s = splitKSweep(h100());
    return s.reduce((b, r) => (r.total < b.total ? r : b)).splits;
  },
  // chapter 11 (one 8192 x 8192 layer on the A100)
  "ch11.int4Speedup.b1": () =>
    quantAt("bf16", 1).total / quantAt("int4", 1).total,
  "ch11.int4OverBf16.b1024": () =>
    quantAt("int4", 1024).total / quantAt("bf16", 1024).total,
  "ch11.w8a8Speedup.b1024": () =>
    quantAt("bf16", 1024).total / quantAt("w8a8", 1024).total,
};

export const LEARN_VALUE_NAMES = Object.keys(VALUES);

const CACHE = new Map<string, number>();

export function learnValue(name: string): number {
  const cached = CACHE.get(name);
  if (cached !== undefined) return cached;
  const f = VALUES[name];
  if (!f) throw new Error(`no learn value ${name}`);
  const v = f();
  if (!Number.isFinite(v)) throw new Error(`learn value ${name} is ${v}`);
  CACHE.set(name, v);
  return v;
}
