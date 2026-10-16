/**
 * Every number the chapters quote comes from here: a path into the model's
 * results for a preset, e.g. "a100.bw.smem" or "h100.kernels.gemm128.total",
 * formatted by kind. The MDX writes <V of="…" fmt="…" />, so prose cannot
 * drift from the tested model (tests/unit/values.test.ts checks every path
 * the chapters use resolves).
 */
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
  GEMM_ORDER,
  KERNELS,
  QUANT_ORDER,
  REDUCE_KINDS,
  attentionTraffic,
  derived,
  gemmVariant,
  kernelTime,
  preset,
  quantSweep,
  reduceSteps,
  rooflineSweep,
  splitKSweep,
  type PresetId,
} from "./model";
import {
  BLOCK_ORDER,
  TC_ORDER,
  bitsPerValue,
  maxStages,
  pipelineDefault,
  ridges,
  stageBytes,
  tcPreset,
  weightBytes,
} from "./blackwell";

export type Fmt =
  | "rate"
  | "flops"
  | "bytes"
  | "kib"
  | "time"
  | "pct"
  | "ai"
  | "num"
  | "int"
  | "raw";

function tree(pid: PresetId): Record<string, unknown> {
  const p = preset(pid);
  const d = derived(p);
  const kernels: Record<string, unknown> = {};
  for (const [k, kern] of Object.entries(KERNELS))
    kernels[k] = kernelTime(p, kern);
  const sweep: Record<string, unknown> = {};
  for (const s of rooflineSweep(p)) sweep[String(s.tile)] = s;
  const gemm: Record<string, unknown> = {};
  for (const v of GEMM_ORDER) gemm[v] = gemmVariant(p, v);
  const splitk: Record<string, unknown> = {};
  for (const r of splitKSweep(p)) splitk[String(r.splits)] = r;
  const quant: Record<string, unknown> = {};
  for (const f of QUANT_ORDER) {
    const byBatch: Record<string, unknown> = {};
    for (const r of quantSweep(p, f)) byBatch[String(r.batch)] = r;
    quant[f] = byBatch;
  }
  // attention traffic does not depend on the GPU: the same under each preset
  const attn: Record<string, unknown> = {};
  for (const n of [1024, 4096, 8192])
    for (const dd of [64, 128])
      for (const b of [64, 128])
        attn[`n${n}_d${dd}_b${b}`] = attentionTraffic(n, dd, b, b);
  // the reductions' final counts (also GPU-independent)
  const reduce: Record<string, unknown> = {};
  for (const k of REDUCE_KINDS) {
    const st = reduceSteps(k).steps;
    reduce[k] = st[st.length - 1];
  }
  return { ...p, ...d, kernels, sweep, gemm, splitk, quant, attn, reduce };
}

/** Chapter 12 and the landing page: "tc.<b200|h100|a100>.…", "tc.pipe.…". */
function tcTree(): Record<string, unknown> {
  const t: Record<string, unknown> = {};
  const sb = stageBytes(128, 256, 64, 2);
  for (const pid of TC_ORDER) {
    const p = tcPreset(pid);
    t[pid] = { ...p, ridge: ridges(pid), stages: maxStages(p.smem_per_sm, sb) };
  }
  t.stageBytes = sb;
  const bits: Record<string, unknown> = {};
  const wb: Record<string, unknown> = {};
  for (const f of BLOCK_ORDER) {
    bits[f] = bitsPerValue(f);
    wb[f] = weightBytes(f, 8192, 8192);
  }
  t.bits = bits;
  t.weightBytes = wb;
  t.pipe = {
    default: pipelineDefault(),
    acc1: pipelineDefault({ accs: 1 }),
    s1: pipelineDefault({ stages: 1 }),
    s2: pipelineDefault({ stages: 2 }),
    s3: pipelineDefault({ stages: 3 }),
    epi30: pipelineDefault({ epi: 30 }),
    epi30acc1: pipelineDefault({ epi: 30, accs: 1 }),
  };
  // B200 over H100 and A100, BF16 dense, and NVFP4 over BF16 on the B200
  const b = tcPreset("b200");
  t.ratio = {
    b200OverH100: b.peak_bf16 / tcPreset("h100").peak_bf16,
    fp4OverBf16: b.peak_fp4! / b.peak_bf16,
  };
  return t;
}

let TC: Record<string, unknown> | null = null;

const CACHE = new Map<PresetId, Record<string, unknown>>();

export function lookup(path: string): number | string {
  const [pid, ...rest] = path.split(".");
  let cur: unknown;
  if (pid === "tc") {
    if (!TC) TC = tcTree();
    cur = TC;
  } else {
    if (pid !== "a100" && pid !== "h100")
      throw new Error(`bad preset in ${path}`);
    if (!CACHE.has(pid)) CACHE.set(pid, tree(pid));
    cur = CACHE.get(pid);
  }
  for (const k of rest) {
    if (cur === null || typeof cur !== "object" || !(k in cur))
      throw new Error(`no value at ${path}`);
    cur = (cur as Record<string, unknown>)[k];
  }
  if (typeof cur !== "number" && typeof cur !== "string")
    throw new Error(`${path} is not a number or string`);
  return cur;
}

export function formatValue(v: number | string, fmt: Fmt): string {
  if (typeof v === "string") return v;
  switch (fmt) {
    case "rate":
      return fmtRate(v);
    case "flops":
      return fmtFlops(v);
    case "bytes":
      return fmtBytes(v);
    case "kib":
      return fmtKiB(v);
    case "time":
      return fmtTime(v);
    case "pct":
      return pct(v);
    case "ai":
      return fmtAi(v);
    case "int":
      return Math.round(v).toLocaleString("en-GB");
    case "raw":
      return String(v);
    default:
      return trim(v);
  }
}
