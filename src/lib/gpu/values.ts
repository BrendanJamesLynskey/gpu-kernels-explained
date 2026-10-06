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
  KERNELS,
  derived,
  kernelTime,
  preset,
  rooflineSweep,
  type PresetId,
} from "./model";

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
  return { ...p, ...d, kernels, sweep };
}

const CACHE = new Map<PresetId, Record<string, unknown>>();

export function lookup(path: string): number | string {
  const [pid, ...rest] = path.split(".");
  if (pid !== "a100" && pid !== "h100")
    throw new Error(`bad preset in ${path}`);
  if (!CACHE.has(pid)) CACHE.set(pid, tree(pid));
  let cur: unknown = CACHE.get(pid);
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
