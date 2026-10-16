/**
 * Hopper and Blackwell GEMM (chapter 12): the TypeScript port of
 * `reference/blackwell.py`, line for line (read its docstrings). The
 * tensor-core presets come from `src/data/tensor.json`, which the Python
 * script writes; `tests/unit/blackwell.test.ts` requires every function to
 * reproduce `tests/fixtures/blackwell_fixtures.json` exactly. The pipeline
 * is integer arithmetic, so its schedules and frames match bit for bit.
 */
import data from "@/data/tensor.json";

import type { Source } from "./model";

export type TcField = {
  v: number | string | null;
  st: string;
  src: string;
  ref: string;
};
export type TcRecord = {
  id: string;
  name: string;
  arch: string;
  fields: Record<string, TcField>;
};
export type TcId = "a100" | "h100" | "b200";
export const TC_ORDER = data.order as TcId[];
export const TC_SOURCES = data.sources as Record<string, Source>;
export const TC_RECORDS = data.presets as TcRecord[];
export const FORMATS = ["bf16", "fp8", "fp4", "int8"] as const;
export type Format = (typeof FORMATS)[number];

export type TcPreset = {
  id: string;
  name: string;
  sms: number;
  smem_per_sm: number;
  peak_bf16: number;
  peak_fp8: number | null;
  peak_fp4: number | null;
  peak_int8: number;
  hbm_bw: number;
  hbm_bytes: number;
  tmem_per_sm: number | null;
  mma: string;
  loads: string;
};

export function tcPreset(pid: TcId): TcPreset {
  const rec = TC_RECORDS.find((r) => r.id === pid)!;
  const out: Record<string, unknown> = { id: rec.id, name: rec.name };
  for (const [k, f] of Object.entries(rec.fields)) out[k] = f.v;
  return out as TcPreset;
}

export function ridges(pid: TcId): Record<Format, number | null> {
  const p = tcPreset(pid);
  const out = {} as Record<Format, number | null>;
  for (const fmt of FORMATS) {
    const peak = p[`peak_${fmt}`];
    out[fmt] = peak === null ? null : peak / p.hbm_bw;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Block-scaled formats
// ---------------------------------------------------------------------------

export type BlockFormat = {
  name: string;
  bits: number;
  block: number;
  scale: string;
  scale_bits: number;
  src: string;
};
export const BLOCK_FORMATS = data.blockFormats as Record<string, BlockFormat>;
export const BLOCK_ORDER = data.blockOrder as string[];

export function bitsPerValue(fmt: string): number {
  const f = BLOCK_FORMATS[fmt]!;
  return f.bits + (f.block ? f.scale_bits / f.block : 0);
}

export function weightBytes(fmt: string, rows: number, cols: number): number {
  const f = BLOCK_FORMATS[fmt]!;
  let b = (rows * cols * f.bits) / 8;
  if (f.block) b += (rows * Math.floor(cols / f.block) * f.scale_bits) / 8;
  return b;
}

// ---------------------------------------------------------------------------
// Shared-memory stages
// ---------------------------------------------------------------------------

export function stageBytes(
  bm: number,
  bn: number,
  bk: number,
  elemBytes: number,
): number {
  return (bm * bk + bk * bn) * elemBytes;
}

export function maxStages(smem: number, sbytes: number, reserve = 0): number {
  return Math.floor((smem - reserve) / sbytes);
}

// ---------------------------------------------------------------------------
// The warp-specialised pipeline
// ---------------------------------------------------------------------------

export type PipeParams = {
  tiles: number;
  ksteps: number;
  stages: number;
  accs: number;
  load: number;
  issue: number;
  mma: number;
  epi: number;
};
export const PIPE_DEFAULT = data.pipeDefault as PipeParams;

export type Schedule = {
  params: PipeParams;
  loadStart: number[];
  loadDone: number[];
  mmaStart: number[];
  mmaDone: number[];
  epiStart: number[];
  epiDone: number[];
  total: number;
  busy: number;
  util: number;
  steady: number;
  stall: number;
};

export function pipeline(q: PipeParams): Schedule {
  const { tiles, ksteps, stages, accs, load, issue, mma, epi } = q;
  const n = tiles * ksteps;
  const ls: number[] = [];
  const ld: number[] = [];
  const ms: number[] = [];
  const md: number[] = [];
  const es: number[] = [];
  const ed: number[] = [];
  for (let g = 0; g < n; g++) {
    const t = Math.floor(g / ksteps);
    const k = g % ksteps;
    let a = g > 0 ? ls[g - 1]! + issue : 0;
    if (g >= stages && md[g - stages]! > a) a = md[g - stages]!;
    ls.push(a);
    ld.push(a + load);
    let b = ld[g]!;
    if (g > 0 && md[g - 1]! > b) b = md[g - 1]!;
    if (k === 0 && t >= accs && ed[t - accs]! > b) b = ed[t - accs]!;
    ms.push(b);
    md.push(b + mma);
    if (k === ksteps - 1) {
      let e = md[g]!;
      if (t > 0 && ed[t - 1]! > e) e = ed[t - 1]!;
      es.push(e);
      ed.push(e + epi);
    }
  }
  const total = ed[ed.length - 1]!;
  const busy = n * mma;
  const steady = Math.max(mma, issue, Math.ceil((load + mma) / stages));
  let stall = 0;
  for (let g = 0; g < n; g++) stall += ms[g]! - (g > 0 ? md[g - 1]! : ld[0]!);
  return {
    params: { tiles, ksteps, stages, accs, load, issue, mma, epi },
    loadStart: ls,
    loadDone: ld,
    mmaStart: ms,
    mmaDone: md,
    epiStart: es,
    epiDone: ed,
    total,
    busy,
    util: busy / total,
    steady,
    stall,
  };
}

export function pipelineDefault(over: Partial<PipeParams> = {}): Schedule {
  return pipeline({ ...PIPE_DEFAULT, ...over });
}

export type SlotState = "empty" | "loading" | "full" | "mma";
export type AccState = "free" | "acc" | "epi";
export type PipeFrame = {
  t: number;
  slots: [SlotState, number][];
  producer: number;
  tensor: number;
  accs: [AccState, number][];
  done: number;
  tilesDone: number;
};

export function frame(s: Schedule, time: number): PipeFrame {
  const p = s.params;
  const st = p.stages;
  const K = p.ksteps;
  const A = p.accs;
  const n = p.tiles * K;
  const slots: [SlotState, number][] = Array.from(
    { length: st },
    (): [SlotState, number] => ["empty", -1],
  );
  let producer = -1;
  let tensor = -1;
  let done = 0;
  for (let g = 0; g < n; g++) {
    if (s.loadStart[g] === time) producer = g;
    if (s.mmaDone[g]! <= time) done += 1;
    const i = g % st;
    if (s.loadStart[g]! <= time && time < s.mmaDone[g]!) {
      if (time < s.loadDone[g]!) slots[i] = ["loading", g];
      else if (time < s.mmaStart[g]!) slots[i] = ["full", g];
      else {
        slots[i] = ["mma", g];
        tensor = g;
      }
    }
  }
  const accs: [AccState, number][] = Array.from(
    { length: A },
    (): [AccState, number] => ["free", -1],
  );
  let tilesDone = 0;
  for (let t = 0; t < p.tiles; t++) {
    const b = t % A;
    const first = s.mmaStart[t * K]!;
    if (s.epiDone[t]! <= time) tilesDone += 1;
    if (first <= time && time < s.epiStart[t]!) accs[b] = ["acc", t];
    else if (s.epiStart[t]! <= time && time < s.epiDone[t]!)
      accs[b] = ["epi", t];
  }
  return { t: time, slots, producer, tensor, accs, done, tilesDone };
}

export function frames(s: Schedule): PipeFrame[] {
  const out: PipeFrame[] = [];
  for (let t = 0; t <= s.total; t++) out.push(frame(s, t));
  return out;
}

/** The live caption for one frame. */
export function pipeCaption(s: Schedule, f: PipeFrame): string {
  const K = s.params.ksteps;
  const name = (g: number) =>
    `tile ${Math.floor(g / K) + 1}, slice ${(g % K) + 1}`;
  const parts: string[] = [`t = ${f.t}`];
  if (f.producer >= 0)
    parts.push(`the producer issues a TMA load (${name(f.producer)})`);
  parts.push(
    f.tensor >= 0
      ? `the tensor core multiplies ${name(f.tensor)}`
      : "the tensor core is idle",
  );
  const epi = f.accs.filter((a) => a[0] === "epi").map((a) => a[1] + 1);
  if (epi.length) parts.push(`the epilogue drains tile ${epi.join(", ")}`);
  parts.push(
    `${f.done} of ${s.params.tiles * K} slices done, ${f.tilesDone} of ${s.params.tiles} tiles stored`,
  );
  return `${parts.join("; ")}.`;
}
