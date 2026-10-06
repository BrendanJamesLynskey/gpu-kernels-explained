/**
 * GPU execution model: the TypeScript port of `reference/gpu_model.py`.
 *
 * Every function repeats its Python counterpart with the same operations in
 * the same order, so that `tests/unit/model.test.ts` can require the port
 * to reproduce the Python fixtures exactly (IEEE doubles, no tolerance).
 * The presets themselves are read from `src/data/presets.json`, which the
 * Python script writes, so both sides start from identical values.
 */
import data from "@/data/presets.json";

export type Status = "spec" | "rule" | "measured" | "derived" | "approximation";
export type Field = {
  v: number | string;
  st: Status;
  src: string;
  ref: string;
};
export type PresetRecord = {
  id: string;
  name: string;
  arch: string;
  fields: Record<string, Field>;
};
export type Source = { title: string; url: string };

export const SOURCES = data.sources as Record<string, Source>;
export const COMMON = data.common as Record<string, Field>;
export const PRESET_RECORDS = data.presets as PresetRecord[];
export const SWEEP_TILES = data.sweepTiles as number[];

export type PresetId = "a100" | "h100";

/** A preset's values, flattened, with the common rules. */
export type Preset = {
  id: string;
  name: string;
  sms: number;
  clock_mhz: number;
  fp32_cores_per_sm: number;
  peak_fp32: number;
  peak_tensor: number;
  hbm_bw: number;
  hbm_bytes: number;
  l2_bytes: number;
  l2_bytes_per_clk: number;
  smem_per_sm: number;
  smem_per_block: number;
  regs_per_sm: number;
  regs_per_block: number;
  max_regs_per_thread: number;
  max_warps_per_sm: number;
  max_blocks_per_sm: number;
  max_threads_per_block: number;
  lat_smem: number;
  lat_l2: number;
  lat_hbm: number;
  mma_shape: string;
  warp_size: number;
  smem_banks: number;
  bank_bytes: number;
  sector_bytes: number;
  reg_alloc_unit: number;
  smem_alloc_unit: number;
  smem_reserved: number;
  sub_partitions: number;
};

export function preset(pid: PresetId): Preset {
  const p = PRESET_RECORDS.find((r) => r.id === pid);
  if (!p) throw new Error(`unknown preset ${pid}`);
  const out: Record<string, number | string> = { id: p.id, name: p.name };
  for (const [k, f] of Object.entries(COMMON)) out[k] = f.v;
  for (const [k, f] of Object.entries(p.fields)) out[k] = f.v;
  return out as unknown as Preset;
}

// ---------------------------------------------------------------------------
// Derived figures
// ---------------------------------------------------------------------------

export const LEVELS = ["reg", "smem", "l2", "hbm"] as const;
export type Level = (typeof LEVELS)[number];
export type PerLevel<T> = Record<Level, T>;

export type Derived = {
  clock_hz: number;
  fp32_cores: number;
  bw: PerLevel<number>;
  ridge_fp32: number;
  ridge_tensor: number;
  regs_bytes_per_sm: number;
};

export function derived(p: Preset): Derived {
  const clock = p.clock_mhz * 1e6;
  const cores = p.sms * p.fp32_cores_per_sm;
  const bw = {
    reg: p.peak_fp32 * 6,
    smem: p.sms * p.smem_banks * p.bank_bytes * clock,
    l2: p.l2_bytes_per_clk * clock,
    hbm: p.hbm_bw,
  };
  return {
    clock_hz: clock,
    fp32_cores: cores,
    bw,
    ridge_fp32: p.peak_fp32 / bw.hbm,
    ridge_tensor: p.peak_tensor / bw.hbm,
    regs_bytes_per_sm: p.regs_per_sm * 4,
  };
}

// ---------------------------------------------------------------------------
// Bytes moved at each level
// ---------------------------------------------------------------------------

export type Traffic = {
  flops: number;
  bytes: PerLevel<number>;
  l2_fits?: boolean;
};

export function elementwiseTraffic(n: number): Traffic {
  return {
    flops: n,
    bytes: { reg: 12 * n, smem: 0, l2: 12 * n, hbm: 12 * n },
  };
}

const idiv = (a: number, b: number): number => Math.floor(a / b);

export function gemmTraffic(
  p: Preset,
  m: number,
  n: number,
  k: number,
  bm: number,
  bn: number,
  tm: number,
  tn: number,
): Traffic {
  if (m % bm || n % bn || bm % tm || bn % tn)
    throw new Error("tile sizes must divide the matrix and the block tile");
  const flops = 2 * m * n * k;
  const loads = 4 * (m * k * idiv(n, bn) + k * n * idiv(m, bm));
  const store = 4 * m * n;
  const l2 = loads + store;
  const compulsory = 4 * (m * k + k * n + m * n);
  const fits = 4 * (m * k + k * n) <= p.l2_bytes;
  const hbm = fits ? compulsory : l2;
  const smemReads = 4 * k * idiv(m * n, tm * tn) * (tm + tn);
  return {
    flops,
    bytes: { reg: 12 * m * n * k, smem: loads + smemReads, l2, hbm },
    l2_fits: fits,
  };
}

export type Kernel =
  | { label: string; kind: "elementwise"; n: number }
  | {
      label: string;
      kind: "gemm";
      m: number;
      n: number;
      k: number;
      bm: number;
      bn: number;
      tm: number;
      tn: number;
    };

export const KERNELS = data.kernels as Record<string, Kernel>;
export type KernelId = "vecadd" | "gemm16" | "gemm128";

export function kernelTraffic(p: Preset, kernel: Kernel): Traffic {
  if (kernel.kind === "elementwise") return elementwiseTraffic(kernel.n);
  return gemmTraffic(
    p,
    kernel.m,
    kernel.n,
    kernel.k,
    kernel.bm,
    kernel.bn,
    kernel.tm,
    kernel.tn,
  );
}

export type Bound = "compute" | Level;

export type KernelTime = {
  flops: number;
  bytes: PerLevel<number>;
  times: PerLevel<number>;
  compute: number;
  total: number;
  bound: Bound;
  util: PerLevel<number>;
  compute_util: number;
  ai: PerLevel<number | null>;
  achieved: number;
};

export function kernelTime(p: Preset, kernel: Kernel): KernelTime {
  const d = derived(p);
  const t = kernelTraffic(p, kernel);
  const times = {} as PerLevel<number>;
  for (const lv of LEVELS) times[lv] = t.bytes[lv] / d.bw[lv];
  const compute = t.flops / p.peak_fp32;
  let total = compute;
  let bound: Bound = "compute";
  for (const lv of LEVELS) {
    if (times[lv] > total) {
      total = times[lv];
      bound = lv;
    }
  }
  const util = {} as PerLevel<number>;
  for (const lv of LEVELS) util[lv] = times[lv] / total;
  const ai = {} as PerLevel<number | null>;
  for (const lv of LEVELS)
    ai[lv] = t.bytes[lv] > 0 ? t.flops / t.bytes[lv] : null;
  return {
    flops: t.flops,
    bytes: t.bytes,
    times,
    compute,
    total,
    bound,
    util,
    compute_util: compute / total,
    ai,
    achieved: t.flops / total,
  };
}

export type FlowStep = {
  i: number;
  t: number;
  moved: PerLevel<number>;
  flops: number;
};

export function flowSteps(p: Preset, kernel: Kernel, steps = 60): FlowStep[] {
  const kt = kernelTime(p, kernel);
  const out: FlowStep[] = [];
  for (let i = 0; i <= steps; i++) {
    const moved = {} as PerLevel<number>;
    for (const lv of LEVELS) moved[lv] = (kt.bytes[lv] * i) / steps;
    out.push({
      i,
      t: (kt.total * i) / steps,
      moved,
      flops: (kt.flops * i) / steps,
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Roofline
// ---------------------------------------------------------------------------

export function attainable(peak: number, bw: number, ai: number): number {
  const mem = ai * bw;
  return mem < peak ? mem : peak;
}

export type Engine = "fp32" | "tensor";
export type RooflinePoint = {
  ai: number;
  perf: number;
  ridge: number;
  bound: "memory" | "compute";
  frac_peak: number;
};

export function rooflinePoint(
  p: Preset,
  ai: number,
  engine: Engine,
): RooflinePoint {
  const peak = engine === "fp32" ? p.peak_fp32 : p.peak_tensor;
  const bw = p.hbm_bw;
  const ridge = peak / bw;
  const perf = attainable(peak, bw, ai);
  return {
    ai,
    perf,
    ridge,
    bound: ai < ridge ? "memory" : "compute",
    frac_peak: perf / peak,
  };
}

export type SweepStep = {
  tile: number;
  per_thread: number;
  flops: number;
  hbm: number;
  ai: number;
  perf: number;
  bound: "memory" | "compute";
};

export function rooflineSweep(p: Preset, size = 4096): SweepStep[] {
  const out: SweepStep[] = [];
  for (const b of SWEEP_TILES) {
    const t = b > 16 ? idiv(b, 16) : 1;
    const g = gemmTraffic(p, size, size, size, b, b, t, t);
    const ai = g.flops / g.bytes.hbm;
    const pt = rooflinePoint(p, ai, "fp32");
    out.push({
      tile: b,
      per_thread: t,
      flops: g.flops,
      hbm: g.bytes.hbm,
      ai,
      perf: pt.perf,
      bound: pt.bound,
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// SIMT
// ---------------------------------------------------------------------------

export const FULL_MASK = 0xffffffff;

export function xorshift32(x0: number): number {
  let x = x0 >>> 0;
  x = (x ^ (x << 13)) >>> 0;
  x = (x ^ (x >>> 17)) >>> 0;
  x = (x ^ (x << 5)) >>> 0;
  return x;
}

export function laneData(seed: number): number[] {
  let x = seed !== 0 ? seed : 1;
  const out: number[] = [];
  for (let i = 0; i < 32; i++) {
    x = xorshift32(x);
    out.push(x % 100);
  }
  return out;
}

export type Cond = "lt" | "mod" | "uniform" | "data";

export function branchMask(cond: Cond, k: number, seed = 2024): number {
  let m = 0;
  const d = cond === "data" ? laneData(seed) : [];
  for (let lane = 0; lane < 32; lane++) {
    let t: boolean;
    if (cond === "lt") t = lane < k;
    else if (cond === "mod") t = lane % k === 0;
    else if (cond === "uniform") t = 0 < k;
    else t = (d[lane] as number) < k;
    if (t) m = (m | (1 << lane)) >>> 0;
  }
  return m;
}

export function popcount(x0: number): number {
  let x = x0 >>> 0;
  let c = 0;
  while (x) {
    x = (x & (x - 1)) >>> 0;
    c += 1;
  }
  return c;
}

export type SimtStep = {
  phase: "pre" | "branch" | "A" | "B" | "C";
  j: number;
  mask: number;
};
export type SimtResult = {
  taken: number;
  steps: SimtStep[];
  issued: number;
  active_lane_slots: number;
  efficiency: number;
  divergent: boolean;
};

export function simtSteps(
  cond: Cond,
  k: number,
  lenA: number,
  lenB: number,
  lenC: number,
  seed = 2024,
): SimtResult {
  const taken = branchMask(cond, k, seed);
  const notTaken = (FULL_MASK ^ taken) >>> 0;
  const steps: SimtStep[] = [
    { phase: "pre", j: 0, mask: FULL_MASK },
    { phase: "branch", j: 0, mask: FULL_MASK },
  ];
  if (taken)
    for (let j = 0; j < lenA; j++) steps.push({ phase: "A", j, mask: taken });
  if (notTaken)
    for (let j = 0; j < lenB; j++)
      steps.push({ phase: "B", j, mask: notTaken });
  for (let j = 0; j < lenC; j++) steps.push({ phase: "C", j, mask: FULL_MASK });
  let active = 0;
  for (const s of steps) active += popcount(s.mask);
  const issued = steps.length;
  return {
    taken,
    steps,
    issued,
    active_lane_slots: active,
    efficiency: active / (32 * issued),
    divergent: taken !== 0 && notTaken !== 0,
  };
}

// ---------------------------------------------------------------------------
// Coalescing
// ---------------------------------------------------------------------------

export type CoalesceStep = {
  lane: number;
  addr: number;
  sectors: number[];
  new: number[];
  total: number;
};
export type CoalesceResult = {
  steps: CoalesceStep[];
  sectors: number;
  lines: number;
  requested: number;
  fetched: number;
  efficiency: number;
};

export function coalesceSteps(
  elemBytes: number,
  stride: number,
  offset: number,
  sector = 32,
): CoalesceResult {
  const seen: number[] = [];
  const steps: CoalesceStep[] = [];
  for (let lane = 0; lane < 32; lane++) {
    const addr = offset + lane * stride * elemBytes;
    const first = idiv(addr, sector);
    const last = idiv(addr + elemBytes - 1, sector);
    const laneSectors: number[] = [];
    for (let s = first; s <= last; s++) laneSectors.push(s);
    const fresh = laneSectors.filter((s) => !seen.includes(s));
    seen.push(...fresh);
    steps.push({
      lane,
      addr,
      sectors: laneSectors,
      new: fresh,
      total: seen.length,
    });
  }
  const lines: number[] = [];
  for (const s of seen) {
    const ln = idiv(s, 4);
    if (!lines.includes(ln)) lines.push(ln);
  }
  const requested = 32 * elemBytes;
  const fetched = seen.length * sector;
  return {
    steps,
    sectors: seen.length,
    lines: lines.length,
    requested,
    fetched,
    efficiency: requested / fetched,
  };
}

// ---------------------------------------------------------------------------
// Bank conflicts
// ---------------------------------------------------------------------------

export type BankPattern = "row" | "col" | "stride" | "broadcast";

export function bankWords(
  pattern: BankPattern,
  pad: number,
  stride = 1,
  cols = 32,
): number[] {
  const pitch = cols + pad;
  const out: number[] = [];
  for (let lane = 0; lane < 32; lane++) {
    if (pattern === "row") out.push(lane);
    else if (pattern === "col") out.push(lane * pitch);
    else if (pattern === "stride") out.push(lane * stride);
    else out.push(idiv(lane, 8));
  }
  return out;
}

export type BankRequest = {
  lane: number;
  word: number;
  bank: number;
  load: number[];
};
export type BankResult = {
  words: number[];
  requests: BankRequest[];
  passes: { pass: number; lanes: number[] }[];
  degree: number;
};

export function bankSteps(
  pattern: BankPattern,
  pad: number,
  stride = 1,
  banks = 32,
): BankResult {
  const words = bankWords(pattern, pad, stride);
  const distinct: number[][] = Array.from({ length: banks }, () => []);
  const requests: BankRequest[] = [];
  words.forEach((w, lane) => {
    const b = w % banks;
    const d = distinct[b] as number[];
    if (!d.includes(w)) d.push(w);
    requests.push({
      lane,
      word: w,
      bank: b,
      load: distinct.map((x) => x.length),
    });
  });
  const degree = Math.max(1, Math.max(...distinct.map((d) => d.length)));
  const passes: { pass: number; lanes: number[] }[] = [];
  for (let pnum = 0; pnum < degree; pnum++) {
    const served: number[] = [];
    words.forEach((w, lane) => {
      const d = distinct[w % banks] as number[];
      if (d.length > pnum && d[pnum] === w) served.push(lane);
    });
    passes.push({ pass: pnum, lanes: served });
  }
  return { words, requests, passes, degree };
}

// ---------------------------------------------------------------------------
// Occupancy
// ---------------------------------------------------------------------------

const roundUp = (x: number, y: number): number => y * idiv(x + y - 1, y);

export type Limiter = "warps" | "regs" | "smem" | "blocks";
export type Occupancy = {
  warps_per_block: number;
  regs_per_warp: number;
  regs_per_block: number;
  smem_alloc: number;
  limits: Record<Limiter, number>;
  blocks: number;
  limiters: Limiter[];
  active_warps: number;
  occupancy: number;
};

export function occupancy(
  p: Preset,
  threads: number,
  regs: number,
  smem: number,
): Occupancy {
  const warpsPerBlock = idiv(threads + p.warp_size - 1, p.warp_size);
  const big = 1 << 30;
  const limWarps =
    threads > p.max_threads_per_block || threads < 1
      ? 0
      : idiv(p.max_warps_per_sm, warpsPerBlock);
  const regsPerWarp = roundUp(regs * p.warp_size, p.reg_alloc_unit);
  const regsPerBlock = regsPerWarp * warpsPerBlock;
  const regsAssumed = regsPerWarp * roundUp(warpsPerBlock, p.sub_partitions);
  let limRegs: number;
  if (
    p.regs_per_block < regsAssumed ||
    p.regs_per_block < regsPerBlock ||
    regs > p.max_regs_per_thread
  )
    limRegs = 0;
  else if (regsPerWarp > 0) {
    const perSub = idiv(idiv(p.regs_per_sm, p.sub_partitions), regsPerWarp);
    limRegs = idiv(perSub * p.sub_partitions, warpsPerBlock);
  } else limRegs = big;
  const smemAlloc = roundUp(smem + p.smem_reserved, p.smem_alloc_unit);
  const limSmem =
    smemAlloc > p.smem_per_block + p.smem_reserved
      ? 0
      : idiv(p.smem_per_sm, smemAlloc);
  const limBlocks = p.max_blocks_per_sm;
  const limits = {
    warps: limWarps,
    regs: limRegs,
    smem: limSmem,
    blocks: limBlocks,
  };
  const blocks = Math.min(limWarps, limRegs, limSmem, limBlocks);
  const limiters = (["warps", "regs", "smem", "blocks"] as const).filter(
    (k) => limits[k] === blocks,
  );
  const activeWarps = blocks * warpsPerBlock;
  return {
    warps_per_block: warpsPerBlock,
    regs_per_warp: regsPerWarp,
    regs_per_block: regsPerBlock,
    smem_alloc: smemAlloc,
    limits,
    blocks,
    limiters: [...limiters],
    active_warps: activeWarps,
    occupancy: activeWarps / p.max_warps_per_sm,
  };
}

export type OccupancyStep = {
  blocks: number;
  warps: number;
  regs: number;
  smem: number;
  rejected: boolean;
};

export function occupancySteps(
  p: Preset,
  threads: number,
  regs: number,
  smem: number,
): { result: Occupancy; steps: OccupancyStep[] } {
  const o = occupancy(p, threads, regs, smem);
  const steps: OccupancyStep[] = [];
  for (let b = 0; b <= o.blocks; b++) {
    steps.push({
      blocks: b,
      warps: b * o.warps_per_block,
      regs: b * o.regs_per_block,
      smem: b * o.smem_alloc,
      rejected: b === o.blocks,
    });
  }
  return { result: o, steps };
}

// ---------------------------------------------------------------------------
// Tile timeline
// ---------------------------------------------------------------------------

export type TileSegment = {
  tile: number;
  load: [number, number];
  compute: [number, number];
  store: [number, number];
};

export function tileTimeline(
  n: number,
  tLoad: number,
  tCompute: number,
  tStore: number,
  buffers: number,
): { segments: TileSegment[]; total: number; compute_busy: number } {
  const loadEnd: number[] = [];
  const compEnd: number[] = [];
  const storeEnd: number[] = [];
  const segs: TileSegment[] = [];
  for (let i = 0; i < n; i++) {
    let ls = i > 0 ? (loadEnd[i - 1] as number) : 0.0;
    if (i >= buffers && (compEnd[i - buffers] as number) > ls)
      ls = compEnd[i - buffers] as number;
    const le = ls + tLoad;
    let cs = le;
    if (i > 0 && (compEnd[i - 1] as number) > cs) cs = compEnd[i - 1] as number;
    const ce = cs + tCompute;
    let ss = ce;
    if (i > 0 && (storeEnd[i - 1] as number) > ss)
      ss = storeEnd[i - 1] as number;
    const se = ss + tStore;
    loadEnd.push(le);
    compEnd.push(ce);
    storeEnd.push(se);
    segs.push({ tile: i, load: [ls, le], compute: [cs, ce], store: [ss, se] });
  }
  const total = n ? (storeEnd[n - 1] as number) : 0.0;
  const busy = n * tCompute;
  return {
    segments: segs,
    total,
    compute_busy: total > 0 ? busy / total : 0.0,
  };
}
