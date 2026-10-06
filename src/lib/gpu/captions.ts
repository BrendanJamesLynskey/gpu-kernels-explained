/**
 * The live caption of every animation step: a pure function of the model's
 * state, so a test can rebuild the caption of any step from the Python
 * reference's state and compare it with the page (tests/e2e/frames.spec.ts).
 */
import {
  fmtAi,
  fmtBytes,
  fmtFlops,
  fmtKiB,
  fmtTime,
  pct,
  trim,
} from "@/lib/format";
import { LEVEL_NAME } from "@/lib/viz/palette";

import type {
  AttentionTraffic,
  BankResult,
  Dequant,
  FlashStep,
  GemmVariantId,
  MarchStep,
  MarchSpec,
  OnlineSoftmax,
  QuantGemm,
  ReduceResult,
  SplitK,
  TimelineStep,
  CoalesceResult,
  FlowStep,
  KernelTime,
  Occupancy,
  OccupancyStep,
  Preset,
  SimtResult,
  SweepStep,
} from "./model";
import { LEVELS, popcount } from "./model";

// --- chapter 1: the memory hierarchy -----------------------------------------

export function flowCaption(
  kt: KernelTime,
  s: FlowStep,
  steps: number,
): string {
  const bound = kt.bound === "compute" ? "the ALUs" : LEVEL_NAME[kt.bound];
  const busy = LEVELS.filter((lv) => lv !== kt.bound)
    .map((lv) => `${LEVEL_NAME[lv]} ${pct(kt.util[lv])}`)
    .join(", ");
  const alu =
    kt.bound === "compute" ? "" : `; the ALUs ${pct(kt.compute_util)}`;
  const head =
    s.i === 0
      ? "Start"
      : s.i === steps
        ? `Done at ${fmtTime(s.t)}`
        : `t = ${fmtTime(s.t)} of ${fmtTime(kt.total)}`;
  const lead = kt.bound === "compute" ? "hbm" : kt.bound;
  return (
    `${head}: ${LEVEL_NAME[lead]} has moved ${fmtBytes(s.moved[lead])} of ${fmtBytes(kt.bytes[lead])}. ` +
    `The bottleneck is ${bound}, busy 100% of the time; ${busy}${alu}.`
  );
}

// --- chapter 2: the roofline ----------------------------------------------------

export function sweepCaption(s: SweepStep, ridge: number): string {
  const per =
    s.per_thread > 1
      ? `${s.per_thread}×${s.per_thread} per thread`
      : "1 per thread";
  const where =
    s.bound === "memory"
      ? `below the ridge at ${trim(ridge)}: memory-bound at ${fmtFlops(s.perf)}`
      : `past the ridge at ${trim(ridge)}: compute-bound at ${fmtFlops(s.perf)}`;
  return `${s.tile}×${s.tile} tiles (${per}): ${fmtBytes(s.hbm)} from HBM, ${fmtAi(s.ai)}, ${where}.`;
}

// --- chapter 3: SIMT ------------------------------------------------------------

/** "0–7, 16–23": the set bits of a mask as runs. */
export function laneRuns(mask: number): string {
  const runs: string[] = [];
  let lane = 0;
  while (lane < 32) {
    if ((mask >>> lane) & 1) {
      const start = lane;
      while (lane < 32 && (mask >>> lane) & 1) lane++;
      runs.push(lane - 1 === start ? `${start}` : `${start}–${lane - 1}`);
    } else lane++;
  }
  return runs.length ? runs.join(", ") : "none";
}

export function simtCaption(
  r: SimtResult,
  i: number,
  lens: [number, number, number],
): string {
  const s = r.steps[i];
  if (!s) return "";
  const n = popcount(s.mask);
  const what =
    s.phase === "pre"
      ? "the warp computes its index; every lane active"
      : s.phase === "branch"
        ? r.divergent
          ? `the branch: ${popcount(r.taken)} lanes take it, ${32 - popcount(r.taken)} do not, so the warp diverges`
          : `the branch: every lane goes the same way, so no divergence`
        : s.phase === "A"
          ? `the if-path, instruction ${s.j + 1} of ${lens[0]}`
          : s.phase === "B"
            ? `the else-path, instruction ${s.j + 1} of ${lens[1]}`
            : `after reconvergence, instruction ${s.j + 1} of ${lens[2]}`;
  const lanes =
    n === 32
      ? "all 32 lanes active"
      : `${n} of 32 lanes active (${laneRuns(s.mask)}); the rest are masked off`;
  return `Issue ${i + 1} of ${r.issued}: ${what}; ${lanes}.`;
}

// --- chapter 4: coalescing -------------------------------------------------------

export function coalesceCaption(
  r: CoalesceResult,
  lane: number,
  eb: number,
): string {
  const s = r.steps[lane];
  if (!s) return "";
  const sec =
    s.sectors.length === 1
      ? `sector ${s.sectors[0]}`
      : `sectors ${s.sectors[0]}–${s.sectors[s.sectors.length - 1]}`;
  const fresh =
    s.new.length === 0
      ? "already fetched for an earlier lane"
      : s.new.length === s.sectors.length
        ? "a new transaction"
        : "partly new";
  const tail =
    lane === 31
      ? ` The warp's request needs ${r.sectors} sector${r.sectors === 1 ? "" : "s"} (${fmtBytes(r.fetched)}) for ${fmtBytes(r.requested)} used: ${pct(Math.min(1, r.efficiency))} efficient.`
      : ` So far: ${s.total} sector${s.total === 1 ? "" : "s"}.`;
  return `Lane ${lane} reads ${eb} B at byte ${s.addr}: ${sec}, ${fresh}.${tail}`;
}

// --- chapter 5: bank conflicts ------------------------------------------------------

/** Steps: 32 requests, then one per service pass. */
export function bankStepCount(r: BankResult): number {
  return 32 + r.degree;
}

export function bankCaption(r: BankResult, i: number): string {
  if (i < 32) {
    const q = r.requests[i];
    if (!q) return "";
    const queued = q.load[q.bank] as number;
    return `Lane ${q.lane} asks for word ${q.word}: bank ${q.bank} (${q.word} mod 32), which now holds ${queued} different word${queued === 1 ? "" : "s"} for this warp.`;
  }
  const p = r.passes[i - 32];
  if (!p) return "";
  const done = r.passes
    .slice(0, i - 31)
    .reduce((a, x) => a + x.lanes.length, 0);
  const conflict =
    r.degree === 1
      ? "No conflict: one pass serves the whole warp."
      : `A ${r.degree}-way conflict: ${r.degree} passes, ${r.degree}× the time of a conflict-free access.`;
  return `Pass ${p.pass + 1} of ${r.degree}: each bank serves one word; ${p.lanes.length} lane${p.lanes.length === 1 ? "" : "s"} served (${done} of 32 so far). ${conflict}`;
}

// --- chapter 6: occupancy ------------------------------------------------------------

export const LIMIT_TEXT = {
  warps: "warp slots",
  regs: "registers",
  smem: "shared memory",
  blocks: "the 32-block limit",
} as const;

export function occupancyCaption(
  p: Preset,
  o: Occupancy,
  s: OccupancyStep,
): string {
  const use = `${s.warps} of ${p.max_warps_per_sm} warps, ${s.regs.toLocaleString("en-GB")} of ${p.regs_per_sm.toLocaleString("en-GB")} registers, ${fmtKiB(s.smem)} of ${fmtKiB(p.smem_per_sm)} shared memory`;
  if (!s.rejected)
    return s.blocks === 0
      ? `An empty SM: ${use}.`
      : `Block ${s.blocks} placed: ${use} in use.`;
  const why = o.limiters.map((l) => LIMIT_TEXT[l]).join(" and ");
  if (o.blocks === 0)
    return `No block fits on the SM: it is limited by ${why}. Occupancy 0%.`;
  return `Block ${s.blocks + 1} does not fit (limited by ${why}). ${o.blocks} block${o.blocks === 1 ? "" : "s"}, ${o.active_warps} warps resident: occupancy ${pct(o.occupancy, 1)}.`;
}

// --- chapter 7: GEMM, step by step -----------------------------------------------

const MARCH_HOW: Record<GemmVariantId, (m: MarchSpec, bk: number) => string> = {
  naive: (m, bk) =>
    `each of its ${m.bm * m.bn} threads reads ${bk} values of A and ${bk} of B straight from global memory`,
  smem: (m, bk) =>
    `the block stages a ${m.bm} × ${bk} tile of A and a ${bk} × ${m.bn} tile of B in shared memory, then each thread reads its row and column from there`,
  regs: (m, bk) =>
    `the block stages ${m.bm} × ${bk} and ${bk} × ${m.bn} tiles in shared memory; each thread loads ${m.tm} + ${m.tn} values into registers per k and does ${m.tm * m.tn} multiply-adds with them`,
  tensor: (m, bk) =>
    `the block stages ${m.bm} × ${bk} and ${bk} × ${m.bn} BF16 tiles in shared memory and one warp-wide matrix instruction multiplies them`,
};

export function marchCaption(
  vid: GemmVariantId,
  spec: MarchSpec,
  s: MarchStep,
  size: number,
  bk: number,
): string {
  if (!s.block || s.kt === null)
    return `Start: C = A·B with 16 × 16 matrices, ${(size / spec.bm) * (size / spec.bn)} blocks of ${spec.bm} × ${spec.bn} outputs, k in tiles of ${bk}.`;
  const kts = size / bk;
  const head = `Block (${s.block[0]}, ${s.block[1]}), k-tile ${s.kt + 1} of ${kts}: ${MARCH_HOW[vid](spec, bk)}.`;
  const store =
    s.kt === kts - 1
      ? ` The block's C tile is finished and written (${fmtBytes(4 * spec.bm * spec.bn)}).`
      : "";
  const smem = spec.smem ? `, ${fmtBytes(s.smem)} through shared memory` : "";
  return `${head}${store} So far ${fmtBytes(s.global)} from global memory${smem}, ${s.flops.toLocaleString("en-GB")} flops: ${fmtAi(s.ai as number)}.`;
}

// --- chapter 8: reductions --------------------------------------------------------------

export function reduceCaption(r: ReduceResult, i: number): string {
  const s = r.steps[i];
  if (!s) return "";
  if (i === 0)
    return `${r.n} numbers in shared memory, one per thread (${r.n / 32} warps); their sum is ${r.expected.toLocaleString("en-GB")}.`;
  const last = i === r.steps.length - 1;
  const total = last
    ? ` Done: x[0] = ${(s.values[0] as number).toLocaleString("en-GB")}, the sum.`
    : "";
  if (r.kind === "shuffle") {
    if (s.stride === 0)
      return `Lane 0 of each warp writes its warp's sum to shared memory; after one __syncthreads, thread 0 adds the ${r.n / 32} partial sums.${total}`;
    return `__shfl_down_sync by ${s.stride}: every lane adds the value of the lane ${s.stride} above it, register to register, in both warps; lanes 0–${s.stride - 1} now hold useful partial sums.${total}`;
  }
  const what =
    r.kind === "divergent"
      ? `threads whose index is a multiple of ${2 * s.stride} add x[tid + ${s.stride}]`
      : r.kind === "strided"
        ? `thread t adds x[${2 * s.stride}t + ${s.stride}] into x[${2 * s.stride}t]`
        : `threads 0–${s.stride - 1} add x[tid + ${s.stride}]`;
  const div =
    s.warps_divergent > 0
      ? `${s.warps_divergent} of ${s.warps_active} active warp${s.warps_active === 1 ? "" : "s"} diverge${s.warps_divergent === 1 ? "s" : ""}`
      : `${s.warps_active} warp${s.warps_active === 1 ? "" : "s"} fully active, no divergence`;
  const bank =
    s.degree > 1 ? `a ${s.degree}-way bank conflict` : "no bank conflict";
  return `Stride ${s.stride}: ${what} (${s.active.length} threads); ${div}; ${bank}. ${s.syncs} __syncthreads so far.${total}`;
}

// --- chapter 9: online softmax and FlashAttention ------------------------------------------

/** Typographic minus signs for negative numbers in running text. */
const minus = (text: string): string =>
  text.replace(/(^|[\s(=])-(?=\d)/g, "$1−");

const f3 = (v: number): string => minus(trim(v, 3));

export function softmaxCaption(o: OnlineSoftmax, i: number): string {
  const blocks = o.steps.length;
  if (i === 0)
    return `${o.x.length} scores, read in ${blocks} block${blocks === 1 ? "" : "s"} of ${o.steps[0]!.hi}. Start with m = −∞ and ℓ = 0.`;
  if (i === blocks + 1)
    return `Normalise: p_i = e^(x_i − ${f3(o.m)}) / ${f3(o.l)}. Ordinary softmax gives the same probabilities; the largest difference is ${o.max_diff === 0 ? "0" : o.max_diff.toExponential(1)}, rounding.`;
  const s = o.steps[i - 1];
  if (!s) return "";
  const range = `x${s.lo}–x${s.hi - 1}`;
  if (s.m_prev === null)
    return `Block 1 of ${blocks} (${range}): the running max becomes ${f3(s.m)} and ℓ = ${f3(s.block_sum)}, the sum of e^(x − ${f3(s.m)}) over the block.`;
  if (s.m > s.m_prev)
    return `Block ${i} of ${blocks} (${range}): its max ${f3(s.block_max)} beats the running max ${f3(s.m_prev)}, so the old sum is rescaled by e^(${f3(s.m_prev)} − ${f3(s.m)}) = ${f3(s.scale)}: ℓ = ${f3(s.l_prev)} × ${f3(s.scale)} + ${f3(s.block_sum)} = ${f3(s.l)}.`;
  return `Block ${i} of ${blocks} (${range}): its max ${f3(s.block_max)} does not beat ${f3(s.m)}, so nothing is rescaled: ℓ = ${f3(s.l_prev)} + ${f3(s.block_sum)} = ${f3(s.l)}.`;
}

export function flashCaption(a: AttentionTraffic, s: FlashStep): string {
  if (s.i === null || s.j === null)
    return `N = ${a.n}, d = ${a.d}: ${a.tr} query blocks × ${a.tc} key blocks of ${a.br}. Standard attention would write the ${a.n} × ${a.n} score matrix to HBM and read it back; FlashAttention never does.`;
  const load =
    s.j === 0
      ? `Q block ${s.i + 1} is loaded (${fmtBytes(2 * a.br * a.d)}), then `
      : "";
  const store =
    s.j === a.tc - 1
      ? ` Row done: O block ${s.i + 1} is written once (${fmtBytes(2 * a.br * a.d)}).`
      : "";
  return `Query block ${s.i + 1} of ${a.tr}, key block ${s.j + 1} of ${a.tc}: ${load}K and V block ${s.j + 1} come in (${fmtBytes(4 * a.bc * a.d)}); the ${a.br} × ${a.bc} score tile stays on chip.${store} HBM so far: ${fmtBytes(s.flash)} if every K, V re-read misses L2, ${fmtBytes(s.flash_l2)} if they all hit; standard attention, ${fmtBytes(s.standard)}.`;
}

// --- chapter 10: overlap and split-K ---------------------------------------------------------

export function timelineCaption(
  steps: TimelineStep[],
  i: number,
  n: number,
): string {
  const s = steps[i];
  if (!s) return "";
  const t = trim(s.t, 3);
  if (i === steps.length - 1)
    return `t = ${t}: all ${n} tiles stored. The compute engine was busy ${pct(s.compute_busy)} of the time.`;
  const parts: string[] = [];
  if (s.load !== null) parts.push(`copy-in loads tile ${s.load + 1}`);
  if (s.compute !== null) parts.push(`compute works on tile ${s.compute + 1}`);
  if (s.store !== null) parts.push(`copy-out stores tile ${s.store + 1}`);
  const doing = parts.length ? parts.join(", ") : "every engine waits";
  const busy = s.t > 0 ? ` Compute busy so far: ${pct(s.compute_busy)}.` : "";
  return `t = ${t}: ${doing}; ${s.stored} of ${n} tiles stored.${busy}`;
}

export function splitKCaption(r: SplitK, base: SplitK, sms: number): string {
  const head =
    r.splits === 1
      ? `No split: ${r.tiles} output tiles make ${r.blocks} blocks`
      : `${r.splits} splits: ${r.tiles} tiles × ${r.splits} = ${r.blocks} blocks`;
  const waves = `${r.waves} wave${r.waves === 1 ? "" : "s"}`;
  const reduce =
    r.splits === 1
      ? ""
      : ` + ${fmtTime(r.t_reduce)} adding up ${fmtBytes(r.extra_bytes)} of partial sums`;
  const speed =
    r.splits === 1
      ? ""
      : ` (${trim(base.total / r.total, 2)}× the unsplit speed)`;
  return `${head} on ${sms} SMs: ${waves}, ${pct(r.sm_util)} of SMs busy. ${fmtTime(r.t_compute)} computing${reduce} = ${fmtTime(r.total)}${speed}.`;
}

// --- chapter 11: quantised kernels ------------------------------------------------------------

const signed = (v: number): string =>
  v < 0 ? `(−${trim(-v, 6)})` : trim(v, 6);

export function dequantCaption(d: Dequant, i: number): string {
  return minus(dequantText(d, i));
}

function dequantText(d: Dequant, i: number): string {
  const s = d.steps[i];
  if (!s) return "";
  const hex = `0x${d.word.toString(16).toUpperCase().padStart(8, "0")}`;
  if (s.stage === "load")
    return `One 32-bit load brings eight 4-bit weights into a register: ${hex}. The group's scale is ${d.scale}.`;
  const n = `Weight ${s.i + 1} of 8`;
  if (s.stage === "unpack")
    return `${n}: shift right by ${4 * s.i} and mask with 0xF: q = ${s.q} (bits ${4 * s.i}–${4 * s.i + 3}).`;
  if (s.stage === "scale")
    return `${n}: dequantise, w = (${s.q} − 8) × ${d.scale} = ${trim(s.w as number, 6)}.`;
  const x = d.x[s.i] as number;
  const prev = d.steps[i - 1]!.acc;
  return `${n}: fused multiply-add with the activation ${trim(x, 6)}: acc = ${trim(prev, 6)} + ${signed(s.w as number)} × ${signed(x)} = ${trim(s.acc, 6)}.`;
}

export const QUANT_NAME = {
  bf16: "BF16",
  int8: "INT8 (W8A16)",
  int4: "INT4 (W4A16)",
  w8a8: "W8A8",
} as const;

export function quantCaption(rows: QuantGemm[]): string {
  const b = rows[0];
  if (!b) return "";
  const parts = rows.map(
    (r) =>
      `${QUANT_NAME[r.fmt]} ${fmtTime(r.total)}, ${r.bound}-bound${r === b ? "" : `, ${trim(b.total / r.total, 3)}× BF16's speed`}`,
  );
  return `${b.batch} token${b.batch === 1 ? "" : "s"} at once: ${parts.join("; ")}.`;
}
