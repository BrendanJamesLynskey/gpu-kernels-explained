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
  BankResult,
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
