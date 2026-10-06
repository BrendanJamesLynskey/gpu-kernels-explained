"use client";

/**
 * Chapter 6's hero: thread blocks land on one SM, one at a time, each
 * taking its warp slots, registers and shared memory, until the next block
 * does not fit; the resource that runs out first is the limiter. The rules
 * (register and shared-memory allocation granules, the reserved kilobyte)
 * are those of the CUDA Toolkit's cuda_occupancy.h. Driven by
 * `occupancySteps`.
 */
import { useId, useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Slider, Stat } from "@/components/ui/Controls";
import { fmtKiB, pct } from "@/lib/format";
import { LIMIT_TEXT, occupancyCaption } from "@/lib/gpu/captions";
import {
  occupancySteps,
  preset,
  type Limiter,
  type PresetId,
} from "@/lib/gpu/model";
import {
  LEVEL_COLOUR,
  MUTED,
  OKABE_ITO,
  STATE_COLOUR,
} from "@/lib/viz/palette";

import { Hatch } from "./Hatch";

const W = 360;
const BAR_X = 96;
const BAR_W = 250;

export default function OccupancyWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [pid, setPid] = useState<PresetId>("a100");
  const [threads, setThreads] = useState(256);
  const [regs, setRegs] = useState(64);
  const [smemKb, setSmemKb] = useState(48);
  const [hover, setHover] = useState<string | null>(null);
  const hatchId = useId().replace(/:/g, "");

  const p = useMemo(() => preset(pid), [pid]);
  const maxKb = Math.floor(p.smem_per_block / 1024);
  const smem = Math.min(smemKb, maxKb) * 1024;
  const { result: o, steps } = useMemo(
    () => occupancySteps(p, threads, regs, smem),
    [p, threads, regs, smem],
  );
  const st = useStepper(steps.length, {
    stepMs: 650,
    resetKey: `${pid}-${threads}-${regs}-${smem}`,
  });
  const s = steps[st.step]!;
  const wpb = o.warps_per_block;
  const rejected = s.rejected;

  // the colour of block b's share (the newest is the active highlight)
  const blockFill = (b: number) =>
    b === s.blocks - 1 && !rejected
      ? STATE_COLOUR.active
      : b % 2
        ? OKABE_ITO.sky
        : "#9ca3af";

  const bar = (
    y: number,
    label: string,
    total: number,
    per: number,
    key: Limiter,
    colour: string,
    fmt: (v: number) => string,
  ) => {
    const used = s.blocks * per;
    const isLimit = rejected && o.limiters.includes(key);
    const hl = hover === key;
    return (
      <g data-resource={key} data-limit={isLimit ? "true" : undefined}>
        <text
          x={4}
          y={y + 11}
          className="fill-neutral-800 text-[12px] font-medium dark:fill-neutral-200"
        >
          {label}
        </text>
        <rect
          x={BAR_X}
          y={y}
          width={BAR_W}
          height={14}
          fill="none"
          stroke={hl || isLimit ? STATE_COLOUR.active : MUTED.light}
          strokeWidth={hl || isLimit ? 2 : 1}
        />
        {Array.from({ length: s.blocks }, (_, b) => (
          <rect
            key={b}
            x={BAR_X + (BAR_W * b * per) / total}
            y={y + 1}
            width={Math.max(0.5, (BAR_W * per) / total - 0.5)}
            height={12}
            fill={blockFill(b)}
          />
        ))}
        {rejected && per > 0 && (
          <rect
            x={BAR_X + (BAR_W * used) / total}
            y={y + 1}
            width={Math.min(
              (BAR_W * per) / total,
              BAR_W + 8 - (BAR_W * used) / total,
            )}
            height={12}
            fill={`url(#${hatchId})`}
            stroke={isLimit ? STATE_COLOUR.stalled : "none"}
          />
        )}
        <rect x={BAR_X - 3} y={y} width={3} height={14} fill={colour} />
        <text
          x={BAR_X}
          y={y + 25}
          className="fill-neutral-600 font-mono text-[10px] dark:fill-neutral-400"
        >
          {fmt(used)} of {fmt(total)}
          {isLimit ? " · next block does not fit" : ""}
        </text>
      </g>
    );
  };

  const visual = (
    <div className="mx-auto max-w-xl">
      <svg
        viewBox={`0 0 ${W} 196`}
        className="h-auto w-full"
        role="img"
        aria-label={`One SM of the ${p.name}: ${o.blocks} blocks of ${threads} threads fit; occupancy ${pct(o.occupancy, 1)}.`}
      >
        <defs>
          <Hatch id={hatchId} />
        </defs>
        {/* warp slots: 4 schedulers x 16 */}
        <text
          x={4}
          y={14}
          className="fill-neutral-800 text-[12px] font-medium dark:fill-neutral-200"
        >
          Warp slots
        </text>
        {Array.from({ length: p.max_warps_per_sm }, (_, w) => {
          const b = Math.floor(w / wpb);
          const filled = w < s.warps;
          const wanted = rejected && w >= s.warps && w < s.warps + wpb;
          return (
            <rect
              key={w}
              data-slot={w}
              x={BAR_X + (w % 16) * (BAR_W / 16) + 0.5}
              y={4 + Math.floor(w / 16) * 9}
              width={BAR_W / 16 - 1}
              height={8}
              fill={
                filled ? blockFill(b) : wanted ? `url(#${hatchId})` : "none"
              }
              stroke={filled ? "none" : MUTED.light}
              strokeWidth={0.5}
              strokeOpacity={0.6}
            />
          );
        })}
        <text
          x={4}
          y={28}
          className="fill-neutral-600 font-mono text-[10px] dark:fill-neutral-400"
        >
          {s.warps}/{p.max_warps_per_sm}
        </text>
        {bar(
          54,
          "Registers",
          p.regs_per_sm,
          o.regs_per_block,
          "regs",
          LEVEL_COLOUR.reg,
          (v) => v.toLocaleString("en-GB"),
        )}
        {bar(
          94,
          "Shared mem.",
          p.smem_per_sm,
          o.smem_alloc,
          "smem",
          LEVEL_COLOUR.smem,
          fmtKiB,
        )}
        {bar(
          134,
          "Block slots",
          p.max_blocks_per_sm,
          1,
          "blocks",
          MUTED.dark,
          (v) => String(v),
        )}
        <text
          x={4}
          y={188}
          className="fill-neutral-700 text-[10.5px] dark:fill-neutral-300"
        >
          {rejected
            ? `Limited by ${o.limiters.map((l) => LIMIT_TEXT[l]).join(" and ")}: ${o.blocks} blocks × ${wpb} warps = ${o.active_warps} warps (${pct(o.occupancy, 1)})`
            : `Placing block ${s.blocks}…`}
        </text>
      </svg>
    </div>
  );

  return (
    <AnimationPanel
      testId="occupancy-widget"
      title="Filling an SM with thread blocks"
      summary={`${threads} threads × ${regs} registers per thread, ${fmtKiB(smem)} of shared memory per block, on one SM of the ${p.name}.`}
      stepper={st}
      stepLabel="block"
      caption={occupancyCaption(p, o, s)}
      visual={visual}
      equation={children}
      hl={hover ?? (rejected ? o.limiters.join(" ") : "")}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Blocks per SM" value={String(o.blocks)} />
          <Stat
            label="Resident warps"
            value={`${o.active_warps} / ${p.max_warps_per_sm}`}
          />
          <Stat label="Occupancy" value={pct(o.occupancy, 1)} />
          <Stat
            label="Limited by"
            value={o.limiters.map((l) => LIMIT_TEXT[l]).join(", ")}
          />
        </div>
      }
      params={
        <>
          <Segmented
            label="GPU"
            value={pid}
            options={[
              { value: "a100", label: "A100" },
              { value: "h100", label: "H100" },
            ]}
            onChange={setPid}
          />
          <Slider
            label="Threads per block"
            value={threads}
            min={32}
            max={1024}
            step={32}
            onChange={setThreads}
          />
          <Slider
            label="Registers per thread"
            value={regs}
            min={16}
            max={255}
            onChange={setRegs}
          />
          <Slider
            label="Shared memory per block"
            value={Math.min(smemKb, maxKb)}
            min={0}
            max={maxKb}
            onChange={setSmemKb}
            format={(v) => `${v} KB`}
          />
        </>
      }
    />
  );
}
