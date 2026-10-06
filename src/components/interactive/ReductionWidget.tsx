"use client";

/**
 * Chapter 8's hero: 64 numbers summed by one block of 64 threads (two
 * warps), four ways. Each row is a warp's 32 values; arrows show which
 * value is added into which at this step; lanes that sit out are hatched
 * when their warp is still running (divergence) and faded when the whole
 * warp is idle. The shuffle version moves values register to register.
 * Every frame comes from `reduceSteps`.
 */
import { useId, useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { reduceCaption } from "@/lib/gpu/captions";
import { reduceSteps, type ReduceKind } from "@/lib/gpu/model";
import { LEVEL_COLOUR, MUTED, STATE_COLOUR } from "@/lib/viz/palette";

import { Hatch } from "./Hatch";

const CELL = 9;
const LEFT = 30;
const W = LEFT + 32 * CELL + 6;
const ROW_Y = [40, 104];
const ROW_H = 18;
const H = 150;

const LABEL: Record<ReduceKind, string> = {
  divergent: "tid % 2s",
  strided: "index 2s·tid",
  sequential: "sequential",
  shuffle: "warp shuffle",
};

export default function ReductionWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [kind, setKind] = useState<ReduceKind>("sequential");
  const [hover, setHover] = useState<string | null>(null);
  const hatchId = useId().replace(/:/g, "");
  const font = useSvgFont(W);
  const fs = font.fs;

  const r = useMemo(() => reduceSteps(kind), [kind]);
  const st = useStepper(r.steps.length, { stepMs: 1300, resetKey: kind });
  const s = r.steps[st.step]!;
  const prev = r.steps[Math.max(0, st.step - 1)]!;
  const active = new Set(s.active);

  // which source each active destination read at this step
  const arrows: [number, number][] = [];
  if (st.step > 0) {
    if (kind === "shuffle" && s.stride > 0) {
      for (let w = 0; w < 2; w++)
        for (let l = 0; l < s.stride; l++)
          arrows.push([32 * w + l + s.stride, 32 * w + l]);
    } else if (kind === "shuffle") arrows.push([32, 0]);
    else
      for (const tid of s.active) {
        const dst = kind === "strided" ? 2 * s.stride * tid : tid;
        arrows.push([dst + s.stride, dst]);
      }
  }
  const dsts = new Set(arrows.map((a) => a[1]));
  const srcs = new Set(arrows.map((a) => a[0]));
  const max = Math.max(...s.values);

  const pos = (i: number) => ({
    x: LEFT + (i % 32) * CELL + CELL / 2,
    y: ROW_Y[Math.floor(i / 32)]!,
  });

  const warpActive = (w: number) =>
    st.step > 0 && s.active.some((t) => Math.floor(t / 32) === w);

  const visual = (
    <div className="mx-auto max-w-xl">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`Summing 64 numbers, ${LABEL[kind]}: step ${st.step} of ${r.steps.length - 1}.`}
      >
        <defs>
          <Hatch id={hatchId} />
          <marker
            id={`${hatchId}-arrow`}
            viewBox="0 0 6 6"
            refX="5"
            refY="3"
            markerWidth="4"
            markerHeight="4"
            orient="auto-start-reverse"
          >
            <path d="M0,0 L6,3 L0,6 z" fill={STATE_COLOUR.active} />
          </marker>
        </defs>
        {[0, 1].map((w) => (
          <g key={w} data-warp={w} data-active={warpActive(w) || undefined}>
            <text
              x={2}
              y={ROW_Y[w]! + ROW_H / 2 + fs(8) / 3}
              style={{ fontSize: fs(8) }}
              className="fill-neutral-600 font-mono dark:fill-neutral-400"
            >
              w{w}
            </text>
            {Array.from({ length: 32 }, (_, l) => {
              const i = 32 * w + l;
              const v = s.values[i] as number;
              const isDst = dsts.has(i);
              const isSrc = srcs.has(i);
              const idleInRunningWarp =
                kind !== "shuffle" && warpActive(w) && !active.has(i);
              const fill = isDst
                ? STATE_COLOUR.active
                : isSrc
                  ? kind === "shuffle"
                    ? LEVEL_COLOUR.reg
                    : LEVEL_COLOUR.smem
                  : MUTED.light;
              const h = Math.max(1.5, (ROW_H * v) / max);
              return (
                <g key={l} data-cell={i}>
                  <rect
                    x={LEFT + l * CELL + 0.5}
                    y={ROW_Y[w]!}
                    width={CELL - 1}
                    height={ROW_H}
                    fill={idleInRunningWarp ? `url(#${hatchId})` : "none"}
                    stroke={MUTED.light}
                    strokeWidth={0.4}
                  />
                  <rect
                    x={LEFT + l * CELL + 1.5}
                    y={ROW_Y[w]! + ROW_H - h}
                    width={CELL - 3}
                    height={h}
                    fill={fill}
                    opacity={isDst || isSrc ? 1 : 0.55}
                  >
                    <title>{`x[${i}] = ${v}`}</title>
                  </rect>
                </g>
              );
            })}
          </g>
        ))}
        {arrows.map(([from, to], k) => {
          const a = pos(from);
          const b = pos(to);
          const sameRow = a.y === b.y;
          const lift = sameRow ? 6 + Math.min(18, Math.abs(a.x - b.x) / 6) : 0;
          const d = sameRow
            ? `M${a.x},${a.y - 1} Q${(a.x + b.x) / 2},${a.y - lift} ${b.x},${b.y - 1}`
            : `M${a.x},${a.y - 1} L${b.x},${b.y + ROW_H + 1}`;
          return (
            <path
              key={k}
              d={d}
              fill="none"
              stroke={STATE_COLOUR.active}
              strokeWidth={0.8}
              opacity={0.75}
              markerEnd={`url(#${hatchId}-arrow)`}
            />
          );
        })}
        <text
          x={LEFT}
          y={H - 4}
          style={{ fontSize: fs(9) }}
          className="fill-neutral-800 font-mono dark:fill-neutral-200"
        >
          x[0] = {(s.values[0] as number).toLocaleString("en-GB")}
          {st.step === r.steps.length - 1
            ? ` = the sum (${r.expected.toLocaleString("en-GB")}) ✓`
            : ` (was ${(prev.values[0] as number).toLocaleString("en-GB")})`}
        </text>
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Bar height = value. Blue: receives a sum now;{" "}
        {kind === "shuffle" ? "orange" : "green"}: read{" "}
        {kind === "shuffle"
          ? "from another lane's register"
          : "from shared memory"}
        ; hatched: an idle lane of a running warp (divergence).
      </p>
    </div>
  );

  return (
    <AnimationPanel
      testId="reduction-widget"
      title="Summing 64 numbers in one block"
      summary={`One block of 64 threads (2 warps), the ${LABEL[kind]} version. Every value and count comes from the model.`}
      stepper={st}
      stepLabel="step"
      caption={reduceCaption(r, st.step)}
      visual={visual}
      equation={children}
      hl={hover ?? (kind === "shuffle" ? "shfl" : "stride")}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Shared-memory accesses" value={String(s.smem)} />
          <Stat label="__syncthreads" value={String(s.syncs)} />
          <Stat
            label="Shuffles (per warp)"
            value={String(kind === "shuffle" ? s.shuffles / 2 : 0)}
          />
          <Stat
            label="This step"
            value={
              st.step === 0
                ? "–"
                : s.degree > 1
                  ? `${s.degree}-way conflict`
                  : s.warps_divergent > 0
                    ? `${s.warps_divergent} warp${s.warps_divergent === 1 ? " diverges" : "s diverge"}`
                    : "clean"
            }
          />
        </div>
      }
      params={
        <Segmented
          label="Version"
          value={kind}
          options={(
            ["divergent", "strided", "sequential", "shuffle"] as const
          ).map((k) => ({ value: k, label: LABEL[k] }))}
          onChange={setKind}
        />
      }
    />
  );
}
