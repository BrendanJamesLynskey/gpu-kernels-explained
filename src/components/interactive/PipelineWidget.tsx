"use client";

/**
 * Chapter 12's hero: a warp-specialised GEMM main loop on Hopper and
 * Blackwell. A producer warp issues TMA loads of K-slices of A and B into a
 * ring of shared-memory stages; one thread issues the tensor-core MMAs,
 * which read a full stage and accumulate into one of the accumulator
 * buffers (registers with one consumer, Hopper style, or a double-buffered
 * Tensor Memory, Blackwell style); an epilogue drains each finished tile.
 * Every frame is `frame(schedule, t)` of the tested model
 * (src/lib/gpu/blackwell.ts, a port of reference/blackwell.py).
 */
import { useId, useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { pct } from "@/lib/format";
import {
  PIPE_DEFAULT,
  frame,
  pipeCaption,
  pipelineDefault,
  type AccState,
  type SlotState,
} from "@/lib/gpu/blackwell";
import { LEVEL_COLOUR, MUTED, STATE_COLOUR } from "@/lib/viz/palette";

import { Hatch } from "./Hatch";

const W = 360;
const SLOT_FILL: Record<SlotState, string> = {
  empty: "none",
  loading: LEVEL_COLOUR.hbm,
  full: LEVEL_COLOUR.smem,
  mma: STATE_COLOUR.active,
};
const ACC_FILL: Record<AccState, string> = {
  free: "none",
  acc: LEVEL_COLOUR.reg,
  epi: LEVEL_COLOUR.l2,
};

export default function PipelineWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [stages, setStages] = useState<number>(PIPE_DEFAULT.stages);
  const [accs, setAccs] = useState<number>(PIPE_DEFAULT.accs);
  const [load, setLoad] = useState<number>(PIPE_DEFAULT.load);
  const [epi, setEpi] = useState<number>(PIPE_DEFAULT.epi);
  const [hover, setHover] = useState<string | null>(null);
  const hatchId = useId().replace(/:/g, "");
  const font = useSvgFont(W);
  const fs = font.fs;

  const s = useMemo(
    () => pipelineDefault({ stages, accs, load, epi }),
    [stages, accs, load, epi],
  );
  const st = useStepper(s.total + 1, {
    stepMs: 260,
    resetKey: `${stages}-${accs}-${load}-${epi}`,
  });
  const f = frame(s, st.step);
  const K = s.params.ksteps;
  const name = (g: number) => `${Math.floor(g / K) + 1}.${(g % K) + 1}`;

  // the machine, top
  const boxY = 8;
  const boxH = font.narrow ? 52 : 40;
  const slotW = 24;
  const slotGap = 3;
  const ringX = 68;
  const ringW = stages * slotW + (stages - 1) * slotGap;
  const tcX = Math.max(ringX + ringW + 14, 220);
  const accX = tcX + 52;
  const accW = 18;
  // the timeline, below
  const gy = boxY + boxH + 40;
  const lane = font.narrow ? 22 : 13;
  const left = font.narrow ? 58 : 46;
  const span = s.total;
  const sx = (t: number) => left + ((W - left - 6) * t) / span;
  const lanes = stages + 2;
  const H = gy + lanes * lane + 26;

  const label = (
    x: number,
    y: number,
    txt: string,
    size = 8,
    anchor = "middle",
  ) => (
    <text
      x={x}
      y={y}
      textAnchor={anchor as "middle" | "start" | "end"}
      style={{ fontSize: fs(size) }}
      className="fill-neutral-700 dark:fill-neutral-300"
    >
      {txt}
    </text>
  );

  const visual = (
    <div className="mx-auto max-w-2xl">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`A warp-specialised GEMM at time ${f.t} of ${s.total}: ${stages} shared-memory stages, ${accs} accumulator buffer${accs === 1 ? "" : "s"}`}
        data-pipe-t={f.t}
        data-tensor={f.tensor}
      >
        <defs>
          <Hatch id={hatchId} />
        </defs>
        {/* producer warp */}
        <rect
          x={2}
          y={boxY}
          width={58}
          height={boxH}
          rx={4}
          fill={f.producer >= 0 ? LEVEL_COLOUR.hbm : "none"}
          fillOpacity={0.35}
          stroke="currentColor"
          strokeOpacity={0.5}
        />
        {label(31, boxY + 14, font.narrow ? "TMA" : "producer")}
        {label(31, boxY + 26, font.narrow ? "warp" : "warp: TMA")}
        {f.producer >= 0 &&
          label(
            31,
            boxY + boxH - 6,
            font.narrow ? name(f.producer) : `load ${name(f.producer)}`,
            8,
          )}
        {/* the shared-memory ring */}
        {label(
          ringX + ringW / 2,
          boxY - 1 + boxH + 14,
          `shared memory: ${stages} stage${stages === 1 ? "" : "s"}`,
        )}
        {f.slots.map(([state, g], i) => (
          <g key={`slot${i}`} data-slot={i} data-state={state}>
            <rect
              x={ringX + i * (slotW + slotGap)}
              y={boxY}
              width={slotW}
              height={boxH}
              rx={3}
              fill={SLOT_FILL[state]}
              fillOpacity={state === "empty" ? 0 : 0.55}
              stroke="currentColor"
              strokeOpacity={0.5}
              strokeDasharray={state === "empty" ? "3 2" : undefined}
            />
            {g >= 0 &&
              label(
                ringX + i * (slotW + slotGap) + slotW / 2,
                boxY + boxH / 2 + 3,
                name(g),
              )}
          </g>
        ))}
        {/* tensor core: one thread issues */}
        <rect
          x={tcX}
          y={boxY}
          width={46}
          height={boxH}
          rx={4}
          fill={f.tensor >= 0 ? STATE_COLOUR.active : `url(#${hatchId})`}
          fillOpacity={f.tensor >= 0 ? 0.35 : 1}
          stroke="currentColor"
          strokeOpacity={0.5}
        />
        {label(tcX + 23, boxY + 14, font.narrow ? "MMA" : "tensor")}
        {!font.narrow && label(tcX + 23, boxY + 26, "core")}
        {label(
          tcX + 23,
          boxY + boxH - 6,
          f.tensor >= 0
            ? font.narrow
              ? name(f.tensor)
              : `mma ${name(f.tensor)}`
            : "idle",
        )}
        {/* accumulators */}
        {f.accs.map(([state, t], b) => (
          <g key={`acc${b}`} data-acc={b} data-state={state}>
            <rect
              x={accX + b * (accW + 4)}
              y={boxY}
              width={accW}
              height={boxH}
              rx={3}
              fill={ACC_FILL[state]}
              fillOpacity={state === "free" ? 0 : 0.6}
              stroke="currentColor"
              strokeOpacity={0.5}
              strokeDasharray={state === "free" ? "3 2" : undefined}
            />
            {t >= 0 &&
              label(
                accX + b * (accW + 4) + accW / 2,
                boxY + boxH / 2 + 3,
                `T${t + 1}`,
              )}
          </g>
        ))}
        {label(
          accX + (accs * (accW + 4)) / 2,
          boxY + boxH + 13,
          accs === 2 ? "TMEM ×2" : "registers",
        )}
        {/* the timeline */}
        {Array.from({ length: stages }, (_, i) => (
          <g key={`lane${i}`}>
            {label(
              2,
              gy + i * lane + lane / 2 + 3,
              `stage ${i + 1}`,
              8,
              "start",
            )}
          </g>
        ))}
        {label(2, gy + stages * lane + lane / 2 + 3, "tensor", 8, "start")}
        {label(
          2,
          gy + (stages + 1) * lane + lane / 2 + 3,
          "epilogue",
          8,
          "start",
        )}
        {s.loadStart.map((a, g) => {
          const y = gy + (g % stages) * lane + 2;
          const h = lane - 4;
          return (
            <g key={`g${g}`}>
              <rect
                x={sx(a)}
                y={y}
                width={Math.max(0.5, sx(s.loadDone[g]!) - sx(a))}
                height={h}
                fill={LEVEL_COLOUR.hbm}
                opacity={0.7}
              />
              <rect
                x={sx(s.loadDone[g]!)}
                y={y}
                width={Math.max(0, sx(s.mmaStart[g]!) - sx(s.loadDone[g]!))}
                height={h}
                fill={LEVEL_COLOUR.smem}
                opacity={0.6}
              />
              <rect
                x={sx(s.mmaStart[g]!)}
                y={y}
                width={Math.max(0.5, sx(s.mmaDone[g]!) - sx(s.mmaStart[g]!))}
                height={h}
                fill={STATE_COLOUR.active}
                opacity={0.7}
              />
              <rect
                x={sx(s.mmaStart[g]!) + 0.2}
                y={gy + stages * lane + 2}
                width={Math.max(
                  0.5,
                  sx(s.mmaDone[g]!) - sx(s.mmaStart[g]!) - 0.4,
                )}
                height={h}
                fill={STATE_COLOUR.active}
                opacity={0.85}
              />
            </g>
          );
        })}
        {/* tensor core idle: hatched */}
        {s.mmaStart.map((a, g) => {
          const prev = g === 0 ? 0 : s.mmaDone[g - 1]!;
          if (a <= prev) return null;
          return (
            <rect
              key={`idle${g}`}
              data-idle
              x={sx(prev)}
              y={gy + stages * lane + 2}
              width={sx(a) - sx(prev)}
              height={lane - 4}
              fill={`url(#${hatchId})`}
            />
          );
        })}
        {s.epiStart.map((a, t) => (
          <rect
            key={`e${t}`}
            x={sx(a)}
            y={gy + (stages + 1) * lane + 2}
            width={Math.max(0.5, sx(s.epiDone[t]!) - sx(a))}
            height={lane - 4}
            fill={LEVEL_COLOUR.l2}
            opacity={0.8}
          />
        ))}
        <line
          data-testid="cursor"
          x1={sx(f.t)}
          x2={sx(f.t)}
          y1={gy - 2}
          y2={gy + lanes * lane + 2}
          stroke="currentColor"
          strokeWidth={1.2}
        />
        <rect
          x={left}
          y={gy}
          width={W - left - 6}
          height={lanes * lane}
          fill="none"
          stroke={MUTED.light}
          strokeWidth={0.4}
        />
        {label(
          left,
          H - 6,
          `time 0 to ${s.total} (units: an MMA slice takes ${s.params.mma})`,
          8,
          "start",
        )}
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Purple: a TMA load in flight; green: the stage is full, waiting; blue:
        the tensor core reading it; sky: the epilogue. Hatched: the tensor core
        idle. Times are illustrative units, not a measured GPU.
      </p>
    </div>
  );

  const hl =
    hover ??
    (f.tensor < 0 && f.t > 0
      ? "hbm"
      : f.slots.some((x) => x[0] === "loading")
        ? "smem"
        : "compute");

  return (
    <AnimationPanel
      testId="pipeline-widget"
      title="A warp-specialised GEMM, one CTA"
      summary={`${s.params.tiles} output tiles of ${K} K-slices each: TMA loads into ${stages} shared-memory stage${stages === 1 ? "" : "s"}, single-thread MMAs, ${accs === 2 ? "two accumulator buffers in Tensor Memory" : "one accumulator in registers"}. Every frame is a step of the tested pipeline model.`}
      stepper={st}
      stepLabel="time"
      caption={pipeCaption(s, f)}
      visual={visual}
      equation={children}
      hl={hl}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Total time" value={String(s.total)} />
          <Stat label="Tensor core busy" value={pct(s.util)} />
          <Stat
            label="Slice interval"
            value={`${s.steady}`}
            hint={`an MMA slice takes ${s.params.mma}`}
          />
          <Stat
            label="Tensor idle"
            value={String(s.stall)}
            hint="after the first load"
          />
        </div>
      }
      params={
        <>
          <Segmented
            label="Stages"
            value={String(stages)}
            options={[1, 2, 3, 4, 5, 6].map((v) => ({
              value: String(v),
              label: String(v),
            }))}
            onChange={(v) => setStages(Number(v))}
          />
          <Segmented
            label="Accumulators"
            value={String(accs)}
            options={[
              { value: "1", label: "1 (registers)" },
              { value: "2", label: "2 (TMEM)" },
            ]}
            onChange={(v) => setAccs(Number(v))}
          />
          <Segmented
            label="TMA latency"
            value={String(load)}
            options={[4, 12, 24].map((v) => ({
              value: String(v),
              label: String(v),
            }))}
            onChange={(v) => setLoad(Number(v))}
          />
          <Segmented
            label="Epilogue"
            value={String(epi)}
            options={[0, 8, 30].map((v) => ({
              value: String(v),
              label: String(v),
            }))}
            onChange={(v) => setEpi(Number(v))}
          />
        </>
      }
    />
  );
}
