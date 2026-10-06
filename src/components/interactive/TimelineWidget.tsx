"use client";

/**
 * Chapter 10's hero: the load / compute / store timeline of a tiled
 * kernel (or of copies and kernels on CUDA streams), one lane per engine.
 * With one buffer the next load must wait for the compute to free it;
 * with two or three, loads run ahead and hide behind compute. A cursor
 * moves from event to event; each frame is a state of `timelineSteps`
 * over the schedule `tileTimeline` computes.
 */
import { useId, useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { pct, trim } from "@/lib/format";
import { timelineCaption } from "@/lib/gpu/captions";
import { tileTimeline, timelineSteps } from "@/lib/gpu/model";
import { LEVEL_COLOUR, MUTED, STATE_COLOUR } from "@/lib/viz/palette";

import { Hatch } from "./Hatch";

/** Scenarios: tiles, then load, compute and store times per tile. */
export const SCENARIOS = {
  compute: { label: "compute-heavy", args: [6, 2.0, 3.0, 1.0] },
  copy: { label: "copy-heavy", args: [6, 3.0, 1.5, 1.0] },
  balanced: { label: "balanced", args: [6, 1.0, 1.0, 1.0] },
} as const;
export type Scenario = keyof typeof SCENARIOS;

const W = 320;
const LEFT = 58;
const RIGHT = 8;
const LANE = 26;
const TOP = 10;
const ENGINES = [
  ["load", "copy in", LEVEL_COLOUR.hbm],
  ["compute", "compute", LEVEL_COLOUR.reg],
  ["store", "copy out", LEVEL_COLOUR.l2],
] as const;

export default function TimelineWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [scenario, setScenario] = useState<Scenario>("compute");
  const [buffers, setBuffers] = useState(2);
  const [hover, setHover] = useState<string | null>(null);
  const hatchId = useId().replace(/:/g, "");
  const font = useSvgFont(W);
  const fs = font.fs;

  const [n, tl, tc, ts] = SCENARIOS[scenario].args;
  const tlv = useMemo(
    () => tileTimeline(n, tl, tc, ts, buffers),
    [n, tl, tc, ts, buffers],
  );
  const serial = useMemo(() => tileTimeline(n, tl, tc, ts, 1), [n, tl, tc, ts]);
  const steps = useMemo(() => timelineSteps(tlv), [tlv]);
  const st = useStepper(steps.length, {
    stepMs: 700,
    resetKey: `${scenario}-${buffers}`,
  });
  const s = steps[st.step]!;
  // the time axis spans the slowest schedule (one buffer), so the
  // pictures for 1, 2 and 3 buffers can be compared by eye
  const span = serial.total;
  const sx = (t: number) => LEFT + ((W - LEFT - RIGHT) * t) / span;
  const H = TOP + 3 * LANE + 30;

  const visual = (
    <div className="mx-auto max-w-xl">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`Timeline of ${n} tiles with ${buffers} buffer${buffers === 1 ? "" : "s"}: total ${trim(tlv.total)} time units.`}
      >
        <defs>
          <Hatch id={hatchId} />
        </defs>
        {ENGINES.map(([eng, label, colour], e) => {
          const y = TOP + e * LANE;
          return (
            <g key={eng} data-engine={eng}>
              <text
                x={2}
                y={y + LANE / 2 + 3}
                style={{ fontSize: fs(9) }}
                className="fill-neutral-700 dark:fill-neutral-300"
              >
                {label}
              </text>
              <rect
                x={LEFT}
                y={y + 3}
                width={W - LEFT - RIGHT}
                height={LANE - 6}
                fill="none"
                stroke={MUTED.light}
                strokeWidth={0.4}
              />
              {tlv.segments.map((sg) => {
                const [a, b] = sg[eng];
                const now = s[eng] === sg.tile;
                const past = b <= s.t;
                const future = a >= s.t && !now;
                return (
                  <g key={sg.tile} data-seg={`${eng}-${sg.tile}`}>
                    <rect
                      x={sx(a) + 0.5}
                      y={y + 4}
                      width={Math.max(0.5, sx(b) - sx(a) - 1)}
                      height={LANE - 8}
                      fill={now ? STATE_COLOUR.active : colour}
                      opacity={future ? 0.18 : past ? 0.55 : 1}
                    />
                    {sx(b) - sx(a) >= fs(9) && (
                      <text
                        x={(sx(a) + sx(b)) / 2}
                        y={y + LANE / 2 + 3}
                        textAnchor="middle"
                        style={{ fontSize: fs(9) }}
                        className={
                          now
                            ? "fill-white font-mono font-semibold"
                            : "fill-neutral-950 font-mono"
                        }
                      >
                        {sg.tile + 1}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          );
        })}
        {/* compute idle before now: hatched, the cost of not overlapping */}
        {tlv.segments.map((sg, k) => {
          const prevEnd = k === 0 ? 0 : tlv.segments[k - 1]!.compute[1];
          const a = prevEnd;
          const b = Math.min(sg.compute[0], s.t);
          if (b - a <= 1e-9) return null;
          return (
            <rect
              key={`idle${k}`}
              x={sx(a)}
              y={TOP + LANE + 4}
              width={sx(b) - sx(a)}
              height={LANE - 8}
              fill={`url(#${hatchId})`}
            />
          );
        })}
        <line
          data-testid="cursor"
          x1={sx(s.t)}
          x2={sx(s.t)}
          y1={TOP}
          y2={TOP + 3 * LANE}
          stroke={STATE_COLOUR.active}
          strokeWidth={1.5}
        />
        <text
          x={Math.min(sx(s.t), W - RIGHT - 30)}
          y={TOP + 3 * LANE + 12}
          style={{ fontSize: fs(9) }}
          className="fill-neutral-800 font-mono dark:fill-neutral-200"
        >
          t={trim(s.t)}
        </text>
        <text
          x={LEFT}
          y={H - 3}
          style={{ fontSize: fs(8) }}
          className="fill-neutral-600 dark:fill-neutral-400"
        >
          axis: 0 to {trim(span)} (one buffer&apos;s total)
        </text>
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Each bar is one tile on one engine (blue = running now). Hatched: the
        compute engine idle, waiting for data.
      </p>
    </div>
  );

  return (
    <AnimationPanel
      testId="timeline-widget"
      title="Overlapping copies and compute"
      summary={`${n} tiles; per tile, load ${tl}, compute ${tc}, store ${ts} time units; ${buffers} buffer${buffers === 1 ? "" : "s"}. The schedule comes from the model.`}
      stepper={st}
      stepLabel="event"
      caption={timelineCaption(steps, st.step, n)}
      visual={visual}
      equation={children}
      hl={hover ?? (buffers > 1 ? "steady" : "serial")}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Total time" value={trim(tlv.total)} />
          <Stat label="One buffer" value={trim(serial.total)} />
          <Stat
            label="Speed-up"
            value={`${trim(serial.total / tlv.total, 3)}×`}
          />
          <Stat label="Compute busy" value={pct(tlv.compute_busy)} />
        </div>
      }
      params={
        <>
          <Segmented
            label="Buffers"
            value={String(buffers)}
            options={[1, 2, 3].map((b) => ({
              value: String(b),
              label: String(b),
            }))}
            onChange={(v) => setBuffers(Number(v))}
          />
          <Segmented
            label="Workload"
            value={scenario}
            options={(Object.keys(SCENARIOS) as Scenario[]).map((k) => ({
              value: k,
              label: SCENARIOS[k].label,
            }))}
            onChange={setScenario}
          />
        </>
      }
    />
  );
}
