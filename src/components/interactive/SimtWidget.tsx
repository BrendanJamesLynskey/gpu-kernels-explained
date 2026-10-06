"use client";

/**
 * Chapter 3's hero: one warp of 32 lanes meets a branch. Each issued
 * instruction is a row: lanes that execute it are solid, lanes masked off
 * are hatched (idle). A divergent branch issues both paths one after the
 * other; a uniform one skips the path no lane takes. Driven by `simtSteps`.
 */
import { useId, useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Slider, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { pct } from "@/lib/format";
import { simtCaption } from "@/lib/gpu/captions";
import { laneData, popcount, simtSteps, type Cond } from "@/lib/gpu/model";
import { MUTED, STATE_COLOUR } from "@/lib/viz/palette";

import { Hatch } from "./Hatch";

const CELL = 10;
const LEFT = 54;
const W = LEFT + 32 * CELL + 4;

const K_RANGE: Record<Cond, [number, number, number]> = {
  lt: [0, 32, 8],
  mod: [1, 32, 2],
  data: [0, 100, 50],
  uniform: [0, 1, 1],
};

function condText(cond: Cond, k: number): string {
  if (cond === "lt") return `i < ${k}`;
  if (cond === "mod") return `i % ${k} == 0`;
  if (cond === "data") return `x[i] < ${k}`;
  return `blockIdx.x < ${k}`;
}

export default function SimtWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [cond, setCond] = useState<Cond>("lt");
  const [k, setK] = useState(8);
  const [lenA, setLenA] = useState(3);
  const [lenB, setLenB] = useState(2);
  const [lenC, setLenC] = useState(2);
  const [hover, setHover] = useState<string | null>(null);
  const hatchId = useId().replace(/:/g, "");
  const font = useSvgFont(W);
  const fs = font.fs;
  // phones: per-lane data does not fit as labels; it is listed below instead
  const laneLabels = !font.narrow || cond !== "data";
  // phones: the row labels are taller than a row; space the rows out
  const gap = font.narrow ? 5 : 3;

  const r = useMemo(
    () => simtSteps(cond, k, lenA, lenB, lenC),
    [cond, k, lenA, lenB, lenC],
  );
  const data = useMemo(() => laneData(2024), []);
  const st = useStepper(r.steps.length, {
    stepMs: 800,
    resetKey: `${cond}-${k}-${lenA}-${lenB}-${lenC}`,
  });
  const cur = r.steps[st.step]!;

  // the source, one line per instruction, and which line each step issues
  const lines: { text: string; key: string }[] = [
    { text: "int i = threadIdx.x;", key: "pre-0" },
    { text: `if (${condText(cond, k)}) {`, key: "branch-0" },
    ...Array.from({ length: lenA }, (_, j) => ({
      text: `  a = f${j}(a);`,
      key: `A-${j}`,
    })),
    { text: "} else {", key: "else" },
    ...Array.from({ length: lenB }, (_, j) => ({
      text: `  b = g${j}(b);`,
      key: `B-${j}`,
    })),
    { text: "}", key: "close" },
    ...Array.from({ length: lenC }, (_, j) => ({
      text: `c = h${j}(a, b);`,
      key: `C-${j}`,
    })),
  ];
  const curKey = `${cur.phase}-${cur.j}`;

  const rows = r.steps.slice(0, st.step + 1);
  const H = 22 + r.steps.length * (CELL + gap) + 4;

  const visual = (
    <div className="grid min-w-0 gap-3 md:grid-cols-[minmax(0,13rem)_minmax(0,1fr)]">
      <pre
        tabIndex={0}
        aria-label="The kernel's code; the highlighted line is being issued"
        className="m-0 overflow-x-auto rounded bg-white p-2 font-mono text-[0.72rem] leading-5 ring-1 ring-neutral-200 dark:bg-neutral-950 dark:ring-neutral-800"
      >
        {lines.map((l) => (
          <div
            key={l.key}
            data-current={l.key === curKey ? "true" : undefined}
            className={
              l.key === curKey
                ? "rounded bg-blue-100 font-semibold text-neutral-950 dark:bg-blue-950 dark:text-white"
                : "text-neutral-700 dark:text-neutral-300"
            }
          >
            {l.text}
          </div>
        ))}
      </pre>
      <div className="min-w-0">
        <svg
          ref={font.ref}
          viewBox={`0 0 ${W} ${H}`}
          className="h-auto w-full"
          role="img"
          aria-label={`A warp of 32 lanes: ${r.issued} instructions issued, SIMT efficiency ${pct(r.efficiency)}.`}
        >
          <defs>
            <Hatch id={hatchId} />
          </defs>
          {Array.from({ length: 32 }, (_, lane) => (
            <text
              key={lane}
              x={LEFT + lane * CELL + CELL / 2}
              y={10}
              textAnchor="middle"
              style={{ fontSize: fs(6.5) }}
              className="fill-neutral-500 font-mono dark:fill-neutral-400"
            >
              {cond === "data" && laneLabels
                ? data[lane]
                : lane % (font.narrow ? 8 : 4) === 0
                  ? lane
                  : ""}
            </text>
          ))}
          <text
            x={2}
            y={10}
            style={{ fontSize: fs(7) }}
            className="fill-neutral-600 font-mono dark:fill-neutral-400"
          >
            {cond === "data" && laneLabels ? "x[i]" : "lane"}
          </text>
          {rows.map((s, i) => {
            const y = 18 + i * (CELL + gap);
            const isCur = i === st.step;
            return (
              <g key={i} data-row={i}>
                <text
                  x={2}
                  y={y + 8}
                  style={{ fontSize: fs(7.5) }}
                  className="fill-neutral-700 font-mono dark:fill-neutral-300"
                >
                  {s.phase === "pre"
                    ? "i=…"
                    : s.phase === "branch"
                      ? "if"
                      : `${s.phase}${s.j}`}
                  {font.narrow ? "" : ` ${popcount(s.mask)}/32`}
                </text>
                {Array.from({ length: 32 }, (_, lane) => {
                  const on = ((s.mask >>> lane) & 1) === 1;
                  return (
                    <rect
                      key={lane}
                      x={LEFT + lane * CELL + 0.5}
                      y={y}
                      width={CELL - 1}
                      height={CELL}
                      fill={
                        on
                          ? isCur
                            ? STATE_COLOUR.active
                            : MUTED.light
                          : `url(#${hatchId})`
                      }
                      opacity={on && !isCur ? 0.8 : 1}
                    />
                  );
                })}
                {isCur && (
                  <rect
                    x={LEFT - 1}
                    y={y - 1.5}
                    width={32 * CELL + 1}
                    height={CELL + 3}
                    fill="none"
                    stroke={STATE_COLOUR.active}
                    strokeWidth={1}
                  />
                )}
              </g>
            );
          })}
        </svg>
        {!laneLabels && (
          <p className="mt-1 break-words font-mono text-[0.7rem] text-neutral-600 dark:text-neutral-400">
            x[0..31] = {data.join(", ")}
          </p>
        )}
        <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
          Solid: the lane executes this instruction (blue = now, grey = done).
          Hatched: masked off, idle.
        </p>
      </div>
    </div>
  );

  const [kmin, kmax] = K_RANGE[cond];
  return (
    <AnimationPanel
      testId="simt-widget"
      title="A warp meets a branch"
      summary="32 lanes share one instruction stream. The model lists every instruction the warp issues and the lanes active for it."
      stepper={st}
      stepLabel="issue"
      caption={simtCaption(r, st.step, [lenA, lenB, lenC])}
      visual={visual}
      equation={children}
      hl={
        hover ??
        (cur.phase === "A" || cur.phase === "B" ? "active issued" : "issued")
      }
      onEquationHover={setHover}
      stats={
        <div className="grid grid-cols-3 gap-2">
          <Stat label="Issued" value={String(r.issued)} />
          <Stat
            label="Lane-slots used"
            value={`${r.active_lane_slots} / ${32 * r.issued}`}
          />
          <Stat label="SIMT efficiency" value={pct(r.efficiency, 1)} />
        </div>
      }
      params={
        <>
          <Segmented
            label="Branch condition"
            value={cond}
            options={[
              { value: "lt", label: "i < k" },
              { value: "mod", label: "i % k == 0" },
              { value: "data", label: "x[i] < k" },
              { value: "uniform", label: "uniform" },
            ]}
            onChange={(c) => {
              setCond(c);
              setK(K_RANGE[c][2]);
            }}
          />
          <Slider
            label="k"
            value={Math.min(Math.max(k, kmin), kmax)}
            min={kmin}
            max={kmax}
            onChange={setK}
          />
          <Slider
            label="If-path length"
            value={lenA}
            min={0}
            max={5}
            onChange={setLenA}
          />
          <Slider
            label="Else-path length"
            value={lenB}
            min={0}
            max={5}
            onChange={setLenB}
          />
          <Slider
            label="Instructions after the branch"
            value={lenC}
            min={0}
            max={3}
            onChange={setLenC}
          />
        </>
      }
    />
  );
}
