"use client";

/**
 * Chapter 5's hero: shared memory drawn as rows of 32 four-byte words, so
 * a word's column is its bank. The warp's 32 requests land one lane at a
 * time; the heat map underneath counts the different words queued at each
 * bank. Then the banks serve the warp in passes, one word per bank per
 * pass. Padding the array by one word moves a column access onto a
 * diagonal and the conflict disappears. Driven by `bankSteps`.
 */
import { useId, useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Slider, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { pct } from "@/lib/format";
import { bankCaption, bankStepCount } from "@/lib/gpu/captions";
import { bankSteps, type BankPattern } from "@/lib/gpu/model";
import { LEVEL_COLOUR, MUTED, STATE_COLOUR } from "@/lib/viz/palette";

import { Hatch } from "./Hatch";

const C = 8;
const LEFT = 30;
const W = LEFT + 32 * C + 6;

export default function BankWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [pattern, setPattern] = useState<BankPattern>("col");
  const [pad, setPad] = useState(0);
  const [stride, setStride] = useState(2);
  const [hover, setHover] = useState<string | null>(null);
  const hatchId = useId().replace(/:/g, "");
  const font = useSvgFont(W);
  const fs = font.fs;

  const r = useMemo(
    () => bankSteps(pattern, pad, stride),
    [pattern, pad, stride],
  );
  const n = bankStepCount(r);
  const st = useStepper(n, {
    stepMs: 260,
    resetKey: `${pattern}-${pad}-${stride}`,
  });
  const i = st.step;
  const requesting = i < 32;
  const passesDone = requesting ? 0 : i - 31;

  const view = useMemo(() => {
    const maxWord = Math.max(...r.words);
    const rows = Math.floor(maxWord / 32) + 1;
    const servedBy: number[] = new Array(32).fill(-1);
    r.passes.forEach((p) => p.lanes.forEach((l) => (servedBy[l] = p.pass)));
    // distinct words per bank, in queue order
    const queues: number[][] = Array.from({ length: 32 }, () => []);
    r.words.forEach((w) => {
      const q = queues[w % 32]!;
      if (!q.includes(w)) q.push(w);
    });
    return { rows, servedBy, queues };
  }, [r]);

  const gridH = view.rows * C;
  const heatTop = 14 + gridH + 22;
  const heatH = r.degree * 3 + 4;
  const H = heatTop + Math.max(heatH, 24) + 16;
  const curLane = requesting ? i : -1;
  const reqLoad = requesting ? r.requests[i]!.load : null;

  const cellState = (
    lane: number,
  ): "now" | "requested" | "served" | "waiting" | null => {
    if (requesting)
      return lane === curLane ? "now" : lane < curLane ? "requested" : null;
    return (view.servedBy[lane] as number) < passesDone ? "served" : "waiting";
  };

  const visual = (
    <div className="mx-auto max-w-md">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`Shared memory as rows of 32 words; the access is a ${r.degree}-way bank conflict.`}
      >
        <defs>
          <Hatch id={hatchId} />
        </defs>
        {[0, 8, 16, 24, 31].map((b) => (
          <text
            key={b}
            x={LEFT + b * C + C / 2}
            y={10}
            textAnchor="middle"
            style={{ fontSize: fs(6.5) }}
            className="fill-neutral-500 font-mono dark:fill-neutral-400"
          >
            {b}
          </text>
        ))}
        <text
          x={0}
          y={10}
          style={{ fontSize: fs(6.5) }}
          className="fill-neutral-600 font-mono dark:fill-neutral-400"
        >
          bank
        </text>
        {/* the memory: faint grid of every word */}
        {Array.from({ length: view.rows }, (_, row) => (
          <g key={row}>
            {row % 4 === 0 && (
              <text
                x={LEFT - 3}
                y={14 + row * C + 6}
                textAnchor="end"
                style={{ fontSize: fs(6) }}
                className="fill-neutral-500 font-mono dark:fill-neutral-400"
              >
                {row * 32}
              </text>
            )}
            <rect
              x={LEFT}
              y={14 + row * C}
              width={32 * C}
              height={C}
              fill="none"
              stroke={MUTED.light}
              strokeOpacity={0.35}
              strokeWidth={0.5}
            />
          </g>
        ))}
        {/* the words the warp asks for */}
        {r.words.map((w, lane) => {
          const s = cellState(lane);
          if (!s) return null;
          const fill =
            s === "now"
              ? STATE_COLOUR.active
              : s === "requested"
                ? MUTED.light
                : s === "served"
                  ? LEVEL_COLOUR.smem
                  : `url(#${hatchId})`;
          return (
            <rect
              key={lane}
              data-lane={lane}
              x={LEFT + (w % 32) * C + 0.5}
              y={14 + Math.floor(w / 32) * C + 0.5}
              width={C - 1}
              height={C - 1}
              fill={fill}
            />
          );
        })}
        {/* the heat map: different words queued per bank */}
        <text
          x={0}
          y={heatTop - 6}
          style={{ fontSize: fs(7.5) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          {font.narrow ? "queued per bank " : "words queued per bank "}
          {requesting
            ? "(requests arriving)"
            : font.narrow
              ? `(pass ${passesDone}/${r.degree})`
              : `(after pass ${passesDone} of ${r.degree})`}
        </text>
        {view.queues.map((q, b) => {
          const total = requesting ? (reqLoad![b] as number) : q.length;
          const served = requesting ? 0 : Math.min(passesDone, q.length);
          return (
            <g key={b} data-bank={b} data-queued={total - served}>
              {Array.from({ length: total }, (_, k) => {
                const y = heatTop + heatH - (k + 1) * 3;
                const isServed = k < served;
                return (
                  <rect
                    key={k}
                    x={LEFT + b * C + 0.5}
                    y={y}
                    width={C - 1}
                    height={2.5}
                    fill={
                      isServed
                        ? MUTED.light
                        : k === 0
                          ? LEVEL_COLOUR.smem
                          : `url(#${hatchId})`
                    }
                    opacity={isServed ? 0.5 : 1}
                  />
                );
              })}
            </g>
          );
        })}
        <line
          x1={LEFT}
          x2={LEFT + 32 * C}
          y1={heatTop + heatH}
          y2={heatTop + heatH}
          stroke={MUTED.light}
        />
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Column = bank. Blue: the word this lane asks for. Green: served.
        Hatched: waiting for a later pass (a conflict).
      </p>
    </div>
  );

  return (
    <AnimationPanel
      testId="bank-widget"
      title="Bank conflicts, request by request"
      summary={
        pattern === "row" || pattern === "col"
          ? `float s[32][${32 + pad}]; lane i reads ${pattern === "row" ? "s[0][i] (a row)" : "s[i][0] (a column, as a transpose does)"}.`
          : pattern === "stride"
            ? `Lane i reads word i × ${stride}.`
            : "Lanes 0–7 read word 0, lanes 8–15 word 1, and so on: shared words are broadcast."
      }
      stepper={st}
      stepLabel="step"
      caption={bankCaption(r, i)}
      visual={visual}
      equation={children}
      hl={hover ?? "degree"}
      onEquationHover={setHover}
      stats={
        <div className="grid grid-cols-3 gap-2">
          <Stat label="Conflict degree" value={`${r.degree}-way`} />
          <Stat label="Passes" value={String(r.degree)} />
          <Stat label="Bandwidth used" value={pct(1 / r.degree, 1)} />
        </div>
      }
      params={
        <>
          <Segmented
            label="Access"
            value={pattern}
            options={[
              { value: "row", label: "row" },
              { value: "col", label: "column" },
              { value: "stride", label: "stride" },
              { value: "broadcast", label: "broadcast" },
            ]}
            onChange={setPattern}
          />
          {pattern === "row" || pattern === "col" ? (
            <Slider
              label="Padding (words per row)"
              value={pad}
              min={0}
              max={4}
              onChange={setPad}
            />
          ) : pattern === "stride" ? (
            <Slider
              label="Stride (words)"
              value={stride}
              min={1}
              max={33}
              onChange={setStride}
            />
          ) : null}
        </>
      }
    />
  );
}
