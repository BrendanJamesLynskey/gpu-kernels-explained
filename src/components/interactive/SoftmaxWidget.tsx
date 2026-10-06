"use client";

/**
 * Chapter 9's second animation: online softmax, a block at a time. Top:
 * the scores, the block being read and the running maximum m. Bottom:
 * each score's term e^(x − m) under the current m; when a block raises the
 * maximum, every earlier term shrinks by the same factor e^(m_old − m_new),
 * which is exactly the rescaling of the running sum ℓ. The last step
 * normalises, and the ordinary softmax's probabilities are drawn over the
 * online ones. Driven by `onlineSoftmax`.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { trim } from "@/lib/format";
import { softmaxCaption } from "@/lib/gpu/captions";
import { onlineSoftmax, softmaxInputs } from "@/lib/gpu/model";
import {
  LEVEL_COLOUR,
  MUTED,
  OKABE_ITO,
  STATE_COLOUR,
} from "@/lib/viz/palette";

const W = 300;
const LEFT = 30;
const COL = (W - LEFT - 6) / 16;
const TOP_Y = 14;
const TOP_H = 90;
const BOT_Y = TOP_Y + TOP_H + 26;
const BOT_H = 70;
const H = BOT_Y + BOT_H + 18;
const XMAX = 5;

export const SOFTMAX_SEEDS = [127, 142, 101] as const;

export default function SoftmaxWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [block, setBlock] = useState(4);
  const [seed, setSeed] = useState<number>(127);
  const [hover, setHover] = useState<string | null>(null);
  const font = useSvgFont(W);
  const fs = font.fs;

  const o = useMemo(
    () => onlineSoftmax(softmaxInputs(16, seed), block),
    [seed, block],
  );
  const nSteps = o.steps.length + 2;
  const st = useStepper(nSteps, {
    stepMs: 1500,
    resetKey: `${seed}-${block}`,
  });
  const i = st.step;
  const final = i === nSteps - 1;
  const s = i >= 1 && i <= o.steps.length ? o.steps[i - 1]! : null;
  const seen = final ? 16 : s ? s.hi : 0;
  const m = final ? o.m : s ? s.m : null;
  const l = final ? o.l : s ? s.l : 0;

  const sy = (v: number) => TOP_Y + TOP_H / 2 - (v / XMAX) * (TOP_H / 2);
  const x = (k: number) => LEFT + k * COL;

  const visual = (
    <div className="mx-auto max-w-md">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`Online softmax of 16 scores in blocks of ${block}: ${seen} read, running max ${m === null ? "none yet" : trim(m)}.`}
      >
        <text
          x={2}
          y={TOP_Y + 4}
          style={{ fontSize: fs(8) }}
          className="fill-neutral-600 font-mono dark:fill-neutral-400"
        >
          x
        </text>
        <line
          x1={LEFT}
          x2={W - 6}
          y1={sy(0)}
          y2={sy(0)}
          stroke={MUTED.light}
          strokeWidth={0.6}
        />
        {o.x.map((v, k) => {
          const inBlock = s !== null && !final && k >= s.lo && k < s.hi;
          const read = k < seen;
          return (
            <rect
              key={k}
              data-score={k}
              x={x(k) + 1}
              y={Math.min(sy(v), sy(0))}
              width={COL - 2}
              height={Math.max(0.8, Math.abs(sy(v) - sy(0)))}
              fill={inBlock ? STATE_COLOUR.active : read ? MUTED.light : "none"}
              stroke={read ? "none" : MUTED.light}
              strokeWidth={0.6}
            />
          );
        })}
        {m !== null && (
          <g data-m={m}>
            <line
              x1={LEFT}
              x2={x(seen)}
              y1={sy(m)}
              y2={sy(m)}
              stroke={OKABE_ITO.vermillion}
              strokeWidth={hover === "m" ? 2.5 : 1.5}
              strokeDasharray="4 2"
            />
            <text
              x={Math.min(x(seen) + 3, W - 40)}
              y={sy(m) + 3}
              style={{ fontSize: fs(8) }}
              className="fill-neutral-800 font-mono dark:fill-neutral-200"
            >
              m={trim(m)}
            </text>
          </g>
        )}
        <text
          x={2}
          y={BOT_Y - 8}
          style={{ fontSize: fs(8) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          {final
            ? "p = e^(x − m) / ℓ   (dots: ordinary softmax)"
            : `e^(x − m), summing to ℓ = ${trim(l, 4)}`}
        </text>
        {m !== null &&
          o.x.slice(0, seen).map((v, k) => {
            const term = Math.exp(v - m);
            const val = final ? term / o.l : term;
            const h = val * (final ? BOT_H * 2.2 : BOT_H);
            return (
              <rect
                key={k}
                data-term={k}
                x={x(k) + 1}
                y={BOT_Y + BOT_H - Math.min(BOT_H, h)}
                width={COL - 2}
                height={Math.min(BOT_H, h)}
                fill={
                  final
                    ? LEVEL_COLOUR.smem
                    : s && k >= s.lo
                      ? STATE_COLOUR.active
                      : LEVEL_COLOUR.l2
                }
              />
            );
          })}
        {final &&
          o.ordinary.map((pv, k) => (
            <circle
              key={k}
              data-ordinary={k}
              cx={x(k) + COL / 2}
              cy={BOT_Y + BOT_H - Math.min(BOT_H, pv * BOT_H * 2.2)}
              r={2.2}
              fill={OKABE_ITO.vermillion}
            />
          ))}
        <line
          x1={LEFT}
          x2={W - 6}
          y1={BOT_Y + BOT_H}
          y2={BOT_Y + BOT_H}
          stroke={MUTED.light}
          strokeWidth={0.6}
        />
        <text
          x={LEFT}
          y={H - 4}
          style={{ fontSize: fs(8) }}
          className="fill-neutral-600 font-mono dark:fill-neutral-400"
        >
          x0 … x15, blocks of {block}
        </text>
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Blue: the block being read. Dashed: the running maximum m. When m rises,
        every earlier term shrinks by the same factor: that is the rescaling.
      </p>
    </div>
  );

  return (
    <AnimationPanel
      testId="softmax-widget"
      title="Online softmax, block by block"
      summary="One pass over the scores, keeping only a running maximum and a running sum, as FlashAttention does for each row of scores."
      stepper={st}
      stepLabel="block"
      caption={softmaxCaption(o, i)}
      visual={visual}
      equation={children}
      hl={hover ?? (s && s.m_prev !== null && s.m > s.m_prev ? "scale" : "l")}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat
            label="m (running max)"
            value={m === null ? "−∞" : trim(m)}
            plain
          />
          <Stat label="ℓ (running sum)" value={trim(l, 5)} plain />
          <Stat
            label="Rescale this block"
            value={s && s.m_prev !== null ? `× ${trim(s.scale, 4)}` : "–"}
          />
          <Stat
            label="max |online − ordinary|"
            value={o.max_diff === 0 ? "0" : o.max_diff.toExponential(1)}
            hint="rounding only"
            plain
          />
        </div>
      }
      params={
        <>
          <Segmented
            label="Block size"
            value={String(block)}
            options={[2, 4, 8, 16].map((b) => ({
              value: String(b),
              label: String(b),
            }))}
            onChange={(v) => setBlock(Number(v))}
          />
          <Segmented
            label="Scores"
            value={String(seed)}
            options={SOFTMAX_SEEDS.map((sd, k) => ({
              value: String(sd),
              label: ["set A", "set B", "set C"][k]!,
            }))}
            onChange={(v) => setSeed(Number(v))}
          />
        </>
      }
    />
  );
}
