"use client";

/**
 * Chapter 9's hero: FlashAttention-2's loop over tiles. The N × N score
 * matrix S = QKᵀ is drawn as a grid of tiles; for each block of queries
 * (a row) the key/value blocks stream past, each score tile is computed
 * and folded into the output on chip, and never written to HBM. Two bars
 * count the HBM bytes moved so far by FlashAttention and, for the same
 * share of the work, by standard attention. Driven by `flashSteps`.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtBytes, fmtKiB, trim } from "@/lib/format";
import { flashCaption } from "@/lib/gpu/captions";
import { attentionTraffic, flashSteps } from "@/lib/gpu/model";
import {
  LEVEL_COLOUR,
  MUTED,
  OKABE_ITO,
  STATE_COLOUR,
} from "@/lib/viz/palette";

const W = 300;
const GX = 44;
const GY = 30;
const G = 200;
const BAR_Y = GY + G + 26;
const H = BAR_Y + 96 + 4;

export default function FlashWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [n, setN] = useState(1024);
  const [d, setD] = useState(64);
  const [B, setB] = useState(128);
  const [hover, setHover] = useState<string | null>(null);
  const font = useSvgFont(W);
  const fs = font.fs;

  const f = useMemo(() => flashSteps(n, d, B, B), [n, d, B]);
  const big = useMemo(() => attentionTraffic(8192, d, B, B), [d, B]);
  const a = f.traffic;
  const st = useStepper(f.steps.length, {
    stepMs: Math.max(45, Math.round(9000 / f.steps.length)),
    resetKey: `${n}-${d}-${B}`,
  });
  const s = f.steps[st.step]!;
  const t = G / a.tr;
  const done = s.i === null ? 0 : (s.i as number) * a.tc + (s.j as number) + 1;
  const bar = (v: number) => ((W - 16) * v) / a.standard;

  const visual = (
    <div className="mx-auto max-w-md">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`FlashAttention over a ${a.tr} by ${a.tc} grid of score tiles: ${done} of ${a.tr * a.tc} done.`}
      >
        <text
          x={GX}
          y={12}
          style={{ fontSize: fs(9) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          K, V blocks (from HBM) →
        </text>
        <text
          x={4}
          y={GY + 10}
          style={{ fontSize: fs(9) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          Q
        </text>
        {/* key/value blocks along the top, query blocks down the side */}
        {Array.from({ length: a.tc }, (_, j) => (
          <rect
            key={`k${j}`}
            x={GX + j * t + 0.5}
            y={GY - 12}
            width={t - 1}
            height={8}
            fill={j === s.j ? LEVEL_COLOUR.hbm : MUTED.light}
            opacity={j === s.j ? 1 : 0.35}
          />
        ))}
        {Array.from({ length: a.tr }, (_, i) => (
          <rect
            key={`q${i}`}
            x={GX - 12}
            y={GY + i * t + 0.5}
            width={8}
            height={t - 1}
            fill={i === s.i ? LEVEL_COLOUR.smem : MUTED.light}
            opacity={i === s.i ? 1 : 0.35}
          />
        ))}
        {/* the score tiles: computed on chip, never stored */}
        {Array.from({ length: a.tr * a.tc }, (_, k) => {
          const i = Math.floor(k / a.tc);
          const j = k % a.tc;
          const isNow = k === done - 1;
          const isDone = k < done - 1;
          return (
            <rect
              key={k}
              data-tile={isNow ? "now" : isDone ? "done" : "todo"}
              x={GX + j * t + 0.5}
              y={GY + i * t + 0.5}
              width={t - 1}
              height={t - 1}
              fill={isNow ? STATE_COLOUR.active : isDone ? MUTED.light : "none"}
              opacity={isDone ? 0.55 : 1}
              stroke={MUTED.light}
              strokeWidth={0.4}
            />
          );
        })}
        <text
          x={GX}
          y={GY + G + 14}
          style={{ fontSize: fs(9) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          S = QKᵀ, {a.tr}×{a.tc} tiles of {B}×{B}: on chip only
        </text>
        {/* HBM traffic so far */}
        {(
          [
            ["Standard attention", s.standard, OKABE_ITO.vermillion, "std"],
            ["Flash, no L2 reuse", s.flash, STATE_COLOUR.active, "flash"],
            ["Flash, K and V from L2", s.flash_l2, LEVEL_COLOUR.l2, "flashl2"],
          ] as const
        ).map(([label, v, colour, key], k) => (
          <g key={label} data-bar={key}>
            <text
              x={4}
              y={BAR_Y + k * 32 + 9}
              style={{ fontSize: fs(9) }}
              className="fill-neutral-700 dark:fill-neutral-300"
            >
              {label}: <tspan className="font-mono">{fmtBytes(v)}</tspan>
            </text>
            <rect
              x={4}
              y={BAR_Y + k * 32 + 13}
              width={W - 16}
              height={12}
              fill="none"
              stroke={MUTED.light}
              strokeWidth={0.6}
            />
            <rect
              x={4}
              y={BAR_Y + k * 32 + 13}
              width={Math.max(0, bar(v))}
              height={12}
              fill={colour}
              opacity={hover === key ? 1 : 0.85}
            />
          </g>
        ))}
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Blue: the score tile being computed now; grey: done (and discarded).
        Bars: HBM bytes so far, to the scale of standard attention&apos;s total.
      </p>
    </div>
  );

  return (
    <AnimationPanel
      testId="flash-widget"
      title="FlashAttention, tile by tile"
      summary={`One attention head, sequence N = ${n}, head dimension d = ${d}, blocks of ${B}, FP16/BF16 values. The byte counts come from the model.`}
      stepper={st}
      stepLabel="tile"
      caption={flashCaption(a, s)}
      visual={visual}
      equation={children}
      hl={hover ?? "kv"}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Standard, total" value={fmtBytes(a.standard)} />
          <Stat
            label="Flash, total"
            value={`${fmtBytes(a.compulsory)} – ${fmtBytes(a.flash)}`}
            hint={`${trim(a.ratio, 2)}× to ${trim(a.ratio_compulsory, 3)}× less`}
          />
          <Stat
            label="At N = 8192"
            value={`${trim(big.ratio, 2)}× to ${trim(big.ratio_compulsory, 3)}× less`}
            hint={`standard: ${fmtBytes(big.standard)}`}
          />
          <Stat
            label="On chip per block"
            value={fmtKiB(a.onchip)}
            hint="Q, K, V tiles + S and O in FP32"
          />
        </div>
      }
      params={
        <>
          <Segmented
            label="Sequence length N"
            value={String(n)}
            options={[512, 1024].map((v) => ({
              value: String(v),
              label: String(v),
            }))}
            onChange={(v) => setN(Number(v))}
          />
          <Segmented
            label="Block size B"
            value={String(B)}
            options={[64, 128].map((v) => ({
              value: String(v),
              label: String(v),
            }))}
            onChange={(v) => setB(Number(v))}
          />
          <Segmented
            label="Head dimension d"
            value={String(d)}
            options={[64, 128].map((v) => ({
              value: String(v),
              label: String(v),
            }))}
            onChange={(v) => setD(Number(v))}
          />
        </>
      }
    />
  );
}
