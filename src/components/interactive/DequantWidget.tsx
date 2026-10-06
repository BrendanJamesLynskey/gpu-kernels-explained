"use client";

/**
 * Chapter 11's hero: dequantising in registers. One 32-bit load brings
 * eight 4-bit weights; for each, the kernel shifts and masks (q), subtracts
 * the offset and multiplies by the group's scale (w), then does a fused
 * multiply-add with the activation. The weights never exist in BF16 in
 * memory: only in a register, for one instruction. Driven by
 * `dequantSteps`; the data are chosen so that every value is exact.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { trim } from "@/lib/format";
import { dequantCaption } from "@/lib/gpu/captions";
import { dequantSteps } from "@/lib/gpu/model";
import {
  LEVEL_COLOUR,
  MUTED,
  OKABE_ITO,
  STATE_COLOUR,
} from "@/lib/viz/palette";

const W = 300;
const NIB = 34;
const RX = (W - 8 * NIB) / 2;
const STAGE_Y = 66;
const TABLE_Y = 130;
const H = TABLE_Y + 96;

export const DEQUANT_SEEDS = [11, 21, 31] as const;
export const DEQUANT_SCALES = [0.0625, 0.125, 0.25] as const;

export default function DequantWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [seed, setSeed] = useState<number>(11);
  const [scale, setScale] = useState<number>(0.0625);
  const [hover, setHover] = useState<string | null>(null);
  const font = useSvgFont(W);
  const fs = font.fs;

  const d = useMemo(() => dequantSteps(seed, scale), [seed, scale]);
  const st = useStepper(d.steps.length, {
    stepMs: 650,
    resetKey: `${seed}-${scale}`,
  });
  const s = d.steps[st.step]!;
  const cur = s.i;
  // nibble i sits at bits 4i..4i+3: drawn most significant first
  const nibX = (i: number) => RX + (7 - i) * NIB;
  const stages = [
    ["unpack", "shift, mask", s.q === null ? "" : `q = ${s.q}`],
    [
      "scale",
      `(q − 8) × ${trim(d.scale, 4)}`,
      s.w === null ? "" : `w = ${trim(s.w, 6)}`,
    ],
    ["fma", "acc += w·x", `acc = ${trim(s.acc, 6)}`],
  ] as const;
  const wMax = 8 * scale;

  const visual = (
    <div className="mx-auto max-w-md">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`A 32-bit register holding eight 4-bit weights; step ${st.step} of ${d.steps.length - 1}.`}
      >
        <text
          x={RX}
          y={12}
          style={{ fontSize: fs(9) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          one 32-bit register (bits 31 … 0)
        </text>
        {Array.from({ length: 8 }, (_, i) => {
          const q = (d.word >>> (4 * i)) & 0xf;
          const now = i === cur;
          const used = cur > i || (cur === i && s.stage !== "unpack");
          return (
            <g key={i} data-nibble={i} data-now={now || undefined}>
              <rect
                x={nibX(i) + 1}
                y={20}
                width={NIB - 2}
                height={28}
                rx={3}
                fill={
                  now
                    ? STATE_COLOUR.active
                    : used
                      ? MUTED.light
                      : LEVEL_COLOUR.reg
                }
                opacity={used && !now ? 0.5 : 1}
              />
              <text
                x={nibX(i) + NIB / 2}
                y={34 + fs(12) / 3}
                textAnchor="middle"
                style={{ fontSize: fs(12) }}
                className="fill-neutral-950 font-mono font-semibold"
              >
                {q.toString(16).toUpperCase()}
              </text>
            </g>
          );
        })}
        {/* the three instructions per weight */}
        {stages.map(([key, label, value], k) => {
          const x = 4 + k * ((W - 8) / 3);
          const w = (W - 8) / 3 - 6;
          const on = s.stage === key;
          return (
            <g key={key} data-stage={key} data-on={on || undefined}>
              <rect
                x={x}
                y={STAGE_Y}
                width={w}
                height={44}
                rx={4}
                fill="none"
                stroke={on ? STATE_COLOUR.active : MUTED.light}
                strokeWidth={on ? 2.5 : 1}
              />
              <text
                x={x + w / 2}
                y={STAGE_Y + 16}
                textAnchor="middle"
                style={{ fontSize: fs(9) }}
                className="fill-neutral-600 dark:fill-neutral-400"
              >
                {label}
              </text>
              <text
                x={x + w / 2}
                y={STAGE_Y + 34}
                textAnchor="middle"
                style={{ fontSize: fs(9.5) }}
                className="fill-neutral-900 font-mono dark:fill-neutral-100"
              >
                {on || (key === "fma" && cur >= 0) ? value : ""}
              </text>
            </g>
          );
        })}
        {/* per weight: q, the dequantised value as a signed bar, x */}
        <text
          x={4}
          y={TABLE_Y + 10}
          style={{ fontSize: fs(8.5) }}
          className="fill-neutral-600 font-mono dark:fill-neutral-400"
        >
          q
        </text>
        <text
          x={4}
          y={TABLE_Y + 50}
          style={{ fontSize: fs(8.5) }}
          className="fill-neutral-600 font-mono dark:fill-neutral-400"
        >
          w
        </text>
        <text
          x={4}
          y={TABLE_Y + 90}
          style={{ fontSize: fs(8.5) }}
          className="fill-neutral-600 font-mono dark:fill-neutral-400"
        >
          x
        </text>
        <line
          x1={RX}
          x2={RX + 8 * NIB}
          y1={TABLE_Y + 46}
          y2={TABLE_Y + 46}
          stroke={MUTED.light}
          strokeWidth={0.6}
        />
        {Array.from({ length: 8 }, (_, i) => {
          const x = nibX(i) + NIB / 2;
          const q = d.q[i] as number;
          const w = (q - 8) * d.scale;
          const hasQ = cur > i || (cur === i && s.stage !== "load");
          const hasW = cur > i || (cur === i && s.stage !== "unpack");
          const h = (Math.abs(w) / wMax) * 26;
          return (
            <g key={i} data-weight={i}>
              <text
                x={x}
                y={TABLE_Y + 10}
                textAnchor="middle"
                style={{ fontSize: fs(9) }}
                className="fill-neutral-800 font-mono dark:fill-neutral-200"
              >
                {hasQ ? q : "·"}
              </text>
              {hasW && (
                <rect
                  x={x - 7}
                  y={w >= 0 ? TABLE_Y + 46 - h : TABLE_Y + 46}
                  width={14}
                  height={Math.max(0.8, h)}
                  fill={i === cur ? STATE_COLOUR.active : OKABE_ITO.green}
                />
              )}
              <text
                x={x}
                y={TABLE_Y + 90}
                textAnchor="middle"
                style={{ fontSize: fs(9) }}
                className="fill-neutral-800 font-mono dark:fill-neutral-200"
              >
                {trim(d.x[i] as number).replace("-", "−")}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Orange: weights still packed in the register; blue: the one being worked
        on now. Bars: the dequantised weights (up = positive).
      </p>
    </div>
  );

  let reference = 0;
  d.q.forEach((q, i) => (reference += (q - 8) * d.scale * (d.x[i] as number)));
  return (
    <AnimationPanel
      testId="dequant-widget"
      title="Dequantising in registers"
      summary="Eight INT4 weights from one 32-bit register, each turned into a real number for one fused multiply-add. Values from the model."
      stepper={st}
      stepLabel="instruction"
      caption={dequantCaption(d, st.step)}
      visual={visual}
      equation={children}
      hl={
        hover ??
        (s.stage === "unpack" ? "q" : s.stage === "scale" ? "s" : "acc")
      }
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat
            label="Register"
            value={`0x${d.word.toString(16).toUpperCase().padStart(8, "0")}`}
          />
          <Stat label="Group scale" value={String(d.scale)} />
          <Stat label="acc" value={trim(s.acc, 6)} plain />
          <Stat
            label="Dot product (check)"
            value={trim(reference, 6)}
            hint="Σ (q − 8)·s·x"
          />
        </div>
      }
      params={
        <>
          <Segmented
            label="Weights"
            value={String(seed)}
            options={DEQUANT_SEEDS.map((sd, k) => ({
              value: String(sd),
              label: ["set A", "set B", "set C"][k]!,
            }))}
            onChange={(v) => setSeed(Number(v))}
          />
          <Segmented
            label="Scale s"
            value={String(scale)}
            options={DEQUANT_SCALES.map((sc) => ({
              value: String(sc),
              label: String(sc),
            }))}
            onChange={(v) => setScale(Number(v))}
          />
        </>
      }
    />
  );
}
