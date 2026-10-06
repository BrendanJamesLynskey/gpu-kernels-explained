"use client";

/**
 * Chapter 11's second animation: an 8192 × 8192 weight matrix applied to
 * a growing batch of tokens, in four formats. The time of each (log scale)
 * against the batch (log scale): flat while HBM bytes set the pace, rising
 * once the tensor cores do. The cursor steps through the batch sizes.
 * Driven by `quantSweep`.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtBytes, fmtTime, trim } from "@/lib/format";
import { QUANT_NAME, quantCaption } from "@/lib/gpu/captions";
import {
  QUANT_BATCHES,
  QUANT_ORDER,
  QUANT_SHAPE,
  preset,
  quantSweep,
  type PresetId,
  type QuantFormat,
} from "@/lib/gpu/model";
import { MUTED, OKABE_ITO, STATE_COLOUR } from "@/lib/viz/palette";

const W = 320;
const H = 210;
const M = { l: 40, r: 12, t: 12, b: 30 };

/** Colour and dash per format: never colour alone. */
const STYLE: Record<QuantFormat, { c: string; dash: string }> = {
  bf16: { c: MUTED.dark, dash: "" },
  int8: { c: OKABE_ITO.sky, dash: "6 3" },
  int4: { c: OKABE_ITO.green, dash: "2 2" },
  w8a8: { c: OKABE_ITO.vermillion, dash: "8 2 2 2" },
};

export default function QuantWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [pid, setPid] = useState<PresetId>("a100");
  const [hover, setHover] = useState<string | null>(null);
  const font = useSvgFont(W);
  const fs = font.fs;

  const p = useMemo(() => preset(pid), [pid]);
  const sweeps = useMemo(() => QUANT_ORDER.map((f) => quantSweep(p, f)), [p]);
  const st = useStepper(QUANT_BATCHES.length, {
    stepMs: 1100,
    resetKey: pid,
  });
  const rows = sweeps.map((sw) => sw[st.step]!);
  const all = sweeps.flat().map((r) => r.total);
  const y0 = Math.log10(Math.min(...all) / 1.3);
  const y1 = Math.log10(Math.max(...all) * 1.3);
  const x0 = 0;
  const x1 = Math.log2(QUANT_BATCHES[QUANT_BATCHES.length - 1]!);
  const sx = (b: number) =>
    M.l + ((Math.log2(b) - x0) / (x1 - x0)) * (W - M.l - M.r);
  const sy = (t: number) =>
    H - M.b - ((Math.log10(t) - y0) / (y1 - y0)) * (H - M.t - M.b);
  const batch = QUANT_BATCHES[st.step]!;
  const xTicks = font.narrow ? [1, 16, 256] : [1, 4, 16, 64, 256, 1024];
  const yTicks = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3].filter(
    (t) => Math.log10(t) >= y0 && Math.log10(t) <= y1,
  );

  const visual = (
    <div className="mx-auto max-w-lg">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`Time of an 8192 by 8192 layer against batch size on the ${p.name}, four weight formats; now at batch ${batch}.`}
      >
        {xTicks.map((b) => (
          <g key={b}>
            <line
              x1={sx(b)}
              x2={sx(b)}
              y1={M.t}
              y2={H - M.b}
              stroke={MUTED.light}
              strokeOpacity={0.3}
            />
            <text
              x={sx(b)}
              y={H - M.b + 12}
              textAnchor="middle"
              style={{ fontSize: fs(9) }}
              className="fill-neutral-600 font-mono dark:fill-neutral-400"
            >
              {b}
            </text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={M.l}
              x2={W - M.r}
              y1={sy(t)}
              y2={sy(t)}
              stroke={MUTED.light}
              strokeOpacity={0.3}
            />
            <text
              x={M.l - 3}
              y={sy(t) + 3}
              textAnchor="end"
              style={{ fontSize: fs(9) }}
              className="fill-neutral-600 font-mono dark:fill-neutral-400"
            >
              {fmtTime(t).replace(" ", "")}
            </text>
          </g>
        ))}
        <text
          x={(W + M.l) / 2}
          y={H - 4}
          textAnchor="middle"
          style={{ fontSize: fs(9) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          tokens in the batch (log)
        </text>
        {sweeps.map((sw, k) => {
          const f = QUANT_ORDER[k]!;
          const d = sw
            .map((r, i) => `${i ? "L" : "M"}${sx(r.batch)},${sy(r.total)}`)
            .join(" ");
          return (
            <path
              key={f}
              data-format={f}
              d={d}
              fill="none"
              stroke={STYLE[f].c}
              strokeWidth={hover === f ? 3 : 1.8}
              strokeDasharray={STYLE[f].dash}
            />
          );
        })}
        <line
          x1={sx(batch)}
          x2={sx(batch)}
          y1={M.t}
          y2={H - M.b}
          stroke={STATE_COLOUR.active}
          strokeWidth={1.5}
        />
        {rows.map((r) => (
          <circle
            key={r.fmt}
            cx={sx(r.batch)}
            cy={sy(r.total)}
            r={3.5}
            fill={STYLE[r.fmt].c}
            stroke="white"
            strokeWidth={1}
          />
        ))}
      </svg>
      <ul className="mt-1 flex flex-wrap justify-center gap-x-4 gap-y-1 text-[0.7rem] text-neutral-700 dark:text-neutral-300">
        {QUANT_ORDER.map((f) => (
          <li key={f} className="flex items-center gap-1">
            <svg aria-hidden width="22" height="8">
              <line
                x1="0"
                x2="22"
                y1="4"
                y2="4"
                stroke={STYLE[f].c}
                strokeWidth="2"
                strokeDasharray={STYLE[f].dash}
              />
            </svg>
            {QUANT_NAME[f]}
          </li>
        ))}
      </ul>
    </div>
  );

  const b = rows[0]!;
  const q4 = rows[2]!;
  return (
    <AnimationPanel
      testId="quant-widget"
      title="Fewer bytes per weight, until the maths takes over"
      summary={`Y = X·Wᵀ, W ${QUANT_SHAPE.rows} × ${QUANT_SHAPE.cols}, on the ${p.name}: the model's roofline time of each format, batch 1 to ${QUANT_BATCHES[QUANT_BATCHES.length - 1]}.`}
      stepper={st}
      stepLabel="point"
      caption={quantCaption(rows)}
      visual={visual}
      equation={children}
      hl={hover ?? (b.bound === "memory" ? "bytes" : "flops")}
      onEquationHover={setHover}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Tokens" value={String(batch)} />
          <Stat label="BF16 weights" value={fmtBytes(b.weight_bytes)} />
          <Stat
            label="INT4 weights + scales"
            value={fmtBytes(q4.weight_bytes)}
          />
          <Stat
            label="INT4 speed-up"
            value={`${trim(b.total / q4.total, 3)}×`}
            hint={`${q4.bound}-bound`}
          />
        </div>
      }
      params={
        <Segmented
          label="GPU"
          value={pid}
          options={[
            { value: "a100", label: "A100" },
            { value: "h100", label: "H100" },
          ]}
          onChange={setPid}
        />
      }
    />
  );
}
