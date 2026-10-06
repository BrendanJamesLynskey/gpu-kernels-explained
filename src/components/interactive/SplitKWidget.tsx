"use client";

/**
 * Chapter 10's second animation: split-K. A 512 × 512 output in 128 × 128
 * tiles is only 16 blocks, far fewer than the GPU has SMs. Splitting the
 * k dimension multiplies the blocks: each step tries the next split count
 * and colours the SMs by the wave their block runs in, with the time split
 * into computing and adding up the partial sums. Driven by `splitKSweep`.
 */
import { useId, useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtFlops, fmtTime, pct } from "@/lib/format";
import { splitKCaption } from "@/lib/gpu/captions";
import { SPLITK, preset, splitKSweep, type PresetId } from "@/lib/gpu/model";
import {
  LEVEL_COLOUR,
  MUTED,
  OKABE_ITO,
  STATE_COLOUR,
} from "@/lib/viz/palette";

import { Hatch } from "./Hatch";

const W = 300;
const PER_ROW = 12;
const CELL = (W - 8) / PER_ROW;

export default function SplitKWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [pid, setPid] = useState<PresetId>("a100");
  const [hover, setHover] = useState<string | null>(null);
  const hatchId = useId().replace(/:/g, "");
  const font = useSvgFont(W);
  const fs = font.fs;

  const p = useMemo(() => preset(pid), [pid]);
  const sweep = useMemo(() => splitKSweep(p), [p]);
  const st = useStepper(sweep.length, { stepMs: 1400, resetKey: pid });
  const r = sweep[st.step]!;
  const base = sweep[0]!;
  const worst = Math.max(...sweep.map((x) => x.total));
  const rows = Math.ceil(p.sms / PER_ROW);
  const gridH = rows * CELL;
  const barY = 18 + gridH + 16;
  const H = barY + 40;
  const waveColour = [STATE_COLOUR.active, OKABE_ITO.orange, OKABE_ITO.purple];
  const tx = (t: number) => ((W - 8) * t) / worst;

  const visual = (
    <div className="mx-auto max-w-md">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`${r.splits} splits: ${r.blocks} blocks on ${p.sms} SMs in ${r.waves} waves.`}
      >
        <defs>
          <Hatch id={hatchId} />
        </defs>
        <text
          x={4}
          y={12}
          style={{ fontSize: fs(9) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          {p.sms} SMs; one band per wave
        </text>
        {Array.from({ length: p.sms }, (_, sm) => {
          // blocks are handed out round-robin: SM sm runs blocks sm,
          // sm + sms, ... so it is busy in waves 1 .. its block count
          const mine = Math.max(0, Math.ceil((r.blocks - sm) / p.sms));
          const x = 4 + (sm % PER_ROW) * CELL;
          const y = 18 + Math.floor(sm / PER_ROW) * CELL;
          return (
            <g key={sm} data-sm={sm} data-waves={mine}>
              <rect
                x={x + 0.5}
                y={y + 0.5}
                width={CELL - 1}
                height={CELL - 1}
                fill={mine === 0 ? `url(#${hatchId})` : "none"}
                stroke={MUTED.light}
                strokeWidth={0.5}
              />
              {/* one band per wave: coloured if busy in it, hatched if idle */}
              {mine > 0 &&
                Array.from({ length: r.waves }, (_, w) => (
                  <rect
                    key={w}
                    x={x + 2}
                    y={y + 2 + (w * (CELL - 4)) / r.waves}
                    width={CELL - 4}
                    height={(CELL - 4) / r.waves - 0.5}
                    fill={
                      w < mine
                        ? waveColour[w % waveColour.length]
                        : `url(#${hatchId})`
                    }
                  />
                ))}
            </g>
          );
        })}
        <text
          x={4}
          y={barY - 4}
          style={{ fontSize: fs(9) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          time: compute + adding the partial sums
        </text>
        <rect
          data-part="compute"
          x={4}
          y={barY}
          width={tx(r.t_compute)}
          height={14}
          fill={STATE_COLOUR.compute}
          opacity={hover === "waves" ? 1 : 0.85}
        />
        <rect
          data-part="reduce"
          x={4 + tx(r.t_compute)}
          y={barY}
          width={tx(r.t_reduce)}
          height={14}
          fill={LEVEL_COLOUR.hbm}
          opacity={hover === "reduce" ? 1 : 0.85}
        />
        <line
          x1={4 + tx(base.total)}
          x2={4 + tx(base.total)}
          y1={barY - 2}
          y2={barY + 16}
          stroke={MUTED.dark}
          strokeDasharray="2 2"
        />
        <text
          x={4}
          y={barY + 30}
          style={{ fontSize: fs(9) }}
          className="fill-neutral-800 font-mono dark:fill-neutral-200"
        >
          {fmtTime(r.total)}
          {r.splits > 1 ? ` (unsplit: ${fmtTime(base.total)})` : ""}
        </text>
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Bands: blue = busy in wave 1, orange = wave 2, purple = wave 3; hatched
        = idle. Bar: yellow = computing (waves × one block&apos;s time), purple
        = writing and re-reading the FP32 partial sums; dashed = the unsplit
        time.
      </p>
    </div>
  );

  const sh = SPLITK.shape;
  return (
    <AnimationPanel
      testId="splitk-widget"
      title="Split-K: more blocks for a small output"
      summary={`C (${sh.m} × ${sh.n}) = A (${sh.m} × ${sh.k}) · B on the ${p.name}'s BF16 tensor cores, ${sh.bm} × ${sh.bn} tiles, one block per SM at a time (illustrative).`}
      stepper={st}
      stepLabel="try"
      caption={splitKCaption(r, base, p.sms)}
      visual={visual}
      equation={children}
      hl={hover ?? (r.splits > 1 ? "reduce" : "waves")}
      onEquationHover={setHover}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Splits" value={String(r.splits)} />
          <Stat label="Blocks / waves" value={`${r.blocks} / ${r.waves}`} />
          <Stat label="SMs busy" value={pct(r.sm_util)} />
          <Stat label="Achieved" value={fmtFlops(r.achieved)} />
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
