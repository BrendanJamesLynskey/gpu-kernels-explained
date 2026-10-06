"use client";

/**
 * Chapter 2's hero: the roofline. The animation steps a square GEMM through
 * tile sizes 1 … 128 (`rooflineSweep`), and the kernel's point slides from
 * the memory-bound slope to the compute-bound roof as its arithmetic
 * intensity grows. Below it, a free slider places any intensity on the
 * roof. Log-log axes; both roofs (FP32 CUDA cores, BF16 tensor cores) are
 * the preset's sourced peaks and HBM bandwidth.
 */
import { useId, useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Slider, Stat } from "@/components/ui/Controls";
import { fmtAi, fmtFlops, pct, trim } from "@/lib/format";
import { sweepCaption } from "@/lib/gpu/captions";
import {
  derived,
  preset,
  rooflinePoint,
  rooflineSweep,
  type Engine,
  type PresetId,
} from "@/lib/gpu/model";
import {
  LEVEL_COLOUR,
  MUTED,
  OKABE_ITO,
  STATE_COLOUR,
} from "@/lib/viz/palette";

const W = 360;
const H = 250;
const M = { l: 46, r: 10, t: 12, b: 34 };
const AI_MIN = 1 / 16;
const AI_MAX = 1024;

export default function RooflineWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [pid, setPid] = useState<PresetId>("a100");
  const [engine, setEngine] = useState<Engine>("fp32");
  const [log2ai, setLog2ai] = useState(0);
  const [hover, setHover] = useState<string | null>(null);
  const clipId = useId().replace(/:/g, "");

  const p = useMemo(() => preset(pid), [pid]);
  const d = useMemo(() => derived(p), [p]);
  const sweep = useMemo(() => rooflineSweep(p), [p]);
  const st = useStepper(sweep.length, {
    stepMs: 1100,
    smooth: true,
    resetKey: pid,
  });

  const yMin = 1e11;
  const yMax = p.peak_tensor * 2;
  const lx0 = Math.log10(AI_MIN);
  const lx1 = Math.log10(AI_MAX);
  const ly0 = Math.log10(yMin);
  const ly1 = Math.log10(yMax);
  const sx = (ai: number) =>
    M.l + ((Math.log10(ai) - lx0) / (lx1 - lx0)) * (W - M.l - M.r);
  const sy = (v: number) =>
    H - M.b - ((Math.log10(v) - ly0) / (ly1 - ly0)) * (H - M.t - M.b);

  const roof = (peak: number) => {
    const ridge = peak / p.hbm_bw;
    return `M${sx(AI_MIN)},${sy(AI_MIN * p.hbm_bw)} L${sx(ridge)},${sy(peak)} L${sx(AI_MAX)},${sy(peak)}`;
  };

  // the sweep's point, sliding in log space towards the next tile size
  const cur = sweep[st.step]!;
  const nxt = sweep[Math.min(st.step + 1, sweep.length - 1)]!;
  const lerpLog = (a: number, b: number) =>
    10 ** (Math.log10(a) + (Math.log10(b) - Math.log10(a)) * st.frac);
  const ptAi = lerpLog(cur.ai, nxt.ai);
  const ptPerf = lerpLog(cur.perf, nxt.perf);

  const free = rooflinePoint(p, 2 ** log2ai, engine);
  const ridgeFp32 = d.ridge_fp32;
  const xTicks = [1 / 16, 1 / 4, 1, 4, 16, 64, 256, 1024];
  const yTicks = [1e11, 1e12, 1e13, 1e14, 1e15].filter(
    (v) => v >= yMin && v <= yMax,
  );

  const visual = (
    <div className="mx-auto max-w-xl">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`Roofline of the ${p.name}: FP32 roof ${fmtFlops(p.peak_fp32)}, tensor roof ${fmtFlops(p.peak_tensor)}, HBM slope ${trim(p.hbm_bw / 1e12)} TB/s.`}
      >
        <defs>
          <clipPath id={clipId}>
            <rect
              x={M.l}
              y={M.t}
              width={W - M.l - M.r}
              height={H - M.t - M.b}
            />
          </clipPath>
        </defs>
        {/* memory-bound region for FP32 */}
        <rect
          x={M.l}
          y={M.t}
          width={sx(ridgeFp32) - M.l}
          height={H - M.t - M.b}
          fill={LEVEL_COLOUR.hbm}
          opacity={hover === "bw" || hover === "ai" ? 0.22 : 0.1}
        />
        {xTicks.map((t) => (
          <g key={t}>
            <line
              x1={sx(t)}
              x2={sx(t)}
              y1={M.t}
              y2={H - M.b}
              stroke={MUTED.light}
              strokeOpacity={0.3}
            />
            <text
              x={sx(t)}
              y={H - M.b + 13}
              textAnchor="middle"
              className="fill-neutral-600 font-mono text-[10px] dark:fill-neutral-400"
            >
              {t < 1 ? `1/${1 / t}` : t}
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
              x={M.l - 4}
              y={sy(t) + 3}
              textAnchor="end"
              className="fill-neutral-600 font-mono text-[10px] dark:fill-neutral-400"
            >
              {fmtFlops(t).replace("FLOP/s", "")}
            </text>
          </g>
        ))}
        <text
          x={(W + M.l) / 2}
          y={H - 4}
          textAnchor="middle"
          className="fill-neutral-700 text-[11px] dark:fill-neutral-300"
        >
          arithmetic intensity (flop per HBM byte, log)
        </text>
        <text
          x={10}
          y={(H - M.b) / 2}
          transform={`rotate(-90 10 ${(H - M.b) / 2})`}
          textAnchor="middle"
          className="fill-neutral-700 text-[11px] dark:fill-neutral-300"
        >
          FLOP/s (log)
        </text>
        <g clipPath={`url(#${clipId})`}>
          <path
            d={roof(p.peak_tensor)}
            fill="none"
            stroke={OKABE_ITO.vermillion}
            strokeWidth={2}
            strokeDasharray="5 4"
          />
          <path
            d={roof(p.peak_fp32)}
            fill="none"
            stroke={hover === "peak" ? STATE_COLOUR.active : LEVEL_COLOUR.reg}
            strokeWidth={hover === "peak" ? 3.5 : 2.5}
          />
          {/* the slope's colour is HBM's */}
          <path
            d={`M${sx(AI_MIN)},${sy(AI_MIN * p.hbm_bw)} L${sx(ridgeFp32)},${sy(p.peak_fp32)}`}
            fill="none"
            stroke={LEVEL_COLOUR.hbm}
            strokeWidth={hover === "bw" ? 4 : 2.5}
          />
          {/* trail of the tile sizes already visited */}
          {sweep.slice(0, st.step + 1).map((s) => (
            <g key={s.tile}>
              <circle cx={sx(s.ai)} cy={sy(s.perf)} r={3} fill={MUTED.light} />
              <text
                x={sx(s.ai) + 4}
                y={sy(s.perf) + 12}
                className="fill-neutral-500 font-mono text-[8px] dark:fill-neutral-400"
              >
                {s.tile}
              </text>
            </g>
          ))}
          <circle
            data-testid="sweep-point"
            cx={sx(ptAi)}
            cy={sy(ptPerf)}
            r={6}
            fill={STATE_COLOUR.active}
            stroke="white"
            strokeWidth={1.5}
          />
          {/* the reader's own point */}
          <rect
            x={sx(free.ai) - 5}
            y={sy(free.perf) - 5}
            width={10}
            height={10}
            transform={`rotate(45 ${sx(free.ai)} ${sy(free.perf)})`}
            fill="none"
            stroke={STATE_COLOUR.active}
            strokeWidth={2}
          />
        </g>
        <text
          x={sx(ridgeFp32) + 4}
          y={sy(p.peak_fp32) + 12}
          className="fill-neutral-700 font-mono text-[9px] dark:fill-neutral-300"
        >
          ridge {trim(ridgeFp32)}
        </text>
      </svg>
      <ul className="mt-1 flex flex-wrap justify-center gap-x-4 gap-y-1 text-[0.7rem] text-neutral-700 dark:text-neutral-300">
        <li className="flex items-center gap-1">
          <svg aria-hidden width="18" height="8">
            <line
              x1="1"
              x2="17"
              y1="4"
              y2="4"
              stroke={LEVEL_COLOUR.reg}
              strokeWidth="2.5"
            />
          </svg>
          FP32 roof
        </li>
        <li className="flex items-center gap-1">
          <svg aria-hidden width="18" height="8">
            <line
              x1="1"
              x2="17"
              y1="4"
              y2="4"
              stroke={OKABE_ITO.vermillion}
              strokeWidth="2"
              strokeDasharray="4 3"
            />
          </svg>
          BF16 tensor roof
        </li>
        <li className="flex items-center gap-1">
          <svg aria-hidden width="18" height="8">
            <line
              x1="1"
              x2="17"
              y1="4"
              y2="4"
              stroke={LEVEL_COLOUR.hbm}
              strokeWidth="2.5"
            />
          </svg>
          HBM slope
        </li>
        <li className="flex items-center gap-1">
          <svg aria-hidden width="10" height="10">
            <circle cx="5" cy="5" r="4" fill={STATE_COLOUR.active} />
          </svg>
          GEMM tile sweep
        </li>
        <li className="flex items-center gap-1">
          <svg aria-hidden width="10" height="10">
            <rect
              x="2"
              y="2"
              width="6"
              height="6"
              transform="rotate(45 5 5)"
              fill="none"
              stroke={STATE_COLOUR.active}
              strokeWidth="1.5"
            />
          </svg>
          your intensity
        </li>
      </ul>
    </div>
  );

  const hl = free.bound === "memory" ? "bw ai" : "peak";

  return (
    <AnimationPanel
      testId="roofline-widget"
      title="Sliding along the roofline"
      summary={`A 4096³ FP32 GEMM on the ${p.name}, tile size growing from 1 to 128: each step is one run of the model's traffic count.`}
      stepper={st}
      stepLabel="tile step"
      caption={sweepCaption(cur, ridgeFp32)}
      visual={visual}
      equation={children}
      hl={hover ?? hl}
      onEquationHover={setHover}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Your intensity" value={fmtAi(free.ai)} />
          <Stat
            label="Attainable"
            value={fmtFlops(free.perf)}
            hint={`${pct(free.frac_peak)} of the ${engine === "fp32" ? "FP32" : "tensor"} peak`}
          />
          <Stat label="Ridge" value={fmtAi(free.ridge)} />
          <Stat
            label="Bound"
            value={free.bound === "memory" ? "memory (HBM)" : "compute"}
          />
        </div>
      }
      params={
        <>
          <Segmented
            label="GPU"
            value={pid}
            options={[
              { value: "a100", label: "A100" },
              { value: "h100", label: "H100" },
            ]}
            onChange={setPid}
          />
          <Segmented
            label="Your point's engine"
            value={engine}
            options={[
              { value: "fp32", label: "FP32 cores" },
              { value: "tensor", label: "BF16 tensor" },
            ]}
            onChange={setEngine}
          />
          <Slider
            label="Your arithmetic intensity"
            value={log2ai}
            min={-4}
            max={10}
            step={0.25}
            onChange={setLog2ai}
            format={(v) => fmtAi(2 ** v)}
          />
        </>
      }
    />
  );
}
