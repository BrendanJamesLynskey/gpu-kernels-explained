"use client";

/**
 * Chapter 1's hero: data flowing HBM → L2 → shared memory → registers →
 * ALUs while a kernel runs. Each pipe's width is its bandwidth, drawn to
 * scale (linear, within the chosen GPU); the packets in a pipe move at the
 * rate that level is actually used, so the bottleneck's packets race and
 * the rest crawl. Every frame is computed from `flowSteps` and
 * `kernelTime` (the model, checked against the Python reference).
 */
import { useId, useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import {
  fmtBytes,
  fmtFlops,
  fmtKiB,
  fmtRate,
  fmtTime,
  pct,
} from "@/lib/format";
import { flowCaption } from "@/lib/gpu/captions";
import {
  KERNELS,
  derived,
  flowSteps,
  kernelTime,
  preset,
  type KernelId,
  type Level,
  type PresetId,
} from "@/lib/gpu/model";
import { LEVEL_COLOUR, LEVEL_NAME, STATE_COLOUR } from "@/lib/viz/palette";

import { Hatch } from "./Hatch";

const STEPS = 60;
const W = 360;
const BOX_H = 52;
const PIPE_H = 40;
const BOX_W = 300;
const X0 = (W - BOX_W) / 2;
const MAX_PIPE = 200;
/** Packet cycles along a fully used pipe over one run. */
const CYCLES = 14;
const ORDER: ("alu" | Level)[] = ["alu", "reg", "smem", "l2", "hbm"];

function frac(x: number): number {
  return x - Math.floor(x);
}

export default function HierarchyWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [pid, setPid] = useState<PresetId>("a100");
  const [kid, setKid] = useState<KernelId>("vecadd");
  const [hover, setHover] = useState<string | null>(null);
  const hatchId = useId().replace(/:/g, "");
  const font = useSvgFont(W);
  const fs = font.fs;

  const p = useMemo(() => preset(pid), [pid]);
  const d = useMemo(() => derived(p), [p]);
  const kernel = KERNELS[kid]!;
  const kt = useMemo(() => kernelTime(p, kernel), [p, kernel]);
  const steps = useMemo(() => flowSteps(p, kernel, STEPS), [p, kernel]);
  const st = useStepper(steps.length, {
    stepMs: 110,
    smooth: true,
    resetKey: `${pid}-${kid}`,
  });
  const s = steps[st.step]!;
  const next = steps[Math.min(st.step + 1, STEPS)]!;
  // in-between frame: interpolate the model's states (both are linear in time)
  const lerp = (a: number, b: number) => a + (b - a) * st.frac;

  const capacity: Record<Level, number> = {
    reg: d.regs_bytes_per_sm * p.sms,
    smem: p.smem_per_sm * p.sms,
    l2: p.l2_bytes,
    hbm: p.hbm_bytes,
  };
  const highlight = hover ?? kt.bound;

  const y = (i: number) => i * (BOX_H + PIPE_H) + 4;
  const H = y(ORDER.length - 1) + BOX_H + 6;

  const visual = (
    <div className="mx-auto max-w-xl">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`Memory hierarchy of the ${p.name} running ${kernel.label}: pipe widths are bandwidths to scale; the bottleneck is ${kt.bound === "compute" ? "the ALUs" : LEVEL_NAME[kt.bound]}.`}
      >
        <defs>
          <Hatch id={hatchId} />
        </defs>
        {ORDER.map((key, i) => {
          const top = y(i);
          const isAlu = key === "alu";
          const colour = isAlu ? STATE_COLOUR.compute : LEVEL_COLOUR[key];
          const util = isAlu ? kt.compute_util : kt.util[key];
          const name = isAlu ? "FP32 ALUs" : LEVEL_NAME[key];
          const right = isAlu
            ? fmtFlops(p.peak_fp32)
            : key === "hbm"
              ? `${fmtBytes(capacity[key])}`
              : key === "l2"
                ? fmtKiB(capacity[key])
                : `${fmtKiB(capacity[key] / p.sms)} × ${p.sms} SMs`;
          const hl = highlight === (isAlu ? "compute" : key);
          const barW = BOX_W - 16;
          return (
            <g key={key} data-level={key}>
              <rect
                x={X0}
                y={top}
                width={BOX_W}
                height={BOX_H}
                rx={6}
                className="fill-white dark:fill-neutral-950"
                stroke={hl ? STATE_COLOUR.active : colour}
                strokeWidth={hl ? 3 : 1.5}
              />
              <rect
                x={X0}
                y={top}
                width={6}
                height={BOX_H}
                rx={3}
                fill={colour}
              />
              <text
                x={X0 + 14}
                y={top + 18}
                style={{ fontSize: fs(12) }}
                className="fill-neutral-900 font-semibold dark:fill-neutral-100"
              >
                {name}
              </text>
              <text
                x={X0 + BOX_W - 8}
                y={top + 18}
                textAnchor="end"
                style={{ fontSize: fs(10) }}
                className="fill-neutral-600 font-mono dark:fill-neutral-400"
              >
                {right}
              </text>
              {/* busy fraction: solid = working, hatched = waiting */}
              <rect
                x={X0 + 8}
                y={top + 28}
                width={barW}
                height={10}
                fill={`url(#${hatchId})`}
                className="opacity-60"
              />
              <rect
                x={X0 + 8}
                y={top + 28}
                width={barW * util}
                height={10}
                fill={colour}
              />
              <text
                x={X0 + 8}
                y={top + 49}
                style={{ fontSize: fs(10) }}
                className="fill-neutral-700 font-mono dark:fill-neutral-300"
              >
                {/* phones: the same facts, shorter (the bar shows "busy") */}
                {isAlu
                  ? font.narrow
                    ? `${pct(util)} · ${lerp(s.flops, next.flops).toExponential(1)} / ${kt.flops.toExponential(1)} FLOP`
                    : `busy ${pct(util)} · ${lerp(s.flops, next.flops).toExponential(2)} of ${kt.flops.toExponential(2)} FLOPs done`
                  : font.narrow
                    ? `${pct(util)} · ${fmtBytes(lerp(s.moved[key], next.moved[key]))} / ${fmtBytes(kt.bytes[key])}`
                    : `busy ${pct(util)} · moved ${fmtBytes(lerp(s.moved[key], next.moved[key]))} of ${fmtBytes(kt.bytes[key])}`}
              </text>
            </g>
          );
        })}
        {(["reg", "smem", "l2", "hbm"] as Level[]).map((lv, j) => {
          const top = y(j) + BOX_H;
          const w = Math.max(2, (MAX_PIPE * d.bw[lv]) / d.bw.reg);
          const x = W / 2 - w / 2 - 40;
          const pos = frac(
            (CYCLES * lerp(s.moved[lv], next.moved[lv])) /
              (d.bw[lv] * kt.total),
          );
          const packets = [];
          for (let k = -1; k < 4; k++) {
            const py = top + PIPE_H - ((k + pos) * PIPE_H) / 3;
            if (py > top - 1 && py < top + PIPE_H - 4)
              packets.push(
                <rect
                  key={k}
                  x={x}
                  y={py}
                  width={w}
                  height={4}
                  fill={LEVEL_COLOUR[lv]}
                />,
              );
          }
          return (
            <g key={`pipe-${lv}`} data-pipe={lv}>
              <rect
                x={x}
                y={top}
                width={w}
                height={PIPE_H}
                fill={LEVEL_COLOUR[lv]}
                opacity={0.22}
              />
              {kt.bytes[lv] > 0 && packets}
              <text
                x={x + w + 8}
                y={top + PIPE_H / 2 + 4}
                style={{ fontSize: fs(10.5) }}
                className="fill-neutral-800 font-mono dark:fill-neutral-200"
              >
                {fmtRate(d.bw[lv])}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="mt-1 text-center text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Pipe width = bandwidth, to scale. Solid bar = busy, hatched = waiting on
        the bottleneck.
      </p>
    </div>
  );

  const stats = (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      <Stat label="Kernel time" value={fmtTime(kt.total)} />
      <Stat
        label="Bottleneck"
        value={kt.bound === "compute" ? "FP32 ALUs" : LEVEL_NAME[kt.bound]}
      />
      <Stat
        label="Achieved"
        value={fmtFlops(kt.achieved)}
        hint={`${pct(kt.achieved / p.peak_fp32)} of peak`}
      />
      <Stat label="FLOPs" value={kt.flops.toExponential(3)} />
    </div>
  );

  return (
    <AnimationPanel
      testId="hierarchy-widget"
      title="Data through the memory hierarchy"
      summary={`${kernel.label} on the ${p.name}. The model computes the bytes each level moves and the time each would take on its own; the slowest sets the pace.`}
      stepper={st}
      stepLabel="time step"
      caption={flowCaption(kt, s, STEPS)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={hover ?? kt.bound}
      onEquationHover={setHover}
      countFrom={0}
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
            label="Kernel"
            value={kid}
            options={[
              { value: "vecadd", label: "vector add" },
              { value: "gemm16", label: "GEMM 16×16" },
              { value: "gemm128", label: "GEMM 128×128" },
            ]}
            onChange={setKid}
          />
        </>
      }
    />
  );
}
