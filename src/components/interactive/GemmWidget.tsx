"use client";

/**
 * Chapter 7's hero: C = A·B on a 16 × 16 × 16 problem small enough to draw
 * every element, computed four ways (naive, shared-memory tiles, register
 * blocking, tensor cores). Blocks of C are worked on in row-major order;
 * for each, the k-loop marches a slice of A and a slice of B in. Counters
 * show the bytes from global memory and through shared memory so far, and
 * the arithmetic intensity they give. Underneath, the same four variants
 * on a 4096³ GEMM, from `gemmVariant`. Every frame comes from `gemmMarch`.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtAi, fmtBytes, fmtFlops, fmtTime, pct, trim } from "@/lib/format";
import { marchCaption } from "@/lib/gpu/captions";
import {
  GEMM_MARCH,
  GEMM_ORDER,
  MARCH_BK,
  MARCH_SIZE,
  gemmMarch,
  gemmVariant,
  preset,
  type GemmVariantId,
  type PresetId,
} from "@/lib/gpu/model";
import {
  LEVEL_COLOUR,
  LEVEL_NAME,
  MUTED,
  STATE_COLOUR,
} from "@/lib/viz/palette";

const N = MARCH_SIZE;
const C = 8;
const SIDE = N * C;
const GAP = 16;
const LEFT = 14;
const TOP = 14;
const W = LEFT + SIDE + GAP + SIDE + 4;
const H = TOP + SIDE + GAP + SIDE + 4;
/** origins of the three matrices */
const AX = LEFT;
const AY = TOP + SIDE + GAP;
const BX = LEFT + SIDE + GAP;
const BY = TOP;
const CX = BX;
const CY = AY;

const SHORT: Record<GemmVariantId, string> = {
  naive: "naive",
  smem: "shared-memory tiles",
  regs: "register blocking",
  tensor: "tensor cores",
};

export default function GemmWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [vid, setVid] = useState<GemmVariantId>("smem");
  const [pid, setPid] = useState<PresetId>("a100");
  const [hover, setHover] = useState<string | null>(null);
  const font = useSvgFont(W);
  const fs = font.fs;

  const spec = GEMM_MARCH[vid];
  const march = useMemo(() => gemmMarch(vid), [vid]);
  const p = useMemo(() => preset(pid), [pid]);
  const variants = useMemo(() => GEMM_ORDER.map((v) => gemmVariant(p, v)), [p]);
  const st = useStepper(march.length, { stepMs: 450, resetKey: vid });
  const s = march[st.step]!;
  const kts = N / MARCH_BK;
  const blocksAcross = N / spec.bn;
  const cur = s.block ? s.block[0] * blocksAcross + s.block[1] : -1;
  const staged = spec.smem;
  const sliceColour = staged ? LEVEL_COLOUR.smem : LEVEL_COLOUR.hbm;

  const cells = (
    x0: number,
    y0: number,
    fill: (r: number, c: number) => { f: string; o: number } | null,
  ) => {
    const out: JSX.Element[] = [];
    for (let r = 0; r < N; r++)
      for (let c = 0; c < N; c++) {
        const v = fill(r, c);
        if (v)
          out.push(
            <rect
              key={`${r}-${c}`}
              x={x0 + c * C + 0.5}
              y={y0 + r * C + 0.5}
              width={C - 1}
              height={C - 1}
              fill={v.f}
              opacity={v.o}
            />,
          );
      }
    return out;
  };

  const inBlock = (r: number, c: number, b: number) =>
    Math.floor(r / spec.bm) * blocksAcross + Math.floor(c / spec.bn) === b;

  const frame = (x0: number, y0: number, label: string, dims: string) => (
    <g>
      <rect
        x={x0}
        y={y0}
        width={SIDE}
        height={SIDE}
        fill="none"
        stroke={MUTED.light}
        strokeWidth={0.75}
      />
      <text
        x={x0}
        y={y0 - 4}
        style={{ fontSize: fs(8) }}
        className="fill-neutral-600 font-mono dark:fill-neutral-400"
      >
        {label} {dims}
      </text>
    </g>
  );

  const kLo = s.kt === null ? -1 : s.kt * MARCH_BK;
  const rowLo = s.block ? s.block[0] * spec.bm : -1;
  const colLo = s.block ? s.block[1] * spec.bn : -1;

  const visual = (
    <div className="mx-auto grid max-w-2xl gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="min-w-0">
        <svg
          ref={font.ref}
          viewBox={`0 0 ${W} ${H}`}
          className="h-auto w-full"
          role="img"
          aria-label={`C = A times B, 16 by 16, computed with ${SHORT[vid]}: block ${s.block ? `(${s.block[0]}, ${s.block[1]})` : "none yet"}.`}
        >
          {frame(AX, AY, "A", "16×16")}
          {frame(BX, BY, "B", "16×16")}
          {frame(CX, CY, "C", "16×16")}
          {/* A: the rows of this block; the k-slice now coming in */}
          {cells(AX, AY, (r, c) => {
            if (r < rowLo || r >= rowLo + spec.bm) return null;
            if (c >= kLo && c < kLo + MARCH_BK) return { f: sliceColour, o: 1 };
            if (c < kLo) return { f: MUTED.light, o: 0.45 };
            return null;
          })}
          {/* B: the columns of this block */}
          {cells(BX, BY, (r, c) => {
            if (c < colLo || c >= colLo + spec.bn) return null;
            if (r >= kLo && r < kLo + MARCH_BK) return { f: sliceColour, o: 1 };
            if (r < kLo) return { f: MUTED.light, o: 0.45 };
            return null;
          })}
          {/* C: finished blocks muted, the current one filling up */}
          {cells(CX, CY, (r, c) => {
            const b =
              Math.floor(r / spec.bm) * blocksAcross + Math.floor(c / spec.bn);
            if (b < cur) return { f: MUTED.light, o: 0.7 };
            if (b === cur && inBlock(r, c, cur))
              return {
                f: STATE_COLOUR.active,
                o: 0.25 + (0.75 * ((s.kt as number) + 1)) / kts,
              };
            return null;
          })}
          {/* what one thread (or, for tensor cores, one warp) owns */}
          {s.block &&
            Array.from(
              { length: (spec.bm / spec.tm) * (spec.bn / spec.tn) },
              (_, t) => {
                const tr = Math.floor(t / (spec.bn / spec.tn));
                const tc = t % (spec.bn / spec.tn);
                return (
                  <rect
                    key={t}
                    x={CX + (colLo + tc * spec.tn) * C}
                    y={CY + (rowLo + tr * spec.tm) * C}
                    width={spec.tn * C}
                    height={spec.tm * C}
                    fill="none"
                    stroke={STATE_COLOUR.active}
                    strokeWidth={vid === "tensor" ? 2 : 0.6}
                  />
                );
              },
            )}
          <text
            x={AX}
            y={TOP + SIDE - 2}
            style={{ fontSize: fs(8) }}
            className="fill-neutral-700 font-mono dark:fill-neutral-300"
          >
            {vid === "tensor"
              ? "1 warp: one MMA"
              : `${spec.tm}×${spec.tn} per thread`}
          </text>
          <text
            x={AX}
            y={TOP + SIDE - 2 - fs(8) * 1.3}
            style={{ fontSize: fs(8) }}
            className="fill-neutral-700 font-mono dark:fill-neutral-300"
          >
            block {spec.bm}×{spec.bn}, k by {MARCH_BK}
          </text>
        </svg>
        <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
          {staged
            ? "Green: the k-slices staged in shared memory now."
            : "Purple: the values each thread fetches from global memory now."}{" "}
          Blue: the block of C being computed (darker as k advances); grey:
          done.
        </p>
      </div>
      <IntensityBars
        variants={variants.map((v) => ({
          id: v.id,
          ai: v.ai.l2,
          smem: v.ai.smem,
        }))}
        current={vid}
      />
    </div>
  );

  const v = variants[GEMM_ORDER.indexOf(vid)]!;
  return (
    <AnimationPanel
      testId="gemm-widget"
      title="GEMM, tile by tile"
      summary={`${SHORT[vid][0]!.toUpperCase()}${SHORT[vid].slice(1)} on a 16 × 16 × 16 problem; the counters are the model's bytes and flops so far.`}
      stepper={st}
      stepLabel="k-tile"
      caption={marchCaption(vid, spec, s, N, MARCH_BK)}
      visual={visual}
      equation={children}
      hl={hover ?? (s.kt === 0 ? "tile" : "k")}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="space-y-2">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat label="Global bytes so far" value={fmtBytes(s.global)} />
            <Stat
              label="Shared-memory bytes"
              value={staged ? fmtBytes(s.smem) : "none"}
            />
            <Stat
              label="Flops so far"
              value={s.flops.toLocaleString("en-GB")}
            />
            <Stat
              label="Intensity so far"
              value={s.ai === null ? "–" : fmtAi(s.ai)}
            />
          </div>
          <p className="text-xs text-neutral-600 dark:text-neutral-400">
            The same kernel on a 4096³ GEMM on the {p.name} (model):
          </p>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat
              label="Bound by"
              value={v.bound === "compute" ? "the ALUs" : LEVEL_NAME[v.bound]}
            />
            <Stat label="Time" value={fmtTime(v.total)} />
            <Stat
              label="Achieved"
              value={fmtFlops(v.achieved)}
              hint={`${pct(v.frac_peak)} of the ${vid === "tensor" ? "BF16 tensor" : "FP32"} peak`}
            />
            <Stat
              label="Intensity (L2, smem)"
              value={`${trim(v.ai.l2 as number)}, ${v.ai.smem === null ? "–" : trim(v.ai.smem)}`}
              hint="flop per byte"
            />
          </div>
        </div>
      }
      params={
        <>
          <Segmented
            label="Kernel"
            value={vid}
            options={GEMM_ORDER.map((g) => ({ value: g, label: SHORT[g] }))}
            onChange={setVid}
          />
          <Segmented
            label="GPU (for the 4096³ figures)"
            value={pid}
            options={[
              { value: "a100", label: "A100" },
              { value: "h100", label: "H100" },
            ]}
            onChange={setPid}
          />
        </>
      }
    />
  );
}

const BW = 288;
const ROW = 34;

/** Arithmetic intensity at L2 of the four variants, on a log scale. */
function IntensityBars({
  variants,
  current,
}: {
  variants: { id: GemmVariantId; ai: number | null; smem: number | null }[];
  current: GemmVariantId;
}): JSX.Element {
  const font = useSvgFont(BW);
  const fs = font.fs;
  const LBL = 4;
  const BAR0 = 4;
  const BARW = BW - 64;
  const lo = Math.log2(1 / 8);
  const hi = Math.log2(128);
  const sx = (v: number) => BAR0 + ((Math.log2(v) - lo) / (hi - lo)) * BARW;
  const H2 = 18 + variants.length * ROW + 14;
  return (
    <div className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${BW} ${H2}`}
        className="h-auto w-full"
        role="img"
        aria-label={`Arithmetic intensity at L2 of a 4096-cubed GEMM: ${variants.map((v) => `${SHORT[v.id]} ${trim(v.ai as number)}`).join(", ")} flop per byte.`}
      >
        <text
          x={LBL}
          y={12}
          style={{ fontSize: fs(9) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          4096³ GEMM: flop per byte from L2 (log)
        </text>
        {variants.map((v, i) => {
          const y = 20 + i * ROW;
          const on = v.id === current;
          return (
            <g key={v.id} data-variant={v.id} data-current={on || undefined}>
              <text
                x={LBL}
                y={y + 9}
                style={{ fontSize: fs(9) }}
                className={
                  on
                    ? "fill-neutral-950 font-semibold dark:fill-white"
                    : "fill-neutral-600 dark:fill-neutral-400"
                }
              >
                {SHORT[v.id]}
              </text>
              <rect
                x={BAR0}
                y={y + 13}
                width={Math.max(2, sx(v.ai as number) - BAR0)}
                height={12}
                fill={on ? STATE_COLOUR.active : LEVEL_COLOUR.l2}
                opacity={on ? 1 : 0.6}
              />
              <text
                x={sx(v.ai as number) + 4}
                y={y + 23}
                style={{ fontSize: fs(9) }}
                className="fill-neutral-800 font-mono dark:fill-neutral-200"
              >
                {trim(v.ai as number)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
