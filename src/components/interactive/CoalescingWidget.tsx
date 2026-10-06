"use client";

/**
 * Chapter 4's hero: each lane of a warp computes an address; the memory
 * system answers the warp in 32-byte sectors. Lane by lane, the sectors a
 * lane touches light up: new ones (a new transaction) in the active colour,
 * ones an earlier lane already brought in muted. Inside each sector, the
 * 4-byte words some lane actually asked for are filled; the empty rest is
 * bandwidth paid for and thrown away. Driven by `coalesceSteps`.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Slider, Stat } from "@/components/ui/Controls";
import { fmtBytes, pct } from "@/lib/format";
import { coalesceCaption, laneRuns } from "@/lib/gpu/captions";
import { coalesceSteps } from "@/lib/gpu/model";
import { LEVEL_COLOUR, STATE_COLOUR } from "@/lib/viz/palette";

type Eb = "1" | "2" | "4" | "8" | "16";

export default function CoalescingWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [eb, setEb] = useState<Eb>("4");
  const [stride, setStride] = useState(1);
  const [offset, setOffset] = useState(0);
  const [hover, setHover] = useState<string | null>(null);
  const e = Number(eb);

  const r = useMemo(
    () => coalesceSteps(e, stride, offset),
    [e, stride, offset],
  );
  const st = useStepper(32, {
    stepMs: 380,
    resetKey: `${e}-${stride}-${offset}`,
  });
  const lane = st.step;
  const cur = r.steps[lane]!;

  // which sectors are fetched by now, which 4-byte words are used, and
  // which lines (128 bytes = 4 sectors) to draw
  const view = useMemo(() => {
    const fetched = new Set<number>();
    const words = new Set<number>();
    const lanesOf = new Map<number, number>();
    for (const s of r.steps.slice(0, lane + 1)) {
      for (const x of s.sectors) {
        fetched.add(x);
        lanesOf.set(x, ((lanesOf.get(x) ?? 0) | (1 << s.lane)) >>> 0);
      }
      for (let b = s.addr; b < s.addr + e; b += 1) words.add(Math.floor(b / 4));
    }
    const allLines = [
      ...new Set(
        r.steps.flatMap((s) => s.sectors.map((x) => Math.floor(x / 4))),
      ),
    ].sort((a, b) => a - b);
    return { fetched, words, allLines, lanesOf };
  }, [r, lane, e]);
  const now = new Set(cur.sectors);
  const fresh = new Set(cur.new);

  const visual = (
    <div className="min-w-0">
      <p className="text-[0.7rem] font-medium uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
        The warp&apos;s 32 lanes
      </p>
      <div
        className="mt-1 grid grid-cols-[repeat(32,minmax(0,1fr))] gap-px"
        aria-hidden
      >
        {Array.from({ length: 32 }, (_, i) => (
          <div
            key={i}
            className="h-4 rounded-sm"
            style={{
              background:
                i === lane
                  ? STATE_COLOUR.active
                  : i < lane
                    ? "#a3a3a3"
                    : "transparent",
              outline: "1px solid #a3a3a3",
            }}
            title={`lane ${i}`}
          />
        ))}
      </div>
      <p className="mt-3 text-[0.7rem] font-medium uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
        Memory: 128-byte lines of four 32-byte sectors (only the lines this warp
        touches)
      </p>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-2" data-testid="lines">
        {view.allLines.map((ln, idx) => {
          const prev = view.allLines[idx - 1];
          const gap = prev === undefined ? 0 : ln - prev - 1;
          return (
            <div key={ln} className="flex items-center gap-2">
              {gap > 0 && (
                <span className="rounded bg-neutral-200 px-1 font-mono text-[0.6rem] text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300">
                  +{gap} line{gap === 1 ? "" : "s"}
                </span>
              )}
              <div>
                <div className="font-mono text-[0.6rem] text-neutral-500 dark:text-neutral-400">
                  byte {ln * 128}
                </div>
                <div className="flex gap-0.5">
                  {[0, 1, 2, 3].map((q) => {
                    const sec = ln * 4 + q;
                    const got = view.fetched.has(sec);
                    const isNow = now.has(sec);
                    const who = view.lanesOf.get(sec);
                    return (
                      <div key={q} className="flex flex-col items-center">
                        <div
                          data-sector={sec}
                          data-state={
                            isNow
                              ? fresh.has(sec)
                                ? "new"
                                : "hit"
                              : got
                                ? "fetched"
                                : "untouched"
                          }
                          className="grid grid-cols-8 gap-px rounded-sm p-px"
                          style={{
                            outline: isNow
                              ? `2px solid ${STATE_COLOUR.active}`
                              : got
                                ? `1px solid ${LEVEL_COLOUR.hbm}`
                                : "1px dashed #a3a3a3",
                            background: got
                              ? `${LEVEL_COLOUR.hbm}33`
                              : "transparent",
                          }}
                        >
                          {Array.from({ length: 8 }, (_, w) => {
                            const word = sec * 8 + w;
                            const used = view.words.has(word);
                            return (
                              <span
                                key={w}
                                className="block h-4 w-1.5 rounded-[1px] sm:h-5 sm:w-3"
                                style={{
                                  background: used
                                    ? isNow && fresh.has(sec)
                                      ? STATE_COLOUR.active
                                      : LEVEL_COLOUR.hbm
                                    : "transparent",
                                }}
                              />
                            );
                          })}
                        </div>
                        <span className="mt-0.5 h-3 font-mono text-[0.6rem] leading-3 text-neutral-600 dark:text-neutral-400">
                          {who ? `L${laneRuns(who)}` : ""}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          );
        })}
      </div>
      <p className="mt-2 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Outlined blue: the sectors of the current lane; under each sector, the
        lanes it serves. Filled: words some lane asked for. A fetched
        sector&apos;s empty slots are wasted bandwidth.
      </p>
    </div>
  );

  return (
    <AnimationPanel
      testId="coalescing-widget"
      title="From thread addresses to memory transactions"
      summary="Lane i reads at offset + i × stride × element size. The model counts the 32-byte sectors the warp's request touches."
      stepper={st}
      stepLabel="lane"
      caption={coalesceCaption(r, lane, e)}
      visual={visual}
      equation={children}
      hl={hover ?? "sectors"}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat
            label="Sectors"
            value={String(r.sectors)}
            hint={`${r.lines} line${r.lines === 1 ? "" : "s"}`}
          />
          <Stat label="Requested" value={fmtBytes(r.requested)} />
          <Stat label="Fetched" value={fmtBytes(r.fetched)} />
          <Stat
            label="Efficiency"
            value={pct(Math.min(1, r.efficiency), 1)}
            hint={
              r.efficiency > 1 ? "lanes share words (broadcast)" : undefined
            }
          />
        </div>
      }
      params={
        <>
          <Segmented
            label="Element size (bytes)"
            value={eb}
            options={(["1", "2", "4", "8", "16"] as const).map((v) => ({
              value: v,
              label: v,
            }))}
            onChange={setEb}
          />
          <Slider
            label="Stride (elements)"
            value={stride}
            min={0}
            max={33}
            onChange={setStride}
          />
          <Slider
            label="Start offset (bytes)"
            value={offset}
            min={0}
            max={128}
            step={4}
            onChange={setOffset}
          />
        </>
      }
    />
  );
}
