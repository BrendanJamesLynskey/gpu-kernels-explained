"use client";

/**
 * The frame every animation sits in: the visual, then the transport
 * controls (reset, step back, play/pause, step forward, speed, scrub bar),
 * then the live caption, then the parameters. The panel takes keyboard
 * focus: Space plays or pauses, the arrow keys step, Home resets and End
 * jumps to the last step. The caption is an aria-live region, so a
 * screen-reader user hears each step as it happens.
 *
 * Styled like the companion sites' WidgetFrame (same border, background,
 * padding), with 44 px touch targets on the controls.
 */
import { useId, type ReactNode } from "react";

import { SPEEDS, keyAction, type Speed } from "@/lib/anim/clock";

import type { Stepper } from "./useStepper";

const BTN =
  "focus-ring inline-flex h-11 min-w-11 items-center justify-center rounded border border-neutral-300 bg-white px-2 text-sm text-neutral-800 hover:bg-neutral-100 disabled:opacity-40 dark:border-neutral-700 dark:bg-neutral-950 dark:text-neutral-200 dark:hover:bg-neutral-800";

export function AnimationPanel({
  title,
  summary,
  stepper,
  stepLabel,
  caption,
  visual,
  equation,
  params,
  stats,
  testId,
  hl,
  onEquationHover,
  countFrom = 1,
}: {
  title: string;
  /** One line under the title: what the animation shows and what drives it. */
  summary: ReactNode;
  stepper: Stepper;
  /** e.g. "cycle", "lane", "block": the scrub bar's unit. */
  stepLabel: string;
  caption: string;
  visual: ReactNode;
  /** Server-rendered KaTeX (children of the widget), with highlightable terms. */
  equation?: ReactNode;
  params?: ReactNode;
  stats?: ReactNode;
  testId: string;
  /** The equation term to highlight now (matches `hl-<key>` classes). */
  hl?: string;
  onEquationHover?: (key: string | null) => void;
  /** Number the steps from 1 (events) or 0 (time, blocks placed). */
  countFrom?: 0 | 1;
}): JSX.Element {
  const scrubId = useId();
  const s = stepper;
  const onKey = (e: React.KeyboardEvent) => {
    // leave keys alone inside form controls (sliders, selects)
    const t = e.target as HTMLElement;
    if (t !== e.currentTarget && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName))
      return;
    const a = keyAction(e.key);
    if (!a) return;
    e.preventDefault();
    if (a === "toggle") s.toggle();
    else if (a === "back") s.back();
    else if (a === "forward") s.forward();
    else if (a === "reset") s.reset();
    else if (a === "end") s.setStep(s.n - 1);
  };
  return (
    <figure
      ref={s.ref}
      data-testid={testId}
      data-step={s.step}
      data-playing={s.playing ? "true" : "false"}
      tabIndex={0}
      onKeyDown={onKey}
      aria-label={`${title}: animation. Space plays or pauses, the arrow keys step.`}
      className="focus-ring my-8 min-w-0 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900"
    >
      <figcaption>
        <p className="font-mono text-[0.65rem] uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
          Animated · driven by the model
        </p>
        <p className="mt-1 font-semibold text-neutral-900 dark:text-neutral-100">
          {title}
        </p>
        <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
          {summary}
        </p>
      </figcaption>

      <div className="mt-4 min-w-0" data-testid="visual">
        {visual}
      </div>

      <div
        role="group"
        aria-label="Animation controls"
        className="mt-3 flex flex-wrap items-center gap-2"
      >
        <button
          type="button"
          className={BTN}
          onClick={s.reset}
          aria-label="Reset to the first step"
          title="Reset (Home)"
        >
          ⏮
        </button>
        <button
          type="button"
          className={BTN}
          onClick={s.back}
          disabled={s.step === 0}
          aria-label="Step back"
          title="Step back (←)"
        >
          ◀
        </button>
        <button
          type="button"
          className={`${BTN} min-w-20 font-medium`}
          onClick={s.toggle}
          aria-label={s.playing ? "Pause" : "Play"}
          data-testid="play"
          title="Play / pause (Space)"
        >
          {s.playing ? "❚❚ Pause" : "▶ Play"}
        </button>
        <button
          type="button"
          className={BTN}
          onClick={s.forward}
          disabled={s.step >= s.n - 1}
          aria-label="Step forward"
          title="Step forward (→)"
        >
          ▶
        </button>
        <label className="ml-auto flex items-center gap-1 text-xs text-neutral-600 dark:text-neutral-400">
          Speed
          <select
            value={s.speed}
            onChange={(e) => s.setSpeed(Number(e.target.value) as Speed)}
            className="focus-ring h-11 rounded border border-neutral-300 bg-white px-1 font-mono text-xs dark:border-neutral-700 dark:bg-neutral-950"
          >
            {SPEEDS.map((v) => (
              <option key={v} value={v}>
                {v}×
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="mt-2 flex items-center gap-3">
        <label
          htmlFor={scrubId}
          className="shrink-0 font-mono text-xs text-neutral-600 dark:text-neutral-400"
        >
          {stepLabel} {s.step + countFrom}/{s.n - 1 + countFrom}
        </label>
        <input
          id={scrubId}
          type="range"
          min={0}
          max={Math.max(0, s.n - 1)}
          value={s.step}
          onChange={(e) => s.setStep(Number(e.target.value))}
          // touching the scrub bar pauses, even on the step already shown
          // (a range input fires no change event when its value is unchanged)
          onPointerDown={s.pause}
          onKeyDown={s.pause}
          className="focus-ring h-11 w-full min-w-0 accent-indigo-600"
          aria-label={`Scrub: ${stepLabel}`}
          data-testid="scrub"
        />
      </div>
      <p
        aria-live="polite"
        data-testid="caption"
        className="mt-2 min-h-10 rounded bg-white px-3 py-2 text-sm text-neutral-800 ring-1 ring-neutral-200 dark:bg-neutral-950 dark:text-neutral-200 dark:ring-neutral-800"
      >
        {caption}
      </p>

      {stats && <div className="mt-3">{stats}</div>}

      {equation && (
        <div
          className="eq-panel mt-3 min-w-0 rounded bg-white px-3 py-1 text-sm ring-1 ring-neutral-200 dark:bg-neutral-950 dark:ring-neutral-800"
          data-hl={hl ?? ""}
          onMouseOver={(e) => {
            const el = (e.target as HTMLElement).closest("[class*='hl-']");
            const m = el?.className.toString().match(/hl-([a-z0-9]+)/);
            onEquationHover?.(m ? (m[1] ?? null) : null);
          }}
          onMouseLeave={() => onEquationHover?.(null)}
        >
          {equation}
        </div>
      )}

      {params && (
        <div className="mt-4 grid min-w-0 gap-3 sm:grid-cols-2">{params}</div>
      )}
    </figure>
  );
}
