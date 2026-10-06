/**
 * The animation clock, as pure functions (unit-tested): every animation on
 * the site is a sequence of model states, and the clock only decides which
 * state (and how far towards the next) is on screen. Rendering is a pure
 * function of (state, fraction), so any frame can be reproduced by setting
 * the step.
 */

/** The speeds the controls offer (the visual standard: 0.25x to 4x). */
export const SPEEDS = [0.25, 0.5, 1, 2, 4] as const;
export type Speed = (typeof SPEEDS)[number];

export type ClockState = {
  /** The current step, 0 .. n - 1. */
  step: number;
  /** Milliseconds accumulated towards the next step. */
  acc: number;
  /** True once the last step has been reached by playing. */
  done: boolean;
};

export const START: ClockState = { step: 0, acc: 0, done: false };

/**
 * Advance the clock by `dtMs` of wall time at `speed`, with each step lasting
 * `stepMs` at 1x. Stops on the last step (it does not loop).
 */
export function tick(
  c: ClockState,
  dtMs: number,
  speed: number,
  stepMs: number,
  n: number,
): ClockState {
  if (n <= 1) return { step: 0, acc: 0, done: true };
  let step = c.step;
  let acc = c.acc + dtMs * speed;
  while (acc >= stepMs && step < n - 1) {
    acc -= stepMs;
    step += 1;
  }
  if (step >= n - 1) return { step: n - 1, acc: 0, done: true };
  return { step, acc, done: false };
}

/** Clamp a step into range (for the scrub bar and the step buttons). */
export function clampStep(step: number, n: number): number {
  if (n <= 0) return 0;
  return Math.max(0, Math.min(n - 1, Math.round(step)));
}

/** Fraction of the way from this step to the next, 0 .. 1. */
export function fraction(c: ClockState, stepMs: number): number {
  return Math.max(0, Math.min(1, c.acc / stepMs));
}

/** The keyboard map shared by every animation. */
export type KeyAction = "toggle" | "back" | "forward" | "reset" | "end" | null;
export function keyAction(key: string): KeyAction {
  switch (key) {
    case " ":
    case "Spacebar":
    case "k":
      return "toggle";
    case "ArrowLeft":
    case "j":
      return "back";
    case "ArrowRight":
    case "l":
      return "forward";
    case "Home":
      return "reset";
    case "End":
      return "end";
    default:
      return null;
  }
}
