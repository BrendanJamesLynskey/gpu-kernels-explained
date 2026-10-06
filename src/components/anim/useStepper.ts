"use client";

/**
 * Drives one animation: a requestAnimationFrame loop with an explicit clock
 * (src/lib/anim/clock.ts), never a CSS timeline the model cannot drive.
 *
 * - Plays when the animation scrolls into view and pauses when it leaves
 *   (IntersectionObserver), unless the reader has paused it.
 * - With `prefers-reduced-motion: reduce` it never plays by itself: it
 *   starts paused and the reader steps (Play still works when pressed).
 * - `resetKey` changes when the parameters change: the model has re-run,
 *   so the animation restarts from step 0.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import {
  START,
  clampStep,
  fraction,
  tick,
  type ClockState,
  type Speed,
} from "@/lib/anim/clock";

export type Stepper = {
  n: number;
  step: number;
  /** 0 .. 1 towards the next step (for smooth in-between frames). */
  frac: number;
  playing: boolean;
  speed: Speed;
  reducedMotion: boolean;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  setStep: (s: number) => void;
  forward: () => void;
  back: () => void;
  reset: () => void;
  setSpeed: (s: Speed) => void;
  /** Attach to the animation's container (visibility tracking). */
  ref: (el: HTMLElement | null) => void;
};

function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || !window.matchMedia) return true;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function useStepper(
  n: number,
  {
    stepMs = 700,
    resetKey = "",
    smooth = false,
  }: { stepMs?: number; resetKey?: string; smooth?: boolean } = {},
): Stepper {
  // The clock lives in a ref (it changes every frame); React state holds
  // only what is on screen, updated when the step changes (or every frame
  // for smooth animations).
  const clockRef = useRef<ClockState>(START);
  const [clock, setClock] = useState<ClockState>(START);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<Speed>(1);
  const [reducedMotion, setReduced] = useState(true);
  const userPaused = useRef(false);
  const visible = useRef(false);
  const observer = useRef<IntersectionObserver | null>(null);

  const show = useCallback((c: ClockState) => {
    clockRef.current = c;
    setClock(c);
  }, []);

  useEffect(() => {
    setReduced(prefersReducedMotion());
  }, []);

  // The parameters changed: the model has re-run, start again.
  useEffect(() => {
    show(START);
    userPaused.current = false;
    setPlaying(!prefersReducedMotion() && visible.current);
  }, [resetKey, n, show]);

  // The clock.
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const dt = Math.min(100, now - last);
      last = now;
      const c = clockRef.current;
      const next = tick(c, dt, speed, stepMs, n);
      if (smooth || next.step !== c.step || next.done) show(next);
      else clockRef.current = next;
      if (next.done) {
        setPlaying(false);
        return;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, stepMs, n, smooth, show]);

  const ref = useCallback((el: HTMLElement | null) => {
    observer.current?.disconnect();
    if (!el || typeof IntersectionObserver === "undefined") return;
    observer.current = new IntersectionObserver(
      (entries) => {
        const e = entries[0];
        if (!e) return;
        visible.current = e.isIntersecting;
        if (!e.isIntersecting) setPlaying(false);
        else if (
          !userPaused.current &&
          !prefersReducedMotion() &&
          !clockRef.current.done
        )
          setPlaying(true);
      },
      { threshold: 0.25 },
    );
    observer.current.observe(el);
  }, []);

  useEffect(() => () => observer.current?.disconnect(), []);

  const setStep = useCallback(
    (s: number) => {
      userPaused.current = true;
      setPlaying(false);
      const step = clampStep(s, n);
      show({ step, acc: 0, done: step === n - 1 });
    },
    [n, show],
  );

  const play = useCallback(() => {
    userPaused.current = false;
    const c = clockRef.current;
    if (c.done || c.step >= n - 1) show(START);
    setPlaying(true);
  }, [n, show]);
  const pause = useCallback(() => {
    userPaused.current = true;
    setPlaying(false);
  }, []);

  return {
    n,
    step: clock.step,
    frac: smooth ? fraction(clock, stepMs) : 0,
    playing,
    speed,
    reducedMotion,
    play,
    pause,
    toggle: () => (playing ? pause() : play()),
    setStep,
    forward: () => setStep(clock.step + 1),
    back: () => setStep(clock.step - 1),
    reset: () => setStep(0),
    setSpeed,
    ref,
  };
}
