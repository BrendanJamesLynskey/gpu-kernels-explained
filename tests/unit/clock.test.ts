/** The animation clock (pure functions). */
import { describe, expect, it } from "vitest";

import {
  SPEEDS,
  START,
  clampStep,
  fraction,
  keyAction,
  tick,
} from "@/lib/anim/clock";

describe("animation clock", () => {
  it("offers 0.25x to 4x", () => {
    expect(SPEEDS[0]).toBe(0.25);
    expect(SPEEDS[SPEEDS.length - 1]).toBe(4);
  });
  it("advances one step per stepMs at 1x, more at higher speed", () => {
    let c = tick(START, 699, 1, 700, 10);
    expect(c.step).toBe(0);
    c = tick(c, 1, 1, 700, 10);
    expect(c).toEqual({ step: 1, acc: 0, done: false });
    expect(tick(START, 700, 4, 700, 10).step).toBe(4);
    expect(tick(START, 700, 0.25, 700, 10).step).toBe(0);
    expect(tick(START, 2800, 0.25, 700, 10).step).toBe(1);
  });
  it("stops on the last step and reports done", () => {
    expect(tick(START, 1e6, 1, 700, 10)).toEqual({
      step: 9,
      acc: 0,
      done: true,
    });
    expect(tick(START, 1, 1, 700, 1)).toEqual({ step: 0, acc: 0, done: true });
  });
  it("is reproducible: the same total time gives the same frame however it is split", () => {
    let a = START;
    for (let i = 0; i < 100; i++) a = tick(a, 16, 2, 300, 50);
    const b = tick(START, 1600, 2, 300, 50);
    expect(a.step).toBe(b.step);
    expect(a.acc).toBeCloseTo(b.acc, 9);
  });
  it("clamps scrubbing and computes the in-between fraction", () => {
    expect(clampStep(-3, 10)).toBe(0);
    expect(clampStep(42, 10)).toBe(9);
    expect(clampStep(3.6, 10)).toBe(4);
    expect(clampStep(5, 0)).toBe(0);
    expect(fraction({ step: 2, acc: 350, done: false }, 700)).toBe(0.5);
    expect(fraction({ step: 2, acc: 900, done: false }, 700)).toBe(1);
  });
  it("maps the keys", () => {
    expect(keyAction(" ")).toBe("toggle");
    expect(keyAction("Spacebar")).toBe("toggle");
    expect(keyAction("k")).toBe("toggle");
    expect(keyAction("ArrowLeft")).toBe("back");
    expect(keyAction("j")).toBe("back");
    expect(keyAction("ArrowRight")).toBe("forward");
    expect(keyAction("l")).toBe("forward");
    expect(keyAction("Home")).toBe("reset");
    expect(keyAction("End")).toBe("end");
    expect(keyAction("x")).toBeNull();
  });
});
