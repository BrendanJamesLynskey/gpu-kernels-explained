/**
 * Chapter 10's timeline scenarios: tiles, then load, compute and store
 * times per tile (illustrative units). Shared by TimelineWidget and the
 * learning layer's self-checks, so both run the same schedule.
 */
export const SCENARIOS = {
  compute: { label: "compute-heavy", args: [6, 2.0, 3.0, 1.0] },
  copy: { label: "copy-heavy", args: [6, 3.0, 1.5, 1.0] },
  balanced: { label: "balanced", args: [6, 1.0, 1.0, 1.0] },
} as const;
export type Scenario = keyof typeof SCENARIOS;
