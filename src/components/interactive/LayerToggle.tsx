"use client";

/**
 * Three-layer toggle (Concept / Maths / Code).
 *
 * Drives visibility via `data-layer-{kind}="on|off"` on `<html>`. The
 * `<Layer>` wrapper components inside MDX each carry a `data-layer="<kind>"`
 * attribute and a CSS rule in `globals.css` hides them when the matching
 * data attribute on `<html>` is "off". This keeps MDX entirely server-
 * rendered — the toggle just flips data attributes.
 *
 * The active layers are persisted in `localStorage` so a reader's
 * preference survives navigation. The default is Concept on, Maths off,
 * Code off (per SPEC §6.1).
 */
import { useEffect, useState } from "react";

const KINDS = ["concept", "maths", "code"] as const;
type Kind = (typeof KINDS)[number];

type State = Record<Kind, boolean>;
const DEFAULT: State = { concept: true, maths: false, code: false };
const STORAGE_KEY = "te:layers";

function applyToDocument(state: State): void {
  if (typeof document === "undefined") return;
  for (const kind of KINDS) {
    document.documentElement.dataset[
      `layer${kind[0]!.toUpperCase()}${kind.slice(1)}`
    ] = state[kind] ? "on" : "off";
  }
}

function loadInitial(): State {
  if (typeof window === "undefined") return DEFAULT;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT;
    const parsed = JSON.parse(raw) as Partial<State>;
    return {
      concept: parsed.concept ?? DEFAULT.concept,
      maths: parsed.maths ?? DEFAULT.maths,
      code: parsed.code ?? DEFAULT.code,
    };
  } catch {
    return DEFAULT;
  }
}

export function LayerToggle(): JSX.Element {
  const [state, setState] = useState<State>(DEFAULT);

  // Hydrate from localStorage and apply the initial CSS state.
  useEffect(() => {
    const initial = loadInitial();
    setState(initial);
    applyToDocument(initial);
  }, []);

  function toggle(kind: Kind): void {
    setState((prev) => {
      const next = { ...prev, [kind]: !prev[kind] };
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      applyToDocument(next);
      return next;
    });
  }

  return (
    <div
      role="group"
      aria-label="Show or hide the conceptual, mathematical, and code layers."
      className="inline-flex rounded-md border border-neutral-300 bg-white text-xs shadow-sm dark:border-neutral-700 dark:bg-neutral-900"
    >
      {KINDS.map((kind) => (
        <button
          key={kind}
          type="button"
          role="switch"
          aria-checked={state[kind]}
          onClick={() => toggle(kind)}
          className={`focus-ring px-3 py-1.5 capitalize first:rounded-l-md last:rounded-r-md ${
            state[kind]
              ? "bg-accent text-accent-fg"
              : "text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-800"
          }`}
        >
          {kind}
        </button>
      ))}
    </div>
  );
}
