/**
 * Boxed asides used inside the MDX chapters:
 *   prereq      — "read this first", usually a Transformer Decoder Explainer chapter;
 *   deeper      — "go deeper": the InfSim decks, the glossary, the papers;
 *   illustrative — flags numbers that are illustrative, not measured.
 * Server Component.
 */
import type { ReactNode } from "react";

const STYLES = {
  prereq: {
    label: "Before this",
    cls: "border-emerald-300 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950/30",
  },
  deeper: {
    label: "Go deeper",
    cls: "border-sky-300 bg-sky-50 dark:border-sky-800 dark:bg-sky-950/30",
  },
  illustrative: {
    label: "Illustrative",
    cls: "border-amber-300 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/30",
  },
} as const;

export function Callout({
  kind,
  children,
}: {
  kind: keyof typeof STYLES;
  children: ReactNode;
}): JSX.Element {
  const s = STYLES[kind];
  return (
    <aside
      data-callout={kind}
      className={`my-6 rounded-lg border px-4 py-3 text-sm ${s.cls}`}
    >
      <p className="font-mono text-[0.65rem] uppercase tracking-widest text-neutral-600 dark:text-neutral-400">
        {s.label}
      </p>
      <div className="callout-body">{children}</div>
    </aside>
  );
}
