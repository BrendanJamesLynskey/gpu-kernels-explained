/**
 * `<Layer>` — wraps a chunk of MDX content and labels it with one of the
 * three teaching layers. The actual show/hide is CSS-driven by the
 * `data-layer-{kind}` attributes that `<LayerToggle>` writes onto `<html>`,
 * so this component itself is a Server Component (no hooks needed).
 *
 * Vendored from transformer-explainer; one change: the label gains
 * `dark:text-neutral-400`, because neutral-500 on the dark background fails
 * WCAG AA contrast (axe, dark mode).
 */
import type { ReactNode } from "react";

type Kind = "concept" | "maths" | "code";

const LABELS: Record<Kind, string> = {
  concept: "Concept",
  maths: "Maths",
  code: "Code",
};

const ACCENTS: Record<Kind, string> = {
  concept: "border-l-emerald-400",
  maths: "border-l-sky-400",
  code: "border-l-amber-400",
};

export function Layer({
  kind,
  children,
}: {
  kind: Kind;
  children: ReactNode;
}): JSX.Element {
  return (
    <section
      data-layer={kind}
      className={`te-layer my-6 border-l-4 ${ACCENTS[kind]} pl-4`}
    >
      <p className="font-mono text-[0.65rem] uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
        {LABELS[kind]}
      </p>
      <div className="mt-1">{children}</div>
    </section>
  );
}
