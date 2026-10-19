/**
 * `Remember`: what to remember from a chapter (brief 44, the learning layer).
 * A box at the foot of the chapter with three to five one-line takeaways
 * (the children, usually a Markdown list) and, in `Keep`, the one number or
 * rule of thumb worth keeping. Its numbers come from the same place as the
 * chapter's. Server Components.
 *
 * This file is byte-identical on every *-explained site
 * (src/components/learn/Remember.tsx); change it on all of them at once.
 */
import type { ReactNode } from "react";

export function Remember({ children }: { children: ReactNode }): JSX.Element {
  return (
    <aside
      aria-labelledby="remember-heading"
      data-remember
      className="mt-10 rounded-lg border-l-4 border-accent bg-neutral-50 p-5 dark:border-indigo-400 dark:bg-neutral-900 [&_li]:my-1 [&_ul]:mt-3 [&_ul]:list-disc [&_ul]:pl-5"
    >
      <h2
        id="remember-heading"
        className="text-lg font-semibold tracking-tight"
      >
        What to remember
      </h2>
      <div className="text-sm leading-6">{children}</div>
    </aside>
  );
}

/**
 * The one number or rule of thumb worth keeping, inside `Remember`. A `div`,
 * not a `p`: MDX wraps multi-line children in their own paragraph, and a
 * `p` inside a `p` breaks hydration.
 */
export function Keep({ children }: { children: ReactNode }): JSX.Element {
  return (
    <div
      data-keep
      className="mt-4 flex flex-wrap items-baseline gap-x-2 border-t border-neutral-200 pt-3 dark:border-neutral-700 [&_p]:m-0"
    >
      <span className="font-mono text-xs uppercase tracking-widest text-neutral-600 dark:text-neutral-400">
        Keep
      </span>
      <div className="min-w-0">{children}</div>
    </div>
  );
}
