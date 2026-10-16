/**
 * The landing page's shared parts (brief 39), the same on every companion site: the primary
 * "Start with chapter 1 →" button, and the slim "How this is checked" strip, which says in one
 * line how the site's numbers are checked and links to the About section that shows it.
 * Server Components.
 */
import type { Route } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

export function StartButton({
  href,
  children = "Start with chapter 1 →",
}: {
  href: Route;
  children?: ReactNode;
}): JSX.Element {
  return (
    <Link
      href={href}
      data-start-button
      className="focus-ring inline-flex min-h-11 items-center rounded bg-accent px-5 py-2 text-sm font-medium text-accent-fg hover:opacity-90"
    >
      {children}
    </Link>
  );
}

export function CheckedStrip({
  href,
  children,
}: {
  href: Route;
  children: ReactNode;
}): JSX.Element {
  return (
    <p
      data-checked-strip
      className="mt-8 rounded-lg border border-neutral-200 bg-neutral-50 px-4 py-3 text-sm leading-relaxed text-neutral-700 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-300"
    >
      <span className="mr-2 font-mono text-xs uppercase tracking-widest text-neutral-600 dark:text-neutral-400">
        How this is checked
      </span>
      {children}{" "}
      <Link
        href={href}
        data-checked-link
        className="focus-ring whitespace-nowrap rounded font-medium underline decoration-neutral-400 underline-offset-4 hover:decoration-current"
      >
        How it is tested →
      </Link>
    </p>
  );
}
