/**
 * Site-wide header.
 *
 * Server Component, adapted from transformer-explainer's `SiteHeader`
 * (same layout and classes), as on the other companion sites: no accounts,
 * so the sign-in controls are replaced by the cross-site switch.
 */
import Link from "next/link";

import { SiteSwitch } from "./SiteSwitch";

const NAV = [
  { href: "/learn", label: "Learn" },
  { href: "/gpus", label: "GPUs" },
  { href: "/about", label: "About" },
] as const;

export function SiteHeader(): JSX.Element {
  return (
    <header className="border-b border-neutral-200 dark:border-neutral-800">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-3 sm:px-6">
        <Link
          href="/"
          className="focus-ring rounded font-mono text-sm font-medium tracking-tight"
        >
          gpu-kernels-explained
        </Link>
        <nav
          aria-label="Site"
          className="flex flex-wrap items-center gap-1 text-sm sm:gap-3"
        >
          {NAV.map((n) => (
            <Link
              key={n.href}
              href={n.href}
              className="focus-ring rounded px-2 py-1 text-neutral-700 hover:text-neutral-950 dark:text-neutral-300 dark:hover:text-white"
            >
              {n.label}
            </Link>
          ))}
          <SiteSwitch current="kernels" />
        </nav>
      </div>
    </header>
  );
}
