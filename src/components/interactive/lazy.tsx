"use client";

/**
 * Code-split client widgets: each loads its own chunk after the page shell,
 * so pages stay light (the pattern of the companion sites' lazy.tsx). Each
 * widget takes its equation as server-rendered children.
 */
import dynamic from "next/dynamic";

function Placeholder({ what }: { what: string }): JSX.Element {
  return (
    <p
      data-pending-widget
      className="my-8 min-h-96 text-sm text-neutral-600 dark:text-neutral-400"
    >
      Loading the {what}…
    </p>
  );
}

const loading = (what: string) =>
  function Loading(): JSX.Element {
    return <Placeholder what={what} />;
  };

export const HierarchyWidget = dynamic(() => import("./HierarchyWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const RooflineWidget = dynamic(() => import("./RooflineWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const SimtWidget = dynamic(() => import("./SimtWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const CoalescingWidget = dynamic(() => import("./CoalescingWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const BankWidget = dynamic(() => import("./BankWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const OccupancyWidget = dynamic(() => import("./OccupancyWidget"), {
  ssr: false,
  loading: loading("animation"),
});
