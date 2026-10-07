/**
 * Cross-site navigation between the companion sites: Decoder · Inference ·
 * Architectures · Kernels · Numerics · Silicon · Trade-offs. The same file,
 * byte for byte, sits in every site's header; only the `current` prop
 * differs.
 *
 * Seven items do not fit one row on a 390 px phone, so the switch is a full
 * row from the `sm` breakpoint up and a compact dropdown (a native
 * <details>, no JavaScript) below it. A site that is not live yet is shown
 * but not linked; flip `live` when it launches (on every site).
 *
 * Server Component (plain links, no state).
 */

const SITES = [
  {
    key: "decoder",
    label: "Decoder",
    href: "https://transformer-decoder-explained.vercel.app",
    title: "Transformer Decoder Explainer: how one forward pass works",
    live: true,
  },
  {
    key: "inference",
    label: "Inference",
    href: "https://llm-inference-explained.vercel.app",
    title: "LLM Inference Explained: how real systems generate text",
    live: true,
  },
  {
    key: "architectures",
    label: "Architectures",
    href: "https://llm-architectures-explained.vercel.app",
    title: "LLM Architectures Explained: how model designs differ",
    live: true,
  },
  {
    key: "kernels",
    label: "Kernels",
    href: "https://gpu-kernels-explained.vercel.app",
    title: "GPU Kernels Explained: how a GPU executes the maths",
    live: true,
  },
  {
    key: "numerics",
    label: "Numerics",
    href: "https://numerics-explained.vercel.app",
    title: "Numerics Explained: number formats and quantisation",
    live: true,
  },
  {
    key: "silicon",
    label: "Silicon",
    href: "https://systolic-arrays-explained.vercel.app",
    title: "Systolic Arrays Explained: the silicon underneath",
    live: true,
  },
  {
    key: "tradeoffs",
    label: "Trade-offs",
    href: "https://inference-tradeoffs-explained.vercel.app",
    title: "Inference Trade-offs Explained: which lever helps which metric",
    live: true,
  },
] as const;

export type SiteKey = (typeof SITES)[number]["key"];

const ITEM =
  "focus-ring rounded-md px-2.5 py-1 text-neutral-700 hover:text-neutral-950 dark:text-neutral-300 dark:hover:text-white";
const CURRENT =
  "focus-ring rounded-md bg-accent px-2.5 py-1 font-medium text-accent-fg";
const SOON =
  "cursor-default px-2.5 py-1 text-neutral-500 dark:text-neutral-400";

function Item({
  s,
  current,
  block = false,
}: {
  s: (typeof SITES)[number];
  current: SiteKey;
  block?: boolean;
}): JSX.Element {
  const size = block ? " flex min-h-11 items-center" : "";
  if (!s.live)
    return (
      <span title={s.title} aria-disabled="true" className={SOON + size}>
        {s.label}
        <span className="ml-1 text-[0.65rem]">(soon)</span>
      </span>
    );
  return (
    <a
      href={s.href}
      title={s.title}
      aria-current={s.key === current ? "true" : undefined}
      className={(s.key === current ? CURRENT : ITEM) + size}
    >
      {s.label}
    </a>
  );
}

export function SiteSwitch({ current }: { current: SiteKey }): JSX.Element {
  const here = SITES.find((s) => s.key === current) ?? SITES[0];
  return (
    <>
      {/* sm and up: the full row */}
      <nav
        aria-label="Companion sites"
        className="hidden items-center rounded-md border border-neutral-300 text-xs sm:inline-flex dark:border-neutral-700"
      >
        {SITES.map((s, i) => (
          <span key={s.key} className="inline-flex items-center">
            {i > 0 && (
              <span aria-hidden className="text-neutral-400">
                ·
              </span>
            )}
            <Item s={s} current={current} />
          </span>
        ))}
      </nav>
      {/* phones: a compact dropdown */}
      <details
        className="relative text-xs sm:hidden"
        data-site-switch="compact"
      >
        <summary className="focus-ring flex min-h-11 cursor-pointer list-none items-center gap-1 rounded-md border border-neutral-300 px-3 dark:border-neutral-700">
          <span className="text-neutral-500 dark:text-neutral-400">Sites:</span>
          <span className="font-medium">{here.label}</span>
          <span aria-hidden>▾</span>
        </summary>
        <nav
          aria-label="Companion sites"
          className="absolute right-0 z-20 mt-1 flex w-48 flex-col rounded-md border border-neutral-300 bg-white p-1 shadow-lg dark:border-neutral-700 dark:bg-neutral-900"
        >
          {SITES.map((s) => (
            <Item key={s.key} s={s} current={current} block />
          ))}
        </nav>
      </details>
    </>
  );
}
