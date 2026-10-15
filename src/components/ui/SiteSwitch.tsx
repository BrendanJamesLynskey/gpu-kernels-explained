/**
 * Cross-site navigation between the companion sites, in four tracks:
 *
 *   Foundations: Decoder · Training
 *   Hardware:    Kernels · Systolic · Accelerators · Racks
 *   Inference:   Serving · Architectures · Numerics · Trade-offs
 *   Agents:      Harnesses · Protocols · Context · Orchestration · Evals · Security
 *
 * One name per level, never reused: track names, site names and each site's
 * own chapter names are all distinct.
 *
 * The same file, byte for byte, sits in every site's header; only the `current` prop differs.
 *
 * One "<Track> › <Site> ▾" button at every width, so the button itself says where the reader
 * is. It is a native <details>, so it works with no JavaScript: one click opens a panel listing
 * every track under its heading and tagline (side by side from the `lg` breakpoint up, stacked
 * below it), each site with a one-line description; one click on a site goes there. The current
 * site is highlighted. The panel is placed against the window's edge (16 px in from both sides on
 * a phone, the right side from `lg`), not against the button, so it stays on screen wherever a
 * site's header puts the button.
 *
 * `placeOf(current)` gives a site's place (track and site name) and `SitePlace` renders it on the
 * button; `nameOf(current)` gives the site's own name ("Agent Security Explained", its title up to
 * the colon) for its header's brand, so the header names the site once and the button says where
 * it sits, and neither drifts from the menu.
 *
 * A site that is not live yet is shown but not linked, marked "(soon)"; flip `live` when it
 * launches (on every site). An `unlisted` entry is never shown in the switch and belongs to no
 * track; it is kept so that its site can still pass its own key as `current` (the Compute in
 * Transit results site names itself on the button). Server Component.
 */

const SITES = [
  {
    key: "decoder",
    group: "foundations",
    label: "Decoder",
    blurb: "How it runs, one forward pass",
    href: "https://transformer-decoder-explained.vercel.app",
    title: "Transformer Decoder Explained: how one forward pass works",
    live: true,
  },
  {
    key: "training",
    group: "foundations",
    label: "Training",
    blurb: "How it learns, from one step to many GPUs",
    href: "https://transformer-training-explained.vercel.app",
    title: "Transformer Training Explained: how the same decoder is trained",
    live: true,
  },
  {
    key: "kernels",
    group: "hardware",
    label: "Kernels",
    blurb: "How a GPU executes the maths",
    href: "https://gpu-kernels-explained.vercel.app",
    title: "GPU Kernels Explained: how a GPU executes the maths",
    live: true,
  },
  {
    key: "silicon",
    group: "hardware",
    label: "Systolic",
    blurb: "Inside a matrix engine",
    href: "https://systolic-arrays-explained.vercel.app",
    title: "Systolic Arrays Explained: inside a matrix engine",
    live: true,
  },
  {
    key: "accelerators",
    group: "hardware",
    label: "Accelerators",
    blurb: "The families of AI chips",
    href: "https://ai-accelerators-explained.vercel.app",
    title: "AI Accelerators Explained: the families of AI silicon",
    live: true,
  },
  {
    key: "racks",
    group: "hardware",
    label: "Racks",
    blurb: "GPUs, networks, power, cooling at scale",
    href: "https://ai-racks-explained.vercel.app",
    title: "AI Racks Explained: GPUs, networks, power and cooling at scale",
    live: true,
  },
  {
    key: "inference",
    group: "inference",
    label: "Serving",
    blurb: "How real systems generate text",
    href: "https://llm-inference-explained.vercel.app",
    title: "LLM Inference Explained: how real systems generate text",
    live: true,
  },
  {
    key: "architectures",
    group: "inference",
    label: "Architectures",
    blurb: "How model designs differ",
    href: "https://llm-architectures-explained.vercel.app",
    title: "LLM Architectures Explained: how model designs differ",
    live: true,
  },
  {
    key: "numerics",
    group: "inference",
    label: "Numerics",
    blurb: "Number formats and quantisation",
    href: "https://numerics-explained.vercel.app",
    title: "Numerics Explained: number formats and quantisation",
    live: true,
  },
  {
    key: "tradeoffs",
    group: "inference",
    label: "Trade-offs",
    blurb: "Which lever helps which metric",
    href: "https://inference-tradeoffs-explained.vercel.app",
    title: "Inference Trade-offs Explained: which lever helps which metric",
    live: true,
  },
  {
    key: "harnesses",
    group: "agents",
    label: "Harnesses",
    blurb: "The loop between a model and the world",
    href: "https://agent-harnesses-explained.vercel.app",
    title: "Agent Harnesses Explained: the loop between a model and the world",
    live: true,
  },
  {
    key: "protocols",
    group: "agents",
    label: "Protocols",
    blurb: "MCP, transports, auth and A2A",
    href: "https://agent-protocols-explained.vercel.app",
    title: "Agent Protocols Explained: MCP, transports, auth and A2A",
    live: true,
  },
  {
    key: "context",
    group: "agents",
    label: "Context",
    blurb: "Retrieval, memory and context engineering",
    href: "https://agent-context-explained.vercel.app",
    title: "Agent Context Explained: retrieval, memory and context engineering",
    live: true,
  },
  {
    key: "orchestration",
    group: "agents",
    label: "Orchestration",
    blurb: "Graphs, multi-agent patterns, durability",
    href: "https://agent-orchestration-explained.vercel.app",
    title:
      "Agent Orchestration Explained: graphs, multi-agent patterns and durability",
    live: true,
  },
  {
    key: "evals",
    group: "agents",
    label: "Evals",
    blurb: "Measuring agents",
    href: "https://agent-evals-explained.vercel.app",
    title: "Agent Evals Explained: measuring agents",
    live: true,
  },
  {
    key: "security",
    group: "agents",
    label: "Security",
    blurb: "Threats and defences",
    href: "https://agent-security-explained.vercel.app",
    title: "Agent Security Explained: threats and defences",
    live: true,
  },
  {
    key: "transit",
    group: "none",
    label: "Compute in transit",
    blurb: "Computing on gradients in the network",
    href: "https://compute-in-transit-results.vercel.app",
    title:
      "Compute in Transit: an investigation of computing on gradients in the network",
    live: true,
    unlisted: true,
  },
] as const;

const GROUPS = [
  {
    key: "foundations",
    label: "Foundations",
    blurb: "One small model, end to end",
  },
  {
    key: "hardware",
    label: "Hardware",
    blurb: "The machines underneath, chip to rack",
  },
  { key: "inference", label: "Inference", blurb: "Serving models at scale" },
  { key: "agents", label: "Agents", blurb: "Building on models" },
] as const;

export type SiteKey = (typeof SITES)[number]["key"];
type Site = (typeof SITES)[number];

/** Where a site sits: its track (none for an unlisted site) and its name. */
export type Place = { track: string | null; site: string; title: string };

/** The place of `current`, for its header: e.g. { track: "Inference", site: "Serving" }. */
export function placeOf(current: SiteKey): Place {
  const s = SITES.find((x) => x.key === current) ?? SITES[0];
  const g = GROUPS.find((x) => x.key === s.group);
  return { track: g ? g.label : null, site: s.label, title: s.title };
}

/** A site's own name, for its header's brand: its title up to the colon, e.g. "Agent Security Explained". */
export function nameOf(current: SiteKey): string {
  const title = placeOf(current).title;
  const colon = title.indexOf(":");
  return colon < 0 ? title : title.slice(0, colon);
}

/** "Track › Site" as inline text (the track muted), as the button shows it. */
export function SitePlace({ current }: { current: SiteKey }): JSX.Element {
  const p = placeOf(current);
  return (
    <span data-site-place={current}>
      {p.track && (
        <>
          <span className="text-neutral-500 dark:text-neutral-400">
            {p.track}
          </span>
          <span aria-hidden className="px-1 text-neutral-400">
            ›
          </span>
          <span className="sr-only">: </span>
        </>
      )}
      <span className="font-medium">{p.site}</span>
    </span>
  );
}

/** The entries of one group that are shown in the switch (`unlisted` ones left out). */
function sitesIn(group: string): Site[] {
  return SITES.filter((s) => s.group === group && !("unlisted" in s));
}

const ITEM =
  "focus-ring flex min-h-11 flex-col justify-center rounded-md px-2.5 py-1 text-neutral-700 hover:bg-neutral-100 hover:text-neutral-950 dark:text-neutral-300 dark:hover:bg-neutral-800 dark:hover:text-white";
const CURRENT =
  "focus-ring flex min-h-11 flex-col justify-center rounded-md bg-accent px-2.5 py-1 text-accent-fg";
const SOON =
  "flex min-h-11 cursor-default flex-col justify-center px-2.5 py-1 text-neutral-500 dark:text-neutral-400";
const BLURB = "text-[0.7rem] leading-snug";
const MUTED = "text-neutral-600 dark:text-neutral-400";

function Item({ s, current }: { s: Site; current: SiteKey }): JSX.Element {
  if (!s.live)
    return (
      <span
        title={s.title}
        aria-disabled="true"
        data-site-key={s.key}
        className={SOON}
      >
        <span>
          {s.label}
          <span className="ml-1 text-[0.65rem]">(soon)</span>
        </span>
        <span className={BLURB}>{s.blurb}</span>
      </span>
    );
  const here = s.key === current;
  return (
    <a
      href={s.href}
      title={s.title}
      data-site-key={s.key}
      aria-current={here ? "true" : undefined}
      className={here ? CURRENT : ITEM}
    >
      <span className="font-medium">{s.label}</span>
      <span className={`${BLURB} ${here ? "" : MUTED}`}>{s.blurb}</span>
    </a>
  );
}

export function SiteSwitch({ current }: { current: SiteKey }): JSX.Element {
  return (
    <details className="relative text-xs" data-site-switch="menu">
      <summary className="focus-ring flex min-h-11 cursor-pointer list-none items-center gap-1 whitespace-nowrap rounded-md border border-neutral-300 px-3 dark:border-neutral-700">
        <SitePlace current={current} />
        <span aria-hidden>▾</span>
      </summary>
      <div className="fixed left-4 right-4 z-20 mt-1 flex max-h-[70vh] flex-col gap-1 overflow-y-auto rounded-md border border-neutral-300 bg-white p-1 shadow-lg lg:left-auto lg:max-h-[80vh] lg:w-[60rem] lg:max-w-[calc(100vw-2rem)] lg:flex-row lg:gap-2 lg:p-2 dark:border-neutral-700 dark:bg-neutral-900">
        {GROUPS.map((g) => (
          <nav
            key={g.key}
            aria-label={`Companion sites: ${g.label}`}
            data-site-group={g.key}
            className="flex flex-col lg:flex-1"
          >
            <p className="px-2.5 pt-2 font-mono text-[0.65rem] uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
              {g.label}
            </p>
            <p className="px-2.5 pb-1 text-[0.7rem] italic text-neutral-500 dark:text-neutral-400">
              {g.blurb}
            </p>
            {sitesIn(g.key).map((s) => (
              <Item key={s.key} s={s} current={current} />
            ))}
          </nav>
        ))}
      </div>
    </details>
  );
}
