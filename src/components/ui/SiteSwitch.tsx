/**
 * Cross-site navigation between the companion sites, in three groups:
 *
 *   Training:  Training
 *   Inference: Decoder · Inference · Architectures · Kernels · Numerics · Silicon · Trade-offs
 *   Agents:    Harnesses · Protocols · Context · Orchestration · Evals · Security
 *
 * The same file, byte for byte, sits in every site's header; only the `current` prop differs.
 *
 * One "Site: <this site> ▾" button at every width. It is a native <details>, so it works with
 * no JavaScript: one click opens a panel listing every group under its heading (side by side
 * from the `lg` breakpoint up, stacked below it), and one click on a site goes there. The
 * current site is highlighted.
 *
 * A site that is not live yet is shown but not linked, marked "(soon)"; flip `live` when it
 * launches (on every site). An `unlisted` entry is never shown in the switch; it is kept so
 * that its site can still pass its own key as `current` (the Compute in Transit results site
 * names itself on the button). Server Component.
 */

const SITES = [
  {
    key: "training",
    group: "training",
    label: "Training",
    href: "https://transformer-decoder-explained.vercel.app/training",
    title: "Transformer Decoder Explained, Part 2: how a decoder is trained",
    live: true,
  },
  {
    key: "transit",
    group: "training",
    label: "Compute in transit",
    href: "https://compute-in-transit-results.vercel.app",
    title:
      "Compute in Transit: an investigation of computing on gradients in the network",
    live: true,
    unlisted: true,
  },
  {
    key: "decoder",
    group: "inference",
    label: "Decoder",
    href: "https://transformer-decoder-explained.vercel.app",
    title: "Transformer Decoder Explainer: how one forward pass works",
    live: true,
  },
  {
    key: "inference",
    group: "inference",
    label: "Inference",
    href: "https://llm-inference-explained.vercel.app",
    title: "LLM Inference Explained: how real systems generate text",
    live: true,
  },
  {
    key: "architectures",
    group: "inference",
    label: "Architectures",
    href: "https://llm-architectures-explained.vercel.app",
    title: "LLM Architectures Explained: how model designs differ",
    live: true,
  },
  {
    key: "kernels",
    group: "inference",
    label: "Kernels",
    href: "https://gpu-kernels-explained.vercel.app",
    title: "GPU Kernels Explained: how a GPU executes the maths",
    live: true,
  },
  {
    key: "numerics",
    group: "inference",
    label: "Numerics",
    href: "https://numerics-explained.vercel.app",
    title: "Numerics Explained: number formats and quantisation",
    live: true,
  },
  {
    key: "silicon",
    group: "inference",
    label: "Silicon",
    href: "https://systolic-arrays-explained.vercel.app",
    title: "Systolic Arrays Explained: the silicon underneath",
    live: true,
  },
  {
    key: "tradeoffs",
    group: "inference",
    label: "Trade-offs",
    href: "https://inference-tradeoffs-explained.vercel.app",
    title: "Inference Trade-offs Explained: which lever helps which metric",
    live: true,
  },
  {
    key: "harnesses",
    group: "agents",
    label: "Harnesses",
    href: "https://agent-harnesses-explained.vercel.app",
    title: "Agent Harnesses Explained: the loop between a model and the world",
    live: true,
  },
  {
    key: "protocols",
    group: "agents",
    label: "Protocols",
    href: "https://agent-protocols-explained.vercel.app",
    title: "Agent Protocols Explained: MCP, transports, auth and A2A",
    live: true,
  },
  {
    key: "context",
    group: "agents",
    label: "Context",
    href: "https://agent-context-explained.vercel.app",
    title: "Agent Context Explained: retrieval, memory and context engineering",
    live: true,
  },
  {
    key: "orchestration",
    group: "agents",
    label: "Orchestration",
    href: "https://agent-orchestration-explained.vercel.app",
    title:
      "Agent Orchestration Explained: graphs, multi-agent patterns and durability",
    live: true,
  },
  {
    key: "evals",
    group: "agents",
    label: "Evals",
    href: "https://agent-evals-explained.vercel.app",
    title: "Agent Evals Explained: measuring agents",
    live: false,
  },
  {
    key: "security",
    group: "agents",
    label: "Security",
    href: "https://agent-security-explained.vercel.app",
    title: "Agent Security Explained: threats and defences",
    live: false,
  },
] as const;

const GROUPS = [
  { key: "training", label: "Training" },
  { key: "inference", label: "Inference" },
  { key: "agents", label: "Agents" },
] as const;

export type SiteKey = (typeof SITES)[number]["key"];
type Site = (typeof SITES)[number];

/** The entries of one group that are shown in the switch (`unlisted` ones left out). */
function sitesIn(group: string): Site[] {
  return SITES.filter((s) => s.group === group && !("unlisted" in s));
}

const ITEM =
  "focus-ring flex min-h-11 items-center rounded-md px-2.5 text-neutral-700 hover:bg-neutral-100 hover:text-neutral-950 dark:text-neutral-300 dark:hover:bg-neutral-800 dark:hover:text-white";
const CURRENT =
  "focus-ring flex min-h-11 items-center rounded-md bg-accent px-2.5 font-medium text-accent-fg";
const SOON =
  "flex min-h-11 cursor-default items-center px-2.5 text-neutral-500 dark:text-neutral-400";

function Item({ s, current }: { s: Site; current: SiteKey }): JSX.Element {
  if (!s.live)
    return (
      <span title={s.title} aria-disabled="true" className={SOON}>
        {s.label}
        <span className="ml-1 text-[0.65rem]">(soon)</span>
      </span>
    );
  return (
    <a
      href={s.href}
      title={s.title}
      aria-current={s.key === current ? "true" : undefined}
      className={s.key === current ? CURRENT : ITEM}
    >
      {s.label}
    </a>
  );
}

export function SiteSwitch({ current }: { current: SiteKey }): JSX.Element {
  const here = SITES.find((s) => s.key === current) ?? SITES[0];
  return (
    <details className="relative text-xs" data-site-switch="menu">
      <summary className="focus-ring flex min-h-11 cursor-pointer list-none items-center gap-1 rounded-md border border-neutral-300 px-3 dark:border-neutral-700">
        <span className="text-neutral-500 dark:text-neutral-400">Site:</span>
        <span className="font-medium">{here.label}</span>
        <span aria-hidden>▾</span>
      </summary>
      <div className="absolute right-0 z-20 mt-1 flex max-h-[70vh] w-52 flex-col gap-1 overflow-y-auto rounded-md border border-neutral-300 bg-white p-1 shadow-lg lg:max-h-none lg:w-[30rem] lg:flex-row lg:gap-2 lg:p-2 dark:border-neutral-700 dark:bg-neutral-900">
        {GROUPS.map((g) => (
          <nav
            key={g.key}
            aria-label={`Companion sites: ${g.label}`}
            data-site-group={g.key}
            className="flex flex-col lg:flex-1"
          >
            <p className="px-2.5 pb-1 pt-2 font-mono text-[0.65rem] uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
              {g.label}
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
