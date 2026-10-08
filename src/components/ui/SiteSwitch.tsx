/**
 * Cross-site navigation between the companion sites, in two groups:
 *
 *   LLM systems: Decoder · Inference · Architectures · Kernels · Numerics · Silicon · Trade-offs
 *   Agents:      Harnesses · Protocols · Context · Orchestration · Evals · Security
 *
 * The same file, byte for byte, sits in every site's header; only the `current` prop differs.
 *
 * - From the `lg` breakpoint up: a group toggle ("LLM systems | Agents") and the chosen
 *   group's row. The toggle is a pair of radio buttons, and CSS (Tailwind's `peer-checked`)
 *   shows the matching row, so it works with no JavaScript. It starts on the current site's
 *   group.
 * - Below `lg`: one "Sites" dropdown (a native <details>) listing both groups under headings.
 *
 * A site that is not live yet is shown but not linked, marked "(soon)"; flip `live` when it
 * launches (on every site). Server Component.
 */

const SITES = [
  {
    key: "decoder",
    group: "llm",
    label: "Decoder",
    href: "https://transformer-decoder-explained.vercel.app",
    title: "Transformer Decoder Explainer: how one forward pass works",
    live: true,
  },
  {
    key: "inference",
    group: "llm",
    label: "Inference",
    href: "https://llm-inference-explained.vercel.app",
    title: "LLM Inference Explained: how real systems generate text",
    live: true,
  },
  {
    key: "architectures",
    group: "llm",
    label: "Architectures",
    href: "https://llm-architectures-explained.vercel.app",
    title: "LLM Architectures Explained: how model designs differ",
    live: true,
  },
  {
    key: "kernels",
    group: "llm",
    label: "Kernels",
    href: "https://gpu-kernels-explained.vercel.app",
    title: "GPU Kernels Explained: how a GPU executes the maths",
    live: true,
  },
  {
    key: "numerics",
    group: "llm",
    label: "Numerics",
    href: "https://numerics-explained.vercel.app",
    title: "Numerics Explained: number formats and quantisation",
    live: true,
  },
  {
    key: "silicon",
    group: "llm",
    label: "Silicon",
    href: "https://systolic-arrays-explained.vercel.app",
    title: "Systolic Arrays Explained: the silicon underneath",
    live: true,
  },
  {
    key: "tradeoffs",
    group: "llm",
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
    live: false,
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
  { key: "llm", label: "LLM systems" },
  { key: "agents", label: "Agents" },
] as const;

export type SiteKey = (typeof SITES)[number]["key"];
type Site = (typeof SITES)[number];

const ITEM =
  "focus-ring rounded-md px-2.5 py-1 text-neutral-700 hover:text-neutral-950 dark:text-neutral-300 dark:hover:text-white";
const CURRENT =
  "focus-ring rounded-md bg-accent px-2.5 py-1 font-medium text-accent-fg";
const SOON =
  "cursor-default px-2.5 py-1 text-neutral-500 dark:text-neutral-400";
const TOGGLE =
  "cursor-pointer select-none px-2.5 py-1 text-neutral-600 hover:text-neutral-950 dark:text-neutral-400 dark:hover:text-white";

function Item({
  s,
  current,
  block = false,
}: {
  s: Site;
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

function Row({
  group,
  current,
  className,
}: {
  group: (typeof GROUPS)[number];
  current: SiteKey;
  className: string;
}): JSX.Element {
  return (
    <nav
      aria-label={`Companion sites: ${group.label}`}
      data-site-group={group.key}
      className={className}
    >
      {SITES.filter((s) => s.group === group.key).map((s, i) => (
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
  );
}

export function SiteSwitch({ current }: { current: SiteKey }): JSX.Element {
  const here = SITES.find((s) => s.key === current) ?? SITES[0];
  const ROW =
    "hidden items-center rounded-md border border-neutral-300 dark:border-neutral-700";
  return (
    <>
      {/* lg and up: the group toggle, then the chosen group's row (CSS only) */}
      <div
        className="hidden items-center gap-2 text-xs lg:flex"
        data-site-switch="full"
      >
        <input
          type="radio"
          name="site-switch-group"
          id="site-switch-llm"
          aria-label="Show the LLM systems sites"
          defaultChecked={here.group === "llm"}
          className="peer/llm sr-only"
        />
        <input
          type="radio"
          name="site-switch-group"
          id="site-switch-agents"
          aria-label="Show the agent sites"
          defaultChecked={here.group === "agents"}
          className="peer/agents sr-only"
        />
        <label
          htmlFor="site-switch-llm"
          className={`${TOGGLE} rounded-l-md border border-neutral-300 peer-checked/llm:bg-neutral-200 peer-checked/llm:font-medium peer-checked/llm:text-neutral-950 peer-focus-visible/llm:ring-2 peer-focus-visible/llm:ring-accent dark:border-neutral-700 dark:peer-checked/llm:bg-neutral-800 dark:peer-checked/llm:text-white`}
        >
          LLM systems
        </label>
        <label
          htmlFor="site-switch-agents"
          className={`${TOGGLE} -ml-2 rounded-r-md border border-l-0 border-neutral-300 peer-checked/agents:bg-neutral-200 peer-checked/agents:font-medium peer-checked/agents:text-neutral-950 peer-focus-visible/agents:ring-2 peer-focus-visible/agents:ring-accent dark:border-neutral-700 dark:peer-checked/agents:bg-neutral-800 dark:peer-checked/agents:text-white`}
        >
          Agents
        </label>
        <Row
          group={GROUPS[0]}
          current={current}
          className={`${ROW} peer-checked/llm:inline-flex`}
        />
        <Row
          group={GROUPS[1]}
          current={current}
          className={`${ROW} peer-checked/agents:inline-flex`}
        />
      </div>
      {/* below lg: one dropdown with both groups */}
      <details
        className="relative text-xs lg:hidden"
        data-site-switch="compact"
      >
        <summary className="focus-ring flex min-h-11 cursor-pointer list-none items-center gap-1 rounded-md border border-neutral-300 px-3 dark:border-neutral-700">
          <span className="text-neutral-500 dark:text-neutral-400">Sites:</span>
          <span className="font-medium">{here.label}</span>
          <span aria-hidden>▾</span>
        </summary>
        <div className="absolute right-0 z-20 mt-1 flex max-h-[70vh] w-52 flex-col overflow-y-auto rounded-md border border-neutral-300 bg-white p-1 shadow-lg dark:border-neutral-700 dark:bg-neutral-900">
          {GROUPS.map((g) => (
            <nav
              key={g.key}
              aria-label={`Companion sites: ${g.label}`}
              className="flex flex-col"
            >
              <p className="px-2.5 pb-1 pt-2 font-mono text-[0.65rem] uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
                {g.label}
              </p>
              {SITES.filter((s) => s.group === g.key).map((s) => (
                <Item key={s.key} s={s} current={current} block />
              ))}
            </nav>
          ))}
        </div>
      </details>
    </>
  );
}
