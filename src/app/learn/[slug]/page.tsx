/**
 * /learn/[slug] — render an MDX chapter with the three-layer toggle pinned
 * at the top.
 *
 * Server Component, statically generated for every chapter. Copied from
 * LLM Inference Explained (itself transformer-explainer's lesson page minus
 * comments and progress):
 * the slug is validated against the catalogue, the MDX is read from disk
 * and handed to `next-mdx-remote/rsc` with our components map.
 *
 * Maths: `remark-math` + `rehype-katex` render `$…$` / `$$…$$` to HTML on
 * the server, so no KaTeX JavaScript ships; only its stylesheet, and only
 * on these pages. `remark-gfm` adds Markdown tables.
 */
import Link from "next/link";
import { notFound } from "next/navigation";
import { MDXRemote } from "next-mdx-remote/rsc";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import { LayerToggle } from "@/components/interactive/LayerToggle";
import { mdxComponents } from "@/lib/mdx/components";
import {
  SECTIONS,
  getSectionMeta,
  isValidSlug,
  readSectionMdx,
} from "@/lib/mdx/sections";

import "katex/dist/katex.min.css";

type HastNode = {
  type: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
};

/**
 * Display equations scroll sideways on a phone; a scrollable region must be
 * reachable by keyboard (axe: scrollable-region-focusable), so every
 * `.katex-display` gets tabIndex 0. Runs after rehype-katex.
 */
function rehypeFocusableMath() {
  const walk = (n: HastNode): void => {
    const cls = n.properties?.className;
    if (
      n.type === "element" &&
      Array.isArray(cls) &&
      cls.includes("katex-display")
    )
      n.properties = { ...n.properties, tabIndex: 0 };
    n.children?.forEach(walk);
  };
  return (tree: HastNode) => walk(tree);
}

export const dynamicParams = false;

export async function generateStaticParams(): Promise<{ slug: string }[]> {
  return SECTIONS.map((s) => ({ slug: s.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: { slug: string };
}) {
  if (!isValidSlug(params.slug)) return {};
  const meta = getSectionMeta(params.slug);
  return {
    title: meta.title,
    description: meta.summary,
  };
}

export default async function SectionPage({
  params,
}: {
  params: { slug: string };
}): Promise<JSX.Element> {
  if (!isValidSlug(params.slug)) notFound();
  const meta = getSectionMeta(params.slug);
  const mdx = await readSectionMdx(params.slug);
  if (mdx === null) notFound();

  const idx = SECTIONS.findIndex((s) => s.slug === params.slug);
  const prev = idx > 0 ? SECTIONS[idx - 1] : null;
  const next = idx < SECTIONS.length - 1 ? SECTIONS[idx + 1] : null;

  return (
    <article className="mx-auto max-w-3xl px-6 py-10">
      <div className="flex flex-wrap items-baseline justify-between gap-4">
        <div>
          <Link
            href="/learn"
            className="focus-ring rounded font-mono text-xs uppercase tracking-widest text-accent hover:underline dark:text-indigo-300"
          >
            ← /learn · {String(idx + 1).padStart(2, "0")}
          </Link>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight">
            {meta.title}
          </h1>
          <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
            {meta.summary}
          </p>
        </div>
        <LayerToggle />
      </div>

      <div className="mdx-content mt-8">
        <MDXRemote
          source={mdx}
          components={mdxComponents}
          options={{
            mdxOptions: {
              remarkPlugins: [remarkGfm, remarkMath],
              rehypePlugins: [rehypeKatex, rehypeFocusableMath],
            },
          }}
        />
      </div>

      <nav
        aria-label="Chapters"
        className="mt-12 flex items-center justify-between gap-4 border-t border-neutral-200 pt-6 text-sm dark:border-neutral-800"
      >
        {prev ? (
          <Link
            href={`/learn/${prev.slug}`}
            className="focus-ring rounded text-neutral-600 hover:text-accent dark:text-neutral-400"
          >
            ← {prev.title}
          </Link>
        ) : (
          <span aria-hidden />
        )}
        {next ? (
          <Link
            href={`/learn/${next.slug}`}
            className="focus-ring rounded text-right text-neutral-600 hover:text-accent dark:text-neutral-400"
          >
            {next.title} →
          </Link>
        ) : (
          <span aria-hidden />
        )}
      </nav>
    </article>
  );
}
