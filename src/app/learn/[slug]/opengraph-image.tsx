/**
 * Each chapter's link-preview card: the chapter's number and title over the site's card
 * (see ../../_og/card.tsx), one per chapter at build time.
 */
import { card, OG_SIZE, OG_TYPE } from "../../_og/card";
import { SECTIONS } from "@/lib/mdx/sections";

export const alt = "A chapter of GPU Kernels Explained";
export const size = OG_SIZE;
export const contentType = OG_TYPE;

export function generateStaticParams(): { slug: string }[] {
  return SECTIONS.map((s) => ({ slug: s.slug }));
}

export default function Image({ params }: { params: { slug: string } }) {
  const i = Math.max(
    SECTIONS.findIndex((s) => s.slug === params.slug),
    0,
  );
  const s = SECTIONS[i]!;
  const n = Number.parseInt(s.slug, 10);
  return card({ number: Number.isNaN(n) ? i + 1 : n, title: s.title });
}
