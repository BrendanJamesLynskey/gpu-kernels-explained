/**
 * Chapter catalogue + filesystem loader for /learn content.
 *
 * MDX sources live under `/content/chapters/`, one per mechanism, each built
 * around an animation. Their slugs and order are defined here (single source
 * of truth); the `[slug]` route validates incoming params against this list
 * before reading from disk. Same shape as the companion sites'
 * `src/lib/mdx/sections.ts`.
 */
import { readFile } from "node:fs/promises";
import { join } from "node:path";

export const SECTIONS = [
  {
    slug: "01-memory-hierarchy",
    title: "The memory hierarchy",
    summary:
      "Registers, shared memory, L2 and HBM: how much each holds, how fast it moves data, and which one sets a kernel's pace.",
  },
  {
    slug: "02-roofline",
    title: "The roofline",
    summary:
      "Arithmetic intensity decides whether a kernel waits on memory or on the ALUs; the ridge point is where they meet.",
  },
  {
    slug: "03-warps-and-divergence",
    title: "Warps, SIMT and divergence",
    summary:
      "32 threads share one instruction stream; a branch they disagree on runs both ways, with lanes masked off.",
  },
  {
    slug: "04-coalescing",
    title: "Coalescing",
    summary:
      "How a warp's 32 addresses become 32-byte memory transactions, and what stride and alignment cost.",
  },
  {
    slug: "05-bank-conflicts",
    title: "Shared-memory bank conflicts",
    summary:
      "32 banks, one word each per cycle: why a column read serialises 32 ways, and how one word of padding fixes it.",
  },
  {
    slug: "06-occupancy",
    title: "Occupancy",
    summary:
      "Registers, shared memory and warp slots decide how many blocks share an SM, and so how much latency it can hide.",
  },
  {
    slug: "07-gemm",
    title: "GEMM, step by step",
    summary:
      "Naive, tiled in shared memory, register-blocked, tensor cores: each step reuses data closer to the ALUs and raises the arithmetic intensity.",
  },
  {
    slug: "08-reductions",
    title: "Reductions and warp shuffles",
    summary:
      "Summing a block's values as a tree in shared memory, three ways, then register to register with warp shuffles.",
  },
  {
    slug: "09-softmax-and-flashattention",
    title: "Softmax and FlashAttention",
    summary:
      "Online softmax keeps a running maximum and rescales; FlashAttention uses it to never write the score matrix to HBM.",
  },
  {
    slug: "10-split-k-and-overlap",
    title: "Split-K, streams and overlap",
    summary:
      "Double buffering hides copies behind compute; split-K makes enough blocks to fill the GPU when the output is small.",
  },
  {
    slug: "11-quantised-kernels",
    title: "Quantised kernels",
    summary:
      "Low-precision weights cut the bytes a decode step reads; the kernel dequantises them in registers, just before the multiply.",
  },
] as const;

export type SectionSlug = (typeof SECTIONS)[number]["slug"];

const SLUG_SET = new Set<string>(SECTIONS.map((s) => s.slug));

export function isValidSlug(slug: string): slug is SectionSlug {
  return SLUG_SET.has(slug);
}

export function getSectionMeta(slug: SectionSlug): (typeof SECTIONS)[number] {
  return SECTIONS.find((s) => s.slug === slug) ?? SECTIONS[0];
}

/** Read the raw MDX source for a chapter, or `null` if it doesn't exist. */
export async function readSectionMdx(
  slug: SectionSlug,
): Promise<string | null> {
  const path = join(process.cwd(), "content", "chapters", `${slug}.mdx`);
  try {
    return await readFile(path, "utf-8");
  } catch {
    return null;
  }
}
