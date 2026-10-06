import Link from "next/link";

import { BandwidthLadder } from "@/components/viz/BandwidthLadder";
import { SECTIONS } from "@/lib/mdx/sections";
import {
  ARCHITECTURES_URL,
  DECODER_URL,
  INFERENCE_URL,
  NUMERICS_URL,
} from "@/lib/site";
import { formatValue, lookup } from "@/lib/gpu/values";

/**
 * Landing page: what the site is, the picture behind every chapter (the
 * bandwidth ladder, to scale), and the ways in. Server Component with no
 * client JavaScript of its own.
 */
export default function HomePage(): JSX.Element {
  const v = (path: string, fmt: Parameters<typeof formatValue>[1]) =>
    formatValue(lookup(path), fmt);
  return (
    <main className="mx-auto max-w-5xl px-6 py-12 sm:py-20">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        GPU Kernels Explained
      </p>
      <h1 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
        How a GPU actually executes the maths.
      </h1>
      <div className="mt-8 grid items-center gap-8 md:grid-cols-[1fr_minmax(0,22rem)]">
        <div>
          <p className="max-w-2xl text-lg text-neutral-600 dark:text-neutral-300">
            A GPU can do arithmetic far faster than it can fetch the numbers to
            do it on. An A100&apos;s FP32 units need{" "}
            {v("a100.ridge_fp32", "num")} flops of work on every byte from its
            HBM just to keep busy. Every fast kernel is a way of closing that
            gap: reuse data close to the ALUs, fetch it in whole transactions,
            avoid serialising on shared memory, keep enough warps in flight.
          </p>
          <p className="mt-4 max-w-2xl text-neutral-600 dark:text-neutral-300">
            Each chapter here is built around an animation, and every frame is
            computed by a small GPU execution model that is tested against a
            Python reference. The hardware figures come from NVIDIA&apos;s
            whitepapers and datasheets, the CUDA Programming Guide and published
            microbenchmarks.
          </p>
        </div>
        <figure className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
          <BandwidthLadder pid="a100" />
          <figcaption className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
            The A100&apos;s memory levels, bandwidth to scale. HBM is the sliver
            at the bottom.
          </figcaption>
        </figure>
      </div>
      <nav
        aria-label="Chapters"
        className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
      >
        {SECTIONS.map((s, i) => (
          <Link
            key={s.slug}
            href={`/learn/${s.slug}`}
            className="focus-ring group rounded-lg border border-neutral-200 p-5 hover:border-accent dark:border-neutral-800 dark:hover:border-indigo-400"
          >
            <p className="font-mono text-xs text-neutral-500 dark:text-neutral-400">
              {String(i + 1).padStart(2, "0")}
            </p>
            <h2 className="mt-1 font-semibold">{s.title}</h2>
            <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-400">
              {s.summary}
            </p>
          </Link>
        ))}
      </nav>
      <p className="mt-6 text-sm text-neutral-600 dark:text-neutral-400">
        Every number on the GPUs the model uses, with its source:{" "}
        <Link
          href="/gpus"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          the GPU presets
        </Link>
        .
      </p>
      <p className="mt-12 text-sm text-neutral-600 dark:text-neutral-400">
        Part of a family of companion sites: the{" "}
        <a
          href={DECODER_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Transformer Decoder Explainer
        </a>{" "}
        (one forward pass),{" "}
        <a
          href={INFERENCE_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          LLM Inference Explained
        </a>{" "}
        (serving it),{" "}
        <a
          href={ARCHITECTURES_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          LLM Architectures Explained
        </a>{" "}
        (how the models differ) and{" "}
        <a
          href={NUMERICS_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Numerics Explained
        </a>{" "}
        (the numbers themselves). This site is the layer underneath: the
        kernels. How it was built, and how to check it:{" "}
        <Link
          href="/about"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          about
        </Link>
        .
      </p>
    </main>
  );
}
