import type { Route } from "next";
import Link from "next/link";

import { CheckedStrip, StartButton } from "@/components/ui/Landing";
import { RidgeLadder } from "@/components/viz/RidgeLadder";
import { SECTIONS } from "@/lib/mdx/sections";
import { formatValue, lookup } from "@/lib/gpu/values";
import { HardwareWork } from "@/components/home/HardwareWork";

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
      <div className="mt-6 grid items-start gap-8 md:grid-cols-[1fr_minmax(0,22rem)]">
        <div>
          <p className="max-w-2xl text-lg text-neutral-600 dark:text-neutral-300">
            An interactive guide to how a GPU runs kernels, from the A100 to
            Blackwell: the memory hierarchy, warps, coalescing, occupancy, GEMM,
            FlashAttention and the warp-specialised tensor-core pipeline, each
            animated by a tested execution model.
          </p>
          <div className="mt-6">
            <StartButton href={`/learn/${SECTIONS[0]!.slug}` as Route} />
          </div>
        </div>
        <figure
          data-hero-figure
          className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-800"
        >
          <RidgeLadder />
          <figcaption className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
            Flops of tensor-core work each byte from HBM must feed before the
            GPU stops waiting on memory (dense peak over HBM bandwidth), to
            scale. Each generation, and each narrower format, moves it right.
          </figcaption>
        </figure>
      </div>
      <CheckedStrip href="/about#checked">
        The GPU model is checked against the CUDA Programming Guide&apos;s
        worked examples (coalescing, bank conflicts, occupancy), and its
        TypeScript port reproduces the Python reference exactly.
      </CheckedStrip>
      <section aria-labelledby="learn-heading" className="mt-12">
        <h2 id="learn-heading" className="text-xl font-semibold tracking-tight">
          What you&apos;ll learn
        </h2>
        <p className="mt-4 max-w-2xl text-neutral-600 dark:text-neutral-300">
          A GPU can do arithmetic far faster than it can fetch the numbers to do
          it on. A B200&apos;s tensor cores need{" "}
          {v("tc.b200.ridge.bf16", "num")} flops of BF16 work on every byte from
          its HBM just to keep busy ({v("tc.b200.ridge.fp4", "num")} at NVFP4);
          an H100&apos;s needed {v("tc.h100.ridge.bf16", "num")} and an
          A100&apos;s {v("tc.a100.ridge.bf16", "num")}. Every fast kernel is a
          way of closing that gap: reuse data close to the ALUs, fetch it in
          whole transactions, avoid serialising on shared memory, keep the
          tensor cores fed from a pipeline of loads.{" "}
          <Link
            href={"/learn/12-hopper-and-blackwell-gemm" as Route}
            className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
          >
            Chapter 12
          </Link>{" "}
          shows how Hopper and Blackwell kernels do it.
        </p>
        <p className="mt-4 max-w-2xl text-neutral-600 dark:text-neutral-300">
          Each chapter here is built around an animation, and every frame is
          computed by a small GPU execution model that is tested against a
          Python reference. The hardware figures come from NVIDIA&apos;s
          whitepapers and datasheets, the CUDA Programming Guide and published
          microbenchmarks.
        </p>
      </section>
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
      <p
        data-track-start
        className="mt-12 text-sm text-neutral-600 dark:text-neutral-400"
      >
        New to the Hardware track?{" "}
        <a
          href={"/hardware"}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Start here
        </a>
        : the four sites in reading order, what you need first, and a glossary
        of the terms they define.
      </p>
      <HardwareWork />
      <p className="mt-12 text-sm text-neutral-600 dark:text-neutral-400">
        Part of a family of sixteen companion sites in four tracks, all in the
        site menu above: Foundations (
        <a
          href="https://transformer-decoder-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Decoder
        </a>
        ,{" "}
        <a
          href="https://transformer-training-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Training
        </a>
        ), Hardware (<strong className="font-medium">Kernels</strong>,{" "}
        <a
          href="https://systolic-arrays-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Systolic
        </a>
        ,{" "}
        <a
          href="https://ai-accelerators-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Accelerators
        </a>
        ,{" "}
        <a
          href="https://ai-racks-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Racks
        </a>
        ), Inference (
        <a
          href="https://llm-inference-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Serving
        </a>
        ,{" "}
        <a
          href="https://llm-architectures-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Architectures
        </a>
        ,{" "}
        <a
          href="https://numerics-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Numerics
        </a>
        ,{" "}
        <a
          href="https://inference-tradeoffs-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Trade-offs
        </a>
        ) and Agents (
        <a
          href="https://agent-harnesses-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Harnesses
        </a>
        ,{" "}
        <a
          href="https://agent-protocols-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Protocols
        </a>
        ,{" "}
        <a
          href="https://agent-context-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Context
        </a>
        ,{" "}
        <a
          href="https://agent-orchestration-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Orchestration
        </a>
        ,{" "}
        <a
          href="https://agent-evals-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Evals
        </a>
        ,{" "}
        <a
          href="https://agent-security-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Security
        </a>
        ). This site is the layer underneath: the kernels. How it was built, and
        how to check it:{" "}
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
