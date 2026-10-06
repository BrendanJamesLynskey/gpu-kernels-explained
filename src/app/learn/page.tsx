/**
 * /learn — index of chapters.
 *
 * Server Component, statically rendered. Same layout as the companion
 * sites' /learn (transformer-explainer's, minus the per-user progress
 * badges: this site has no accounts).
 */
import Link from "next/link";

import { SECTIONS } from "@/lib/mdx/sections";
import { ARCHITECTURES_URL, CUDA_HUB, NVIDIA_GPU_HUB } from "@/lib/site";

export const metadata = {
  title: "Learn",
  description:
    "Chapters on how a GPU executes a kernel, each built around an animation driven by the tested execution model.",
};

export default function LearnIndex(): JSX.Element {
  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        /learn
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        How a GPU runs a kernel
      </h1>
      <p className="mt-4 text-neutral-600 dark:text-neutral-300">
        One chapter per mechanism, each opening with an animation. Every frame
        is computed by this site&apos;s GPU execution model (checked against its
        Python reference), and every hardware figure comes from a published
        source. Toggle layers (Concept / Maths / Code) inside any chapter to
        choose how deep to go. The model architectures these kernels serve are
        on{" "}
        <a
          href={ARCHITECTURES_URL}
          className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
        >
          LLM Architectures Explained
        </a>
        ; for the slides behind each chapter, see the{" "}
        <a
          href={NVIDIA_GPU_HUB}
          className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
        >
          NVIDIA GPU
        </a>{" "}
        and{" "}
        <a
          href={CUDA_HUB}
          className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
        >
          CUDA
        </a>{" "}
        series.
      </p>

      <ol className="mt-10 divide-y divide-neutral-200 dark:divide-neutral-800">
        {SECTIONS.map((s, i) => (
          <li key={s.slug} className="py-5">
            <div className="flex items-baseline gap-3">
              <span className="font-mono text-xs text-neutral-500 dark:text-neutral-400">
                {String(i + 1).padStart(2, "0")}
              </span>
              <Link
                href={`/learn/${s.slug}`}
                className="focus-ring rounded text-lg font-medium text-neutral-900 hover:text-accent dark:text-neutral-100"
              >
                {s.title}
              </Link>
            </div>
            <p className="mt-1 pl-9 text-sm text-neutral-600 dark:text-neutral-400">
              {s.summary}
            </p>
          </li>
        ))}
      </ol>
    </main>
  );
}
