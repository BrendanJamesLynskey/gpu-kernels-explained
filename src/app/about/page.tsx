/**
 * /about: what the site is, what the model computes and what it leaves
 * out, how the animations are driven and checked, and where the design
 * came from. Server Component, static.
 */
import Link from "next/link";

import {
  ARCHITECTURES_URL,
  NUMERICS_URL,
  CUDA_HUB,
  DECODER_URL,
  GITHUB_URL,
  INFERENCE_URL,
  NVIDIA_GPU_HUB,
  repoFile,
} from "@/lib/site";

export const metadata = {
  title: "About",
  description:
    "What GPU Kernels Explained's execution model computes, what is illustrative, and how the model and its animations are checked.",
};

const A =
  "focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300";

export default function AboutPage(): JSX.Element {
  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        /about
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        About this site
      </h1>
      <div className="mdx-content mt-6">
        <p>
          GPU Kernels Explained shows how a GPU executes the maths behind modern
          models, one mechanism per chapter, each around an animation. It is the
          fourth of a family of companion sites: the{" "}
          <a href={DECODER_URL} className={A}>
            Transformer Decoder Explainer
          </a>
          ,{" "}
          <a href={INFERENCE_URL} className={A}>
            LLM Inference Explained
          </a>
          ,{" "}
          <a href={ARCHITECTURES_URL} className={A}>
            LLM Architectures Explained
          </a>{" "}
          and{" "}
          <a href={NUMERICS_URL} className={A}>
            Numerics Explained
          </a>
          . Each chapter links the matching slides of the{" "}
          <a href={NVIDIA_GPU_HUB} className={A}>
            NVIDIA GPU
          </a>{" "}
          and{" "}
          <a href={CUDA_HUB} className={A}>
            CUDA
          </a>{" "}
          series.
        </p>

        <h2>The execution model</h2>
        <p>
          <a href={repoFile("reference/gpu_model.py")} className={A}>
            reference/gpu_model.py
          </a>{" "}
          is an <strong>illustrative, parameterised</strong> model of a GPU: SM
          count, clock, FP32 and tensor peaks, warps, registers, shared memory
          with 32 banks, L2 and HBM bandwidths. For a kernel configuration it
          computes the bytes moved at each memory level, arithmetic intensity
          and the roofline position, occupancy, bank conflicts, coalescing
          (32-byte sectors per warp request), the active mask of every
          instruction a diverging warp issues, and a load / compute / store
          timeline per tile; and, for the later chapters, a GEMM four ways, the
          steps of four reductions, online softmax and FlashAttention&apos;s
          memory traffic, split-K and quantised layers. It is not
          cycle-accurate.
        </p>
        <ul>
          <li>
            Every preset figure has a source and a status (see{" "}
            <Link href="/gpus" className={A}>
              the GPU presets
            </Link>
            ).
          </li>
          <li>
            Kernel time is a hierarchical roofline: each memory level and the
            ALUs work in parallel at peak, and the slowest sets the time.
            Latency, instruction issue and cache behaviour beyond a simple
            fits-in-L2 rule are not modelled.
          </li>
          <li>
            Occupancy follows the CUDA Toolkit&apos;s{" "}
            <code>cuda_occupancy.h</code> for compute capability 8.0 and 9.0,
            with the shared-memory carveout at its maximum.
          </li>
          <li>
            Divergence follows the classic SIMT order (if-path, else-path,
            reconverge); since Volta the scheduler may order the paths
            differently, at the same cost.
          </li>
        </ul>

        <h2>How it is checked</h2>
        <ul>
          <li>
            <strong>Closed forms.</strong>{" "}
            <a href={repoFile("tests/python/test_gpu_model.py")} className={A}>
              tests/python/test_gpu_model.py
            </a>{" "}
            checks the reference against its sources&apos; worked examples: the
            CUDA Programming Guide&apos;s 12.5% coalescing worst case, two-way
            and 32-way bank conflicts and the padding fix, the 75% and 50%
            occupancy examples, and the GEMM traffic formulas.
          </li>
          <li>
            <strong>Exact parity.</strong> The TypeScript port,{" "}
            <a href={repoFile("src/lib/gpu/model.ts")} className={A}>
              src/lib/gpu/model.ts
            </a>
            , repeats every function with the same operations in the same order;{" "}
            <a href={repoFile("scripts/make_fixtures.py")} className={A}>
              scripts/make_fixtures.py
            </a>{" "}
            writes the reference&apos;s results over grids of every parameter,
            and the unit tests require the port to reproduce all of them
            exactly, with no tolerance (the online softmax, which calls{" "}
            <code>exp</code>, to a relative 10<sup>−14</sup>, because
            JavaScript&apos;s and the C library&apos;s <code>exp</code> can
            differ in the last bit). CI fails if the fixtures are out of date.
          </li>
          <li>
            <strong>Animations from the model.</strong> Every animation draws a
            sequence of states the model computes; a frame is a pure function of
            one state. The end-to-end tests set chosen frames of every animation
            and require the caption to match the caption built from the Python
            reference&apos;s state for that frame.
          </li>
          <li>
            <strong>Numbers in the prose</strong> are printed from the model
            when the page is built, not typed.
          </li>
        </ul>

        <h2>The animations</h2>
        <p>
          Every animation has play and pause, step back and forward, a scrub
          bar, speeds from 0.25× to 4× and reset; with the animation focused,
          Space plays or pauses and the arrow keys step. Each step has a
          one-line caption, also announced to screen readers. With{" "}
          <em>reduce motion</em> set in your system, nothing plays by itself.
          Animations pause when scrolled out of view. Colours come from Okabe
          and Ito&apos;s colour-blind-safe palette, one colour per memory level
          (registers orange, shared memory green, L2 sky blue, HBM purple), the
          same in light and dark mode; a stall is always a hatched pattern as
          well as a warning colour.
        </p>

        <h2>Source</h2>
        <p>
          The code, the model and the tests are on{" "}
          <a href={GITHUB_URL} className={A}>
            GitHub
          </a>{" "}
          (MIT licence). The design system is copied from the companion sites;
          the README records where each piece came from.
        </p>
      </div>
    </main>
  );
}
