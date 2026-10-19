/**
 * /hardware — the Hardware track's start page (brief 44): the four sites in
 * reading order, what to know first, how a chapter works (layers,
 * self-checks, what to remember), and a glossary whose terms link to the
 * chapter section that defines each one, on whichever site defines it.
 * Hosted here because Kernels is the track's first site. Server Component,
 * no client JavaScript.
 */
import { ReadingOrder } from "@/components/learn/ReadingOrder";
import {
  FOUNDATIONS,
  HARDWARE_GLOSSARY as GLOSSARY,
  HARDWARE_SITES,
} from "@/lib/learn/hardware";
import { withShare } from "@/lib/share";
import { CUDA_HUB, NVIDIA_GPU_HUB } from "@/lib/site";

export const metadata = withShare("/hardware", {
  title: "Hardware: start here",
  description:
    "The Hardware track's start page: Kernels, Systolic, Accelerators and Racks in reading order, what you need to know first, and a glossary of the terms they define.",
});

const LINK =
  "focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300";

export default function HardwarePage(): JSX.Element {
  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        Hardware track
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        Start here: the machines that run the models
      </h1>
      <p className="mt-4 text-neutral-600 dark:text-neutral-300">
        Four sites, one question: what does it take to do a transformer&rsquo;s
        arithmetic fast? Kernels follows the maths through one GPU; Systolic
        builds the other kind of matrix engine, register by register;
        Accelerators runs the same decoder on every family of AI chip; Racks
        joins the chips into a data centre. Read them in order: each assumes the
        one before.
      </p>

      <ReadingOrder
        track="Hardware"
        intro="Each site's chapters build on each other; start at chapter 1 of Kernels. Every number is computed by the site's tested model."
        sites={HARDWARE_SITES}
        current="kernels"
      />

      <section aria-labelledby="prereq-heading" className="mt-12">
        <h2
          id="prereq-heading"
          className="text-xl font-semibold tracking-tight"
        >
          What you need first
        </h2>
        <ul className="mt-4 list-disc space-y-2 pl-6 text-neutral-700 dark:text-neutral-300">
          <li>
            <strong>What a transformer computes.</strong> Matrix multiplies of
            activations by weights, attention, and decoding one token at a time.
            The{" "}
            <a href={FOUNDATIONS} className={LINK}>
              Foundations track
            </a>{" "}
            (Decoder and Training) covers it.
          </li>
          <li>
            <strong>Bytes, bandwidth and latency.</strong> That a number takes
            1, 2 or 4 bytes, that bandwidth is bytes per second and latency the
            wait before the first one arrives.
          </li>
          <li>
            <strong>A little computer architecture.</strong> Caches, DRAM and
            the idea that memory close to the arithmetic is small and fast. The{" "}
            <a href={NVIDIA_GPU_HUB} className={LINK}>
              NVIDIA GPU slide series
            </a>{" "}
            starts from there.
          </li>
          <li>
            <strong>Reading CUDA or RTL is optional.</strong> The Code layer
            shows the model&rsquo;s TypeScript and, where it helps, CUDA or
            SystemVerilog (the{" "}
            <a href={CUDA_HUB} className={LINK}>
              CUDA slide series
            </a>{" "}
            teaches it); hide it with the layer switch.
          </li>
        </ul>
      </section>

      <section aria-labelledby="how-heading" className="mt-12">
        <h2 id="how-heading" className="text-xl font-semibold tracking-tight">
          How a chapter works
        </h2>
        <p className="mt-4 text-neutral-700 dark:text-neutral-300">
          A chapter opens with an animation of its mechanism, computed by the
          site&rsquo;s tested model, then explains it in three layers you can
          switch on and off: Concept (words and pictures), Maths (the formulas)
          and Code (the code that runs). It ends with{" "}
          <strong>Check yourself</strong>, three questions to commit to before
          you reveal the answers (the numbers come from the same model), and{" "}
          <strong>What to remember</strong>, the chapter in a few lines.
        </p>
      </section>

      <section aria-labelledby="glossary-heading" className="mt-12">
        <h2
          id="glossary-heading"
          className="text-xl font-semibold tracking-tight"
        >
          Glossary
        </h2>
        <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-400">
          Each term links to the section that defines it, on the site that
          defines it. Some terms mean different things on different sites
          (goodput, ρ, the pipeline bubble); the{" "}
          <a
            href="https://agent-harnesses-explained.vercel.app/definitions"
            className={LINK}
            data-definitions-link
          >
            shared definitions
          </a>{" "}
          give each one definition and one formula, and say where a site
          differs.
        </p>
        <dl
          data-track-glossary
          className="mt-4 divide-y divide-neutral-200 text-sm dark:divide-neutral-800"
        >
          {GLOSSARY.map((g) => (
            <div key={g.term} className="py-2 sm:flex sm:gap-4">
              <dt className="font-medium sm:w-56 sm:shrink-0">
                <a href={g.href} className={LINK}>
                  {g.term}
                </a>
              </dt>
              <dd className="text-neutral-600 dark:text-neutral-400">
                {g.what}
              </dd>
            </div>
          ))}
        </dl>
      </section>
    </main>
  );
}
