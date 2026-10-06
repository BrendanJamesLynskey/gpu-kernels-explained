/**
 * /gpus: every figure the model uses, per preset, with its status and its
 * source, and the figures derived from them with their formulas.
 * Server Component, static.
 */
import { BandwidthLadder } from "@/components/viz/BandwidthLadder";
import { fmtAi, fmtBytes, fmtFlops, fmtKiB, fmtRate, trim } from "@/lib/format";
import {
  COMMON,
  PRESET_RECORDS,
  SOURCES,
  derived,
  preset,
  type Field,
  type PresetId,
  type Status,
} from "@/lib/gpu/model";

export const metadata = {
  title: "GPU presets",
  description:
    "The A100 and H100 figures behind every animation, each with its source and status, and the bandwidths and ridge points derived from them.",
};

const LABELS: Record<string, [string, (v: number) => string]> = {
  sms: ["Streaming multiprocessors (SMs)", (v) => String(v)],
  clock_mhz: ["Boost clock", (v) => `${trim(v, 5)} MHz`],
  fp32_cores_per_sm: ["FP32 cores per SM", (v) => String(v)],
  peak_fp32: ["Peak FP32 (CUDA cores)", fmtFlops],
  peak_tensor: ["Peak BF16 tensor (dense)", fmtFlops],
  hbm_bw: ["HBM bandwidth", fmtRate],
  hbm_bytes: ["HBM capacity", fmtBytes],
  l2_bytes: ["L2 cache", fmtKiB],
  l2_bytes_per_clk: ["L2 bandwidth per clock", (v) => `${v} B/clk`],
  smem_per_sm: ["Shared memory per SM (max)", fmtKiB],
  smem_per_block: ["Shared memory per block (max)", fmtKiB],
  regs_per_sm: ["32-bit registers per SM", (v) => v.toLocaleString("en-GB")],
  regs_per_block: [
    "Registers per block (max)",
    (v) => v.toLocaleString("en-GB"),
  ],
  max_regs_per_thread: ["Registers per thread (max)", (v) => String(v)],
  max_warps_per_sm: ["Warps per SM (max)", (v) => String(v)],
  max_blocks_per_sm: ["Blocks per SM (max)", (v) => String(v)],
  max_threads_per_block: ["Threads per block (max)", (v) => String(v)],
  lat_smem: ["Shared-memory latency", (v) => `${v} cycles`],
  lat_l2: ["L2 latency", (v) => `${v} cycles`],
  lat_hbm: ["HBM (global) latency", (v) => `${v} cycles`],
  warp_size: ["Threads per warp", (v) => String(v)],
  smem_banks: ["Shared-memory banks", (v) => String(v)],
  bank_bytes: ["Bank width", (v) => `${v} bytes per clock`],
  sector_bytes: ["Global-memory transaction (sector)", (v) => `${v} bytes`],
  reg_alloc_unit: [
    "Register allocation unit (per warp)",
    (v) => `${v} registers`,
  ],
  smem_alloc_unit: ["Shared-memory allocation unit", (v) => `${v} bytes`],
  smem_reserved: ["Shared memory reserved per block", (v) => `${v} bytes`],
  sub_partitions: ["SM sub-partitions", (v) => String(v)],
};

/** Fields in the order of LABELS (the JSON is key-sorted), then any others. */
function ordered(fields: Record<string, Field>): [string, Field][] {
  const keys = Object.keys(LABELS).filter((k) => k in fields);
  for (const k of Object.keys(fields)) if (!keys.includes(k)) keys.push(k);
  return keys.map((k) => [k, fields[k] as Field]);
}

const STATUS_STYLE: Record<Status, string> = {
  spec: "bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200",
  rule: "bg-sky-100 text-sky-900 dark:bg-sky-950 dark:text-sky-200",
  measured:
    "bg-violet-100 text-violet-900 dark:bg-violet-950 dark:text-violet-200",
  derived:
    "bg-neutral-200 text-neutral-900 dark:bg-neutral-800 dark:text-neutral-100",
  approximation:
    "border border-dashed border-amber-600 bg-amber-50 text-amber-900 dark:bg-amber-950 dark:text-amber-200",
};

const A =
  "focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300";

function Row({ k, f }: { k: string; f: Field }): JSX.Element {
  const [label, fmt] = LABELS[k] ?? [k, (v: number) => String(v)];
  const src = SOURCES[f.src];
  return (
    <tr>
      <td>{label}</td>
      <td className="whitespace-nowrap font-mono">
        {typeof f.v === "number" ? fmt(f.v) : f.v}
      </td>
      <td>
        <span
          data-status={f.st}
          className={`rounded px-1.5 py-0.5 font-mono text-[0.7rem] ${STATUS_STYLE[f.st]}`}
        >
          {f.st}
        </span>
      </td>
      <td className="text-xs">
        {src ? (
          <a href={src.url} className={A}>
            {f.src}
          </a>
        ) : (
          f.src
        )}
        : {f.ref}
      </td>
    </tr>
  );
}

function Table({ children }: { children: React.ReactNode }): JSX.Element {
  return (
    <div
      role="region"
      aria-label="Table"
      tabIndex={0}
      className="focus-ring my-4 max-w-full overflow-x-auto rounded"
    >
      <table className="w-full text-sm [&_td]:border-b [&_td]:border-neutral-200 [&_td]:px-2 [&_td]:py-1.5 [&_td]:align-top dark:[&_td]:border-neutral-800 [&_th]:border-b [&_th]:border-neutral-300 [&_th]:px-2 [&_th]:py-1.5 [&_th]:text-left dark:[&_th]:border-neutral-700">
        {children}
      </table>
    </div>
  );
}

export default function GpusPage(): JSX.Element {
  return (
    <main className="mx-auto max-w-4xl px-6 py-12">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        /gpus
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        The GPU presets
      </h1>
      <p className="mt-4 text-neutral-600 dark:text-neutral-300">
        The model is parameterised: any GPU is a set of the figures below. Two
        presets ship, each figure with a status. <strong>spec</strong>: NVIDIA
        states it. <strong>rule</strong>: the CUDA documentation or toolkit
        states it. <strong>measured</strong>: a published microbenchmark.{" "}
        <strong>derived</strong>: computed from other figures, with the formula.{" "}
        <strong>approximation</strong>: a stand-in, named as such. The presets
        are approximations of the real parts and the model is not
        cycle-accurate; the figures are what the animations compute from.
      </p>

      {PRESET_RECORDS.map((r) => {
        const pid = r.id as PresetId;
        const d = derived(preset(pid));
        return (
          <section key={r.id} className="mt-12" data-preset={r.id}>
            <h2 className="text-2xl font-semibold tracking-tight">{r.name}</h2>
            <p className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
              {r.arch}
            </p>
            <div className="mt-4 max-w-md">
              <BandwidthLadder pid={pid} />
            </div>
            <Table>
              <thead>
                <tr>
                  <th>Figure</th>
                  <th>Value</th>
                  <th>Status</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {ordered(r.fields).map(([k, f]) => (
                  <Row key={k} k={k} f={f} />
                ))}
              </tbody>
            </Table>
            <h3 className="mt-6 font-semibold">Derived</h3>
            <Table>
              <thead>
                <tr>
                  <th>Figure</th>
                  <th>Value</th>
                  <th>Formula</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Register bandwidth</td>
                  <td className="font-mono">{fmtRate(d.bw.reg)}</td>
                  <td>
                    6 bytes per flop × peak FP32 (three 4-byte operands per
                    2-flop FFMA)
                  </td>
                </tr>
                <tr>
                  <td>Shared-memory bandwidth</td>
                  <td className="font-mono">{fmtRate(d.bw.smem)}</td>
                  <td>SMs × 32 banks × 4 bytes × clock</td>
                </tr>
                <tr>
                  <td>L2 bandwidth</td>
                  <td className="font-mono">{fmtRate(d.bw.l2)}</td>
                  <td>L2 bytes per clock × clock</td>
                </tr>
                <tr>
                  <td>FP32 ridge point</td>
                  <td className="font-mono">{fmtAi(d.ridge_fp32)}</td>
                  <td>peak FP32 / HBM bandwidth</td>
                </tr>
                <tr>
                  <td>BF16 tensor ridge point</td>
                  <td className="font-mono">{fmtAi(d.ridge_tensor)}</td>
                  <td>peak tensor / HBM bandwidth</td>
                </tr>
              </tbody>
            </Table>
          </section>
        );
      })}

      <section className="mt-12">
        <h2 className="text-2xl font-semibold tracking-tight">
          Rules shared by both
        </h2>
        <Table>
          <thead>
            <tr>
              <th>Figure</th>
              <th>Value</th>
              <th>Status</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {ordered(COMMON).map(([k, f]) => (
              <Row key={k} k={k} f={f} />
            ))}
          </tbody>
        </Table>
      </section>

      <section className="mt-12">
        <h2 className="text-2xl font-semibold tracking-tight">Sources</h2>
        <ul className="mt-4 list-disc space-y-2 pl-6 text-sm">
          {Object.entries(SOURCES).map(([id, s]) => (
            <li key={id}>
              <span className="font-mono text-xs">{id}</span>:{" "}
              <a href={s.url} className={A}>
                {s.title}
              </a>
            </li>
          ))}
        </ul>
        <p className="mt-4 text-sm text-neutral-600 dark:text-neutral-400">
          The H100 whitepaper was published before launch and lists its
          specifications as preliminary; the H100 figures that changed (FP32 and
          tensor peaks, HBM bandwidth) are taken from NVIDIA&apos;s current
          product page instead, and its clock is derived from the FP32 peak.
          NVIDIA does not publish H100&apos;s L2 bandwidth, so the model uses
          the per-clock rate Luo et al. measured on an H800 (the same GH100
          die). Luo et al. measured about 2,008 bytes per clock on an A100
          against NVIDIA&apos;s stated 5,120, so the A100&apos;s L2 figure is a
          peak the model treats as attainable.
        </p>
      </section>
    </main>
  );
}
