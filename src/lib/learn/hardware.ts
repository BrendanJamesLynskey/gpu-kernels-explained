/**
 * The Hardware track's start page (/hardware, brief 44): the four sites in
 * reading order and a glossary whose terms link to the chapter section that
 * defines each one, on whichever of the four sites defines it.
 * tests/unit/hardware.test.ts checks that every link on this site names a
 * chapter and an anchor that exist; the other sites' anchors are checked by
 * their own learning-layer tests.
 */
import type { TrackSite } from "@/components/learn/ReadingOrder";

export const KERNELS = "https://gpu-kernels-explained.vercel.app";
export const SYSTOLIC = "https://systolic-arrays-explained.vercel.app";
export const ACCELERATORS = "https://ai-accelerators-explained.vercel.app";
export const RACKS = "https://ai-racks-explained.vercel.app";
export const FOUNDATIONS =
  "https://transformer-decoder-explained.vercel.app/foundations";

export const HARDWARE_SITES: readonly TrackSite[] = [
  {
    key: "kernels",
    name: "Kernels",
    url: `${KERNELS}/learn`,
    why: "how a GPU executes the maths, from the memory hierarchy and the roofline to a Blackwell tensor-core pipeline; assumes the Foundations track",
  },
  {
    key: "systolic",
    name: "Systolic",
    url: `${SYSTOLIC}/learn`,
    why: "the other way to build a matrix engine, cycle by cycle and checked against RTL, up to the TPU; assumes Kernels' roofline",
  },
  {
    key: "accelerators",
    name: "Accelerators",
    url: `${ACCELERATORS}/learn`,
    why: "eight families of AI chip running the same decoder, from GPUs to wafers, LPUs and in-memory compute; assumes Kernels and Systolic",
  },
  {
    key: "racks",
    name: "Racks",
    url: `${RACKS}/learn`,
    why: "from one package to a data centre: NVLink, networks, power, cooling, disaggregated serving and cost; assumes the three before it",
  },
];

export type GlossaryTerm = { term: string; href: string; what: string };

const k = (p: string) => `${KERNELS}/learn/${p}`;
const s = (p: string) => `${SYSTOLIC}/learn/${p}`;
const a = (p: string) => `${ACCELERATORS}/learn/${p}`;
const r = (p: string) => `${RACKS}/learn/${p}`;

export const HARDWARE_GLOSSARY: readonly GlossaryTerm[] = [
  {
    term: "Memory hierarchy",
    href: k("01-memory-hierarchy#k-levels"),
    what: "registers, shared memory, L2 and HBM: the closer to the ALUs, the faster and the smaller",
  },
  {
    term: "Roofline, arithmetic intensity, ridge point",
    href: k("02-roofline#k-ridge"),
    what: "attainable speed is min(peak, bandwidth × flops per byte); the ridge is peak over bandwidth",
  },
  {
    term: "Warp, SIMT, divergence",
    href: k("03-warps-and-divergence#k-diverge"),
    what: "32 threads share one instruction stream; a branch they disagree on runs both ways",
  },
  {
    term: "Coalescing, sector",
    href: k("04-coalescing#k-coalesce"),
    what: "a warp's addresses become 32-byte sectors; consecutive threads should read consecutive addresses",
  },
  {
    term: "Bank conflict",
    href: k("05-bank-conflicts#k-banks"),
    what: "different words in one of shared memory's 32 banks are served one per pass",
  },
  {
    term: "Occupancy",
    href: k("06-occupancy#k-occupancy"),
    what: "active warps over the SM's maximum, limited by warp slots, registers, shared memory or blocks",
  },
  {
    term: "Tiling, register blocking",
    href: k("07-gemm#k-four-ways"),
    what: "reusing each value from shared memory and registers so a GEMM stops waiting on memory",
  },
  {
    term: "Warp shuffle",
    href: k("08-reductions#k-tree"),
    what: "lanes of a warp read each other's registers directly, with no shared memory or barrier",
  },
  {
    term: "Online softmax, FlashAttention",
    href: k("09-softmax-and-flashattention#k-online"),
    what: "a running maximum and sum let attention stream K and V without storing the score matrix",
  },
  {
    term: "Double buffering, split-K",
    href: k("10-split-k-and-overlap#k-overlap"),
    what: "loading the next tile while computing this one; splitting k to fill every SM",
  },
  {
    term: "TMA, warp specialisation, Tensor Memory",
    href: k("12-hopper-and-blackwell-gemm#k-warp-spec"),
    what: "Hopper and Blackwell give loads, MMAs and the epilogue their own engines and warps",
  },
  {
    term: "Systolic array",
    href: s("01-why-systolic#s-kung"),
    what: "fetch each number once and pass it neighbour to neighbour every cycle",
  },
  {
    term: "Weight-stationary, skew",
    href: s("02-weight-stationary#s-machine"),
    what: "weights stay in the PEs; skewed activations meet their partial sums exactly",
  },
  {
    term: "Output- and input-stationary",
    href: s("03-output-and-input-stationary#s-three"),
    what: "the other two dataflows, named after what stays in each PE",
  },
  {
    term: "Wavefront, fill and drain, utilisation",
    href: s("04-skew-fill-drain#s-wavefront"),
    what: "the fixed overhead of each pass, and the share of MAC slots doing useful work",
  },
  {
    term: "Shadow weights",
    href: s("05-tiling#s-shadow"),
    what: "a second weight register so the next tile loads while this one streams",
  },
  {
    term: "bfloat16, FP32 accumulation",
    href: s("06-inside-a-pe#s-bf16"),
    what: "multiply narrow, accumulate wide: the product is exact, only the add rounds",
  },
  {
    term: "MXU, ICI torus, all-reduce",
    href: s("07-the-tpu#s-allreduce"),
    what: "a TPU's systolic matrix units, and its pods' torus of chip-to-chip links",
  },
  {
    term: "Row-stationary, in-memory compute",
    href: s("08-other-ways#s-rs"),
    what: "Eyeriss's dataflow for convolutions, and computing where the weight is stored",
  },
  {
    term: "im2col, lowering",
    href: s("09-graph-to-silicon#s-im2col"),
    what: "turning a convolution into a GEMM so the array can run it",
  },
  {
    term: "Accelerator families",
    href: a("01-the-design-space#a-axes"),
    what: "where the weights live, who schedules the data, how big the engines are",
  },
  {
    term: "Decode bound, batching",
    href: a("02-gpus-the-baseline#a-decode"),
    what: "batch-1 decode is a full read of the weights; batching shares it",
  },
  {
    term: "Wafer-scale yield",
    href: a("04-wafer-scale#a-yield"),
    what: "Poisson yield, and why a whole wafer must survive its defects",
  },
  {
    term: "Static schedule",
    href: a("05-compiler-scheduled#a-static"),
    what: "every instruction and transfer on a cycle fixed by the compiler",
  },
  {
    term: "Kernel fusion, dataflow",
    href: a("06-reconfigurable-dataflow#a-fusion"),
    what: "a layer laid out as a circuit, its activations never leaving the chip",
  },
  {
    term: "Crossbar, analogue compute",
    href: a("09-in-memory-compute#a-hermes"),
    what: "weights as conductances; the column currents are the dot products",
  },
  {
    term: "Per-layer floor",
    href: a("10-choosing#a-floor"),
    what: "the fixed time per layer that sets an SRAM machine's speed",
  },
  {
    term: "NVLink domain, scale-up",
    href: r("01-chip-to-data-centre#r-cliff"),
    what: "the GPUs that share NVLink; crossing its edge costs an order of magnitude",
  },
  {
    term: "Package, HBM stack",
    href: r("02-the-package#r-capacity"),
    what: "compute dies and HBM on an interposer; capacity sets the GPU count",
  },
  {
    term: "Ring all-reduce, α–β model",
    href: r("04-scale-up-nvlink#r-ring"),
    what: "bandwidth-optimal collective; small messages are latency-bound",
  },
  {
    term: "Fat tree, oversubscription",
    href: r("05-scale-out-network#r-size"),
    what: "leaf, spine and core switches; fewer uplinks trade bandwidth for hardware",
  },
  {
    term: "Busbar, power smoothing",
    href: r("06-power#r-busbar"),
    what: "DC power down the rack; I²R loss falls with the voltage squared",
  },
  {
    term: "Liquid cooling, CDU",
    href: r("07-cooling#r-loop"),
    what: "cold plates and a coolant loop: Q = ṁ c_p ΔT",
  },
  {
    term: "Disaggregated serving",
    href: r("08-disaggregated-inference#r-pools"),
    what: "separate prefill and decode pools, and the KV hand-off between them",
  },
  {
    term: "Tensor, pipeline and data parallelism",
    href: r("09-training-clusters#r-split"),
    what: "how a training job is laid out across NVLink and the network",
  },
  {
    term: "Cost per token",
    href: r("11-cost#r-shape"),
    what: "capital and energy over tokens served; capital dominates",
  },
];
