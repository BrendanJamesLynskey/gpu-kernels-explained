"""
Hopper and Blackwell GEMM (chapter 12): the Python reference.

Three pieces, each ported line for line to ``src/lib/gpu/blackwell.ts`` and
checked by the fixtures ``scripts/make_fixtures.py`` writes
(``tests/fixtures/blackwell_fixtures.json``):

1. **Tensor-core presets** (``TC_PRESETS``): the A100, H100 and B200 as their
   tensor cores see them, every field a value, a status and a source, in the
   same shape as ``gpu_model.PRESETS``. The A100 and H100 figures are the
   ones ``gpu_model`` already uses; the B200 is the Blackwell GPU of a GB200
   NVL72, with the same figures and sources as the AI Racks and AI
   Accelerators sites (per-GPU values from NVIDIA's GB200 NVL72 page and
   the Blackwell Ultra blog's Table 2). Fields NVIDIA does not publish are
   ``None`` with status ``undisclosed``.
2. **Ridge points** by number format: peak dense tensor throughput divided
   by HBM bandwidth (flops per byte), the roofline of chapter 2 for the
   tensor cores.
3. **A warp-specialised GEMM pipeline** (``pipeline``): producer warps load
   K-slices of A and B with TMA into a ring of shared-memory stages, one
   thread issues the tensor-core MMAs, and accumulators live in one or two
   buffers (registers on Hopper with one consumer; Tensor Memory, double
   buffered, on Blackwell) while an epilogue drains finished tiles. Times are
   integers in illustrative units (the default MMA takes 4); the schedule is
   a set of max-plus recurrences, and ``frame`` is the state at one time
   step.

Integer arithmetic throughout the pipeline, so the port matches exactly.
"""
from __future__ import annotations

from typing import Any

from gpu_model import KB, PRESETS, _f

SOURCES: dict[str, dict[str, str]] = {
    "gb200-page": {
        "title": "NVIDIA GB200 NVL72 product page, Specifications (GB200 Superchip column: two Blackwell GPUs)",
        "url": "https://www.nvidia.com/en-us/data-center/gb200-nvl72/",
    },
    "bwu-blog": {
        "title": "Inside NVIDIA Blackwell Ultra: The Chip Powering the AI Factory Era (NVIDIA Technical Blog), Table 2",
        "url": "https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/",
    },
    "h100-page": {
        "title": "NVIDIA H100 product page, Product Specifications (H100 SXM column)",
        "url": "https://www.nvidia.com/en-us/data-center/h100/",
    },
    "a100-wp": {
        "title": "NVIDIA A100 Tensor Core GPU Architecture (whitepaper)",
        "url": "https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf",
    },
    "cuda-cc": {
        "title": "CUDA Programming Guide, Compute capabilities (technical specifications per compute capability)",
        "url": "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html",
    },
    "ptx-tcgen05": {
        "title": "PTX ISA, §9.7.18 TensorCore 5th Generation Family Instructions (Tensor Memory; Table 55, issue granularity)",
        "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions",
    },
    "ptx-wgmma": {
        "title": "PTX ISA, §9.7.17 Asynchronous Warpgroup Level Matrix Multiply-Accumulate (wgmma)",
        "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions",
    },
    "jarmusch2025": {
        "title": "Jarmusch and Chandrasekaran, Microbenchmarking NVIDIA's Blackwell Architecture: An in-depth Architectural Analysis (arXiv 2512.02189)",
        "url": "https://arxiv.org/abs/2512.02189",
    },
    "ocp-mx": {
        "title": "Rouhani et al., Microscaling Data Formats for Deep Learning (arXiv 2310.10537): the OCP MX formats, 32 elements per shared E8M0 scale",
        "url": "https://arxiv.org/abs/2310.10537",
    },
}


def _from(pid: str, key: str) -> dict[str, Any]:
    """A field gpu_model already sources (the same value, status and source)."""
    return dict(PRESETS[pid]["fields"][key])


TC_PRESETS: dict[str, dict[str, Any]] = {
    "a100": {
        "id": "a100",
        "name": "A100 SXM4 40 GB",
        "arch": "Ampere, compute capability 8.0",
        "fields": {
            "sms": _from("a100", "sms"),
            "smem_per_sm": _from("a100", "smem_per_sm"),
            "peak_bf16": _from("a100", "peak_tensor"),
            "peak_fp8": _f(None, "none", "a100-wp", "Ampere tensor cores have no FP8 mode"),
            "peak_fp4": _f(None, "none", "a100-wp", "no FP4 mode"),
            "peak_int8": _from("a100", "peak_int8_tensor"),
            "hbm_bw": _from("a100", "hbm_bw"),
            "hbm_bytes": _from("a100", "hbm_bytes"),
            "tmem_per_sm": _f(None, "none", "a100-wp", "no Tensor Memory: accumulators live in registers"),
            "mma": _f("mma.sync: one warp, operands and accumulators in registers", "rule", "ptx-wgmma", "§9.7.16 (warp-level mma)"),
            "loads": _f("cp.async: threads copy global to shared memory, 4 to 16 bytes each", "rule", "cuda-cc", "compute capability 8.0"),
        },
    },
    "h100": {
        "id": "h100",
        "name": "H100 SXM5 80 GB",
        "arch": "Hopper, compute capability 9.0",
        "fields": {
            "sms": _from("h100", "sms"),
            "smem_per_sm": _from("h100", "smem_per_sm"),
            "peak_bf16": _from("h100", "peak_tensor"),
            "peak_fp8": _f(1979e12, "derived", "h100-page", "FP8 Tensor Core 3,958 teraFLOPS is 'with sparsity'; dense is half"),
            "peak_fp4": _f(None, "none", "h100-page", "no FP4 mode"),
            "peak_int8": _from("h100", "peak_int8_tensor"),
            "hbm_bw": _from("h100", "hbm_bw"),
            "hbm_bytes": _from("h100", "hbm_bytes"),
            "tmem_per_sm": _f(None, "none", "ptx-wgmma", "no Tensor Memory: wgmma accumulates in registers"),
            "mma": _f(
                "wgmma.mma_async: a warpgroup (4 warps), asynchronous; B from shared memory, A from registers or shared memory, accumulators in registers",
                "rule", "ptx-wgmma", "§9.7.17",
            ),
            "loads": _f("TMA (cp.async.bulk.tensor): one thread copies a whole tile, completion on an mbarrier", "rule", "ptx-wgmma", "§9.7.10.28.5.3 (cp.async.bulk.tensor)"),
        },
    },
    "b200": {
        "id": "b200",
        "name": "B200 (as in GB200 NVL72)",
        "arch": "Blackwell, compute capability 10.0",
        "fields": {
            "sms": _f(148, "measured", "jarmusch2025", "§III-A: 'feature 148 SMs spread across eight GPCs' (NVIDIA's pages give no SM count)"),
            "smem_per_sm": _f(228 * KB, "rule", "cuda-cc", "compute capability 10.0: maximum shared memory per SM 228 KB"),
            "peak_bf16": _f(2.5e15, "derived", "gb200-page", "GB200 Superchip FP16/BF16 10 PFLOPS is sparse for 2 GPUs: dense per GPU = 10 / 2 / 2"),
            "peak_fp8": _f(5e15, "spec", "bwu-blog", "Table 2, Blackwell: FP8 dense 5 PetaFLOPS"),
            "peak_fp4": _f(10e15, "spec", "bwu-blog", "Table 2, Blackwell: NVFP4 dense 10 PetaFLOPS (GB200 Superchip: 20 PFLOPS dense for 2 GPUs)"),
            "peak_int8": _f(5e15, "derived", "gb200-page", "GB200 Superchip INT8 20 POPS sparse for 2 GPUs: dense per GPU = 20 / 2 / 2"),
            "hbm_bw": _f(8e12, "derived", "gb200-page", "GB200 Superchip: 16 TB/s for 2 GPUs, so 8 TB/s per GPU (bwu-blog Table 2: 8 TB/s)"),
            "hbm_bytes": _f(186e9, "derived", "gb200-page", "GB200 Superchip: 372 GB HBM3E for 2 GPUs, so 186 GB per GPU"),
            "tmem_per_sm": _f(256 * KB, "rule", "ptx-tcgen05", "§9.7.18.1: 512 columns x 128 lanes of 32-bit cells per CTA on sm_100a"),
            "mma": _f(
                "tcgen05.mma: issued by a single thread (or one thread of a CTA pair), operands in shared memory (A may be in Tensor Memory), accumulators in Tensor Memory",
                "rule", "ptx-tcgen05", "§9.7.18.5, Table 55",
            ),
            "loads": _f("TMA, as on Hopper; tcgen05.cp copies shared memory into Tensor Memory", "rule", "ptx-tcgen05", "§9.7.18.9"),
        },
    },
}
TC_ORDER = ("a100", "h100", "b200")
FORMATS = ("bf16", "fp8", "fp4", "int8")


def tc_preset(pid: str) -> dict[str, Any]:
    p = TC_PRESETS[pid]
    out: dict[str, Any] = {"id": p["id"], "name": p["name"]}
    for k, f in p["fields"].items():
        out[k] = f["v"]
    return out


def ridges(pid: str) -> dict[str, Any]:
    """Flops (or ops) per HBM byte at which each tensor format turns compute-bound."""
    p = tc_preset(pid)
    out: dict[str, Any] = {}
    for fmt in FORMATS:
        peak = p[f"peak_{fmt}"]
        out[fmt] = None if peak is None else peak / p["hbm_bw"]
    return out


# ---------------------------------------------------------------------------
# Block-scaled formats
# ---------------------------------------------------------------------------

# bits: element bits; block: elements sharing one scale; scale_bits: bits
# of that scale (0 = a per-tensor scale only, negligible per element)
BLOCK_FORMATS: dict[str, dict[str, Any]] = {
    "bf16": {"name": "BF16", "bits": 16, "block": 0, "scale": "none", "scale_bits": 0, "src": "ptx-tcgen05"},
    "fp8": {"name": "FP8 E4M3, per-tensor scale", "bits": 8, "block": 0, "scale": "FP32 per tensor", "scale_bits": 0, "src": "ptx-tcgen05"},
    "mxfp8": {"name": "MXFP8 (E4M3)", "bits": 8, "block": 32, "scale": "E8M0", "scale_bits": 8, "src": "ocp-mx"},
    "mxfp4": {"name": "MXFP4 (E2M1)", "bits": 4, "block": 32, "scale": "E8M0", "scale_bits": 8, "src": "ocp-mx"},
    "nvfp4": {"name": "NVFP4 (E2M1)", "bits": 4, "block": 16, "scale": "E4M3, plus FP32 per tensor", "scale_bits": 8, "src": "bwu-blog"},
}
BLOCK_ORDER = ("bf16", "fp8", "mxfp8", "mxfp4", "nvfp4")


def bits_per_value(fmt: str) -> float:
    f = BLOCK_FORMATS[fmt]
    return f["bits"] + (f["scale_bits"] / f["block"] if f["block"] else 0)


def weight_bytes(fmt: str, rows: int, cols: int) -> float:
    """Bytes of a rows x cols weight matrix, scales included (blocks along cols)."""
    f = BLOCK_FORMATS[fmt]
    b = rows * cols * f["bits"] / 8
    if f["block"]:
        b += rows * (cols // f["block"]) * f["scale_bits"] / 8
    return b


# ---------------------------------------------------------------------------
# Shared-memory stages
# ---------------------------------------------------------------------------


def stage_bytes(bm: int, bn: int, bk: int, elem_bytes: int) -> int:
    """One pipeline stage: a bm x bk slice of A and a bk x bn slice of B."""
    return (bm * bk + bk * bn) * elem_bytes


def max_stages(smem: int, sbytes: int, reserve: int = 0) -> int:
    return (smem - reserve) // sbytes


# ---------------------------------------------------------------------------
# The warp-specialised pipeline
# ---------------------------------------------------------------------------

PIPE_DEFAULT: dict[str, int] = {
    "tiles": 3,   # output tiles this CTA computes, one after another
    "ksteps": 6,  # K-slices per tile
    "stages": 4,  # shared-memory ring slots
    "accs": 2,    # accumulator buffers (1: registers, one consumer; 2: TMEM double buffer)
    "load": 12,   # TMA latency, issue to data in shared memory
    "issue": 2,   # the shortest gap between two TMA issues (bandwidth)
    "mma": 4,     # one K-slice of MMAs for one tile
    "epi": 8,     # epilogue: drain one tile's accumulators and store C
}


def pipeline(
    tiles: int, ksteps: int, stages: int, accs: int, load: int, issue: int, mma: int, epi: int
) -> dict[str, Any]:
    """The schedule, as max-plus recurrences over K-slices g = tile * ksteps + k.

    load_start[g] = max(load_start[g-1] + issue, free[g - stages])
    load_done[g]  = load_start[g] + load
    mma_start[g]  = max(load_done[g], mma_done[g-1], acc_free(tile) if k = 0)
    mma_done[g]   = mma_start[g] + mma;  free[g] = mma_done[g]
    acc_free(t)   = epi_done[t - accs]   (the buffer the tile writes is drained)
    epi_start[t]  = max(mma_done[last g of t], epi_done[t-1]);  epi_done = + epi
    """
    n = tiles * ksteps
    ls: list[int] = []
    ld: list[int] = []
    ms: list[int] = []
    md: list[int] = []
    es: list[int] = []
    ed: list[int] = []
    for g in range(n):
        t, k = divmod(g, ksteps)
        a = ls[g - 1] + issue if g > 0 else 0
        if g >= stages and md[g - stages] > a:
            a = md[g - stages]
        ls.append(a)
        ld.append(a + load)
        b = ld[g]
        if g > 0 and md[g - 1] > b:
            b = md[g - 1]
        if k == 0 and t >= accs and ed[t - accs] > b:
            b = ed[t - accs]
        ms.append(b)
        md.append(b + mma)
        if k == ksteps - 1:
            e = md[g]
            if t > 0 and ed[t - 1] > e:
                e = ed[t - 1]
            es.append(e)
            ed.append(e + epi)
    total = ed[-1]
    busy = n * mma
    # the steady-state K-slice interval the recurrences settle to (no tile
    # boundaries): the ring must cover a load and an MMA, S slices at a time
    steady = max(mma, issue, -(-(load + mma) // stages))
    return {
        "params": {"tiles": tiles, "ksteps": ksteps, "stages": stages, "accs": accs,
                   "load": load, "issue": issue, "mma": mma, "epi": epi},
        "loadStart": ls, "loadDone": ld, "mmaStart": ms, "mmaDone": md,
        "epiStart": es, "epiDone": ed,
        "total": total, "busy": busy, "util": busy / total, "steady": steady,
        "stall": sum(ms[g] - (md[g - 1] if g > 0 else ld[0]) for g in range(n)),
    }


def pipeline_default(**over: int) -> dict[str, Any]:
    p = dict(PIPE_DEFAULT)
    p.update(over)
    return pipeline(p["tiles"], p["ksteps"], p["stages"], p["accs"], p["load"], p["issue"], p["mma"], p["epi"])


def frame(s: dict[str, Any], time: int) -> dict[str, Any]:
    """Everything the animation draws at one time step (a pure function).

    slots[i]: [state, g] for ring slot i, state in empty | loading | full |
    mma (the tensor core is reading it); producer: the K-slice whose TMA is
    issued at this step, or -1; tensor: the K-slice being multiplied, or -1;
    accs[b]: [state, tile] with state free | acc | epi; done: K-slices
    finished; tilesDone: tiles stored.
    """
    p = s["params"]
    st, K, A = p["stages"], p["ksteps"], p["accs"]
    n = p["tiles"] * K
    slots: list[list[Any]] = [["empty", -1] for _ in range(st)]
    producer = -1
    tensor = -1
    done = 0
    for g in range(n):
        if s["loadStart"][g] == time:
            producer = g
        if s["mmaDone"][g] <= time:
            done += 1
        i = g % st
        if s["loadStart"][g] <= time < s["mmaDone"][g]:
            if time < s["loadDone"][g]:
                slots[i] = ["loading", g]
            elif time < s["mmaStart"][g]:
                slots[i] = ["full", g]
            else:
                slots[i] = ["mma", g]
                tensor = g
    accs: list[list[Any]] = [["free", -1] for _ in range(A)]
    tiles_done = 0
    for t in range(p["tiles"]):
        b = t % A
        first = s["mmaStart"][t * K]
        if s["epiDone"][t] <= time:
            tiles_done += 1
        if first <= time < s["epiStart"][t]:
            accs[b] = ["acc", t]
        elif s["epiStart"][t] <= time < s["epiDone"][t]:
            accs[b] = ["epi", t]
    return {"t": time, "slots": slots, "producer": producer, "tensor": tensor,
            "accs": accs, "done": done, "tilesDone": tiles_done}


def frames(s: dict[str, Any]) -> list[dict[str, Any]]:
    return [frame(s, t) for t in range(s["total"] + 1)]
