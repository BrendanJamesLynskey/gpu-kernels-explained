"""
GPU execution model: the Python reference.

An illustrative, parameterised model of how a GPU executes a kernel. It is
not cycle-accurate. Every preset figure is a field with a value, a status and
a source (``PRESETS`` below); every derived figure is a closed form of those
fields (``derived``). The TypeScript port, ``src/lib/gpu/model.ts``, repeats
each function with the same operations in the same order, and must reproduce
the fixtures written by ``scripts/make_fixtures.py`` exactly.

What it computes, per kernel configuration:
  - bytes moved at each memory level (registers, shared memory, L2, HBM);
  - arithmetic intensity and the roofline position;
  - occupancy (threads, registers, shared memory, block limits), following
    the rules of the CUDA Toolkit's ``cuda_occupancy.h``;
  - shared-memory bank conflicts for an access pattern;
  - coalescing (32-byte sectors per warp access);
  - SIMT divergence (the active mask of every issued instruction);
  - a load / compute / store timeline per tile.

The functions that drive the site's animations return a list of states
(``*_steps``); a frame on the site is a pure function of one state.

Conventions:
  - bytes are SI (1 GB = 1e9 B) except capacities stated in KB/MB by the
    vendor, which are binary (1 KB = 1024 B), as the whitepapers use them;
  - all GEMM traffic is for FP32 operands on the CUDA cores (FFMA), with
    row-major A (M x K), B (K x N) and C (M x N);
  - "register bytes" are the operand bytes the ALUs read and write: three
    4-byte operands per fused multiply-add.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Sources and presets
# ---------------------------------------------------------------------------

CUDA_GUIDE = "https://docs.nvidia.com/cuda/cuda-programming-guide"

SOURCES: dict[str, dict[str, str]] = {
    "a100-wp": {
        "title": "NVIDIA A100 Tensor Core GPU Architecture (whitepaper)",
        "url": "https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf",
    },
    "h100-wp": {
        "title": "NVIDIA H100 Tensor Core GPU Architecture (whitepaper, GTC 2022; preliminary specifications)",
        "url": "https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper",
    },
    "h100-page": {
        "title": "NVIDIA H100 product page, Product Specifications (H100 SXM column)",
        "url": "https://www.nvidia.com/en-us/data-center/h100/",
    },
    "cuda-kernels": {
        "title": "CUDA Programming Guide, §2.3 Writing CUDA kernels",
        "url": f"{CUDA_GUIDE}/02-basics/writing-cuda-kernels.html",
    },
    "cuda-cc": {
        "title": "CUDA Programming Guide, Compute capabilities (Table 32: shared memory)",
        "url": f"{CUDA_GUIDE}/05-appendices/compute-capabilities.html",
    },
    "cuda-occ": {
        "title": "cuda_occupancy.h, the CUDA Toolkit's occupancy calculator header (CUDA 12.5 copy)",
        "url": "https://github.com/SciNim/nimcuda/blob/319b8deb31812fd9277c6232f0ae3fca31c6d55d/include/cuda12_5/cuda_occupancy.h",
    },
    "luo2024": {
        "title": "Luo et al., Benchmarking and Dissecting the Nvidia Hopper GPU Architecture (arXiv 2402.13499)",
        "url": "https://arxiv.org/abs/2402.13499",
    },
    "ptx": {
        "title": "PTX ISA: matrix shapes for mma and wgmma",
        "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html",
    },
}

KB = 1024
MB = 1024 * 1024


def _f(v: Any, st: str, src: str, ref: str) -> dict[str, Any]:
    """One preset field: value, status, source id, where in the source."""
    return {"v": v, "st": st, "src": src, "ref": ref}


# Statuses: spec (the vendor states it), rule (the CUDA documentation or
# toolkit states it), measured (a microbenchmark paper), derived (computed
# from other fields; the formula is in `ref`), approximation (a stand-in,
# named as such).
PRESETS: dict[str, dict[str, Any]] = {
    "a100": {
        "id": "a100",
        "name": "A100 SXM4 40 GB",
        "arch": "Ampere (GA100), compute capability 8.0",
        "fields": {
            "sms": _f(108, "spec", "a100-wp", "Table 4, p. 36"),
            "clock_mhz": _f(1410, "spec", "a100-wp", "Table 4 (GPU Boost Clock), p. 36"),
            "fp32_cores_per_sm": _f(64, "spec", "a100-wp", "Table 4, p. 36"),
            "peak_fp32": _f(19.5e12, "spec", "a100-wp", "Table 4 (Peak FP32 TFLOPS, non-Tensor), p. 36"),
            "peak_tensor": _f(312e12, "spec", "a100-wp", "Table 4 (Peak BF16 Tensor TFLOPS with FP32 accumulate, dense), p. 36"),
            "hbm_bw": _f(1555e9, "spec", "a100-wp", "Table 4 (Memory Bandwidth), p. 37"),
            "hbm_bytes": _f(40e9, "spec", "a100-wp", "Table 4 (Memory Size 40 GB), p. 36"),
            "l2_bytes": _f(40960 * KB, "spec", "a100-wp", "Table 4 (L2 Cache Size 40960 KB), p. 37"),
            "l2_bytes_per_clk": _f(5120, "spec", "a100-wp", "p. 35: 'The A100 L2 read bandwidth is 5120 Bytes/clk'"),
            "smem_per_sm": _f(164 * KB, "spec", "a100-wp", "Table 5 (configurable up to 164 KB), p. 43; CUDA guide Table 32"),
            "smem_per_block": _f(163 * KB, "rule", "cuda-cc", "Table 32, compute capability 8.0 (max per thread block, opt-in)"),
            "regs_per_sm": _f(65536, "spec", "a100-wp", "Table 5 (Max 32-bit Registers / SM), p. 43"),
            "regs_per_block": _f(65536, "spec", "a100-wp", "Table 5 (Max Registers / Block), p. 43"),
            "max_regs_per_thread": _f(255, "spec", "a100-wp", "Table 5 (Max Registers / Thread), p. 43"),
            "max_warps_per_sm": _f(64, "spec", "a100-wp", "Table 5 (Max Warps / SM), p. 43"),
            "max_blocks_per_sm": _f(32, "spec", "a100-wp", "Table 5 (Max Thread Blocks / SM), p. 43"),
            "max_threads_per_block": _f(1024, "spec", "a100-wp", "Table 5 (Max Thread Block Size), p. 43"),
            "lat_smem": _f(29.0, "measured", "luo2024", "Table IV, A100 PCIe (cycles)"),
            "lat_l2": _f(261.5, "measured", "luo2024", "Table IV, A100 PCIe (cycles)"),
            "lat_hbm": _f(466.3, "measured", "luo2024", "Table IV, A100 PCIe (cycles)"),
            "mma_shape": _f("mma.sync m16n8k16 (BF16, one warp)", "rule", "ptx", "#warp-level-matrix-shape"),
        },
    },
    "h100": {
        "id": "h100",
        "name": "H100 SXM5 80 GB",
        "arch": "Hopper (GH100), compute capability 9.0",
        "fields": {
            "sms": _f(132, "spec", "h100-wp", "Table 3 (H100 SXM5), p. 39"),
            "clock_mhz": _f(
                67e12 / (2 * 132 * 128) / 1e6,
                "derived",
                "h100-page",
                "peak FP32 / (2 x FP32 cores): 67e12 / (2 x 132 x 128) Hz; the whitepaper lists the clock as not finalised",
            ),
            "fp32_cores_per_sm": _f(128, "spec", "h100-wp", "Table 3 (FP32 Cores / SM), p. 39"),
            "peak_fp32": _f(67e12, "spec", "h100-page", "FP32 67 teraFLOPS"),
            "peak_tensor": _f(
                989.5e12,
                "derived",
                "h100-page",
                "BFLOAT16 Tensor Core 1,979 teraFLOPS is 'with sparsity'; dense is half",
            ),
            "hbm_bw": _f(3.35e12, "spec", "h100-page", "GPU Memory Bandwidth 3.35TB/s"),
            "hbm_bytes": _f(80e9, "spec", "h100-page", "GPU Memory 80GB"),
            "l2_bytes": _f(50 * MB, "spec", "h100-wp", "Table 3 (L2 Cache Size 50 MB), p. 40"),
            "l2_bytes_per_clk": _f(
                4472.3,
                "approximation",
                "luo2024",
                "Table V: L2 throughput measured on H800 PCIe (same GH100 die), FP32 loads; NVIDIA does not publish H100's",
            ),
            "smem_per_sm": _f(228 * KB, "spec", "h100-wp", "Table 4 (configurable up to 228 KB), p. 41; CUDA guide Table 32"),
            "smem_per_block": _f(227 * KB, "rule", "cuda-cc", "Table 32, compute capability 9.0 (max per thread block, opt-in)"),
            "regs_per_sm": _f(65536, "spec", "h100-wp", "Table 4 (Max 32-bit Registers / SM), p. 41"),
            "regs_per_block": _f(65536, "spec", "h100-wp", "Table 4 (Max Registers / Thread Block), p. 41"),
            "max_regs_per_thread": _f(255, "spec", "h100-wp", "Table 4 (Max Registers / Thread), p. 41"),
            "max_warps_per_sm": _f(64, "spec", "h100-wp", "Table 4 (Max Warps / SM), p. 41"),
            "max_blocks_per_sm": _f(32, "spec", "h100-wp", "Table 4 (Max Thread Blocks / SM), p. 41"),
            "max_threads_per_block": _f(1024, "spec", "h100-wp", "Table 4 (Max Thread Block Size), p. 41"),
            "lat_smem": _f(29.0, "measured", "luo2024", "Table IV, H800 PCIe (cycles)"),
            "lat_l2": _f(263.0, "measured", "luo2024", "Table IV, H800 PCIe (cycles)"),
            "lat_hbm": _f(478.8, "measured", "luo2024", "Table IV, H800 PCIe (cycles)"),
            "mma_shape": _f(
                "wgmma.mma_async m64nNk16 (BF16, one warpgroup, N up to 256)",
                "rule",
                "ptx",
                "#asynchronous-warpgroup-level-matrix-shape",
            ),
        },
    },
}

# Rules shared by both presets (compute capability 8.0 and 9.0).
COMMON: dict[str, dict[str, Any]] = {
    "warp_size": _f(32, "rule", "cuda-kernels", "§2.3 (a warp is 32 threads)"),
    "smem_banks": _f(32, "rule", "cuda-kernels", "#shared-memory-access-patterns: 32 banks"),
    "bank_bytes": _f(4, "rule", "cuda-kernels", "#shared-memory-access-patterns: successive 32-bit words, 32 bits per bank per clock"),
    "sector_bytes": _f(32, "rule", "cuda-kernels", "#coalesced-global-memory-access: 32-byte transactions"),
    "reg_alloc_unit": _f(256, "rule", "cuda-occ", "cudaOccRegAllocationGranularity: 256 registers per warp allocation"),
    "smem_alloc_unit": _f(128, "rule", "cuda-occ", "cudaOccSMemAllocationGranularity: 128 bytes (CC 8.x, 9.x)"),
    "smem_reserved": _f(1024, "rule", "cuda-occ", "reservedSharedMemPerBlock: 1 KB per block (per-SM minus per-block maximum, Table 32)"),
    "sub_partitions": _f(4, "rule", "cuda-occ", "cudaOccSubPartitionsPerMultiprocessor: 4"),
}


def preset(pid: str) -> dict[str, Any]:
    """Flat dict of a preset's values, with the common rules."""
    p = PRESETS[pid]
    out: dict[str, Any] = {"id": p["id"], "name": p["name"]}
    for k, f in COMMON.items():
        out[k] = f["v"]
    for k, f in p["fields"].items():
        out[k] = f["v"]
    return out


# ---------------------------------------------------------------------------
# Derived figures: the bandwidth at each level, and the ridge points
# ---------------------------------------------------------------------------

LEVELS = ("reg", "smem", "l2", "hbm")


def derived(p: dict[str, Any]) -> dict[str, Any]:
    clock = p["clock_mhz"] * 1e6
    cores = p["sms"] * p["fp32_cores_per_sm"]
    bw = {
        # what the FP32 ALUs consume at peak: three 4-byte operands per
        # FFMA, which is 2 flops, so 6 bytes per flop
        "reg": p["peak_fp32"] * 6,
        # 32 banks x 4 bytes per SM per clock
        "smem": p["sms"] * p["smem_banks"] * p["bank_bytes"] * clock,
        "l2": p["l2_bytes_per_clk"] * clock,
        "hbm": p["hbm_bw"],
    }
    return {
        "clock_hz": clock,
        "fp32_cores": cores,
        "bw": bw,
        "ridge_fp32": p["peak_fp32"] / bw["hbm"],
        "ridge_tensor": p["peak_tensor"] / bw["hbm"],
        "regs_bytes_per_sm": p["regs_per_sm"] * 4,
    }


# ---------------------------------------------------------------------------
# Bytes moved at each level, per kernel
# ---------------------------------------------------------------------------


def elementwise_traffic(n: int) -> dict[str, Any]:
    """c[i] = a[i] + b[i] on n FP32 elements: two loads, one store, one add."""
    return {
        "flops": n,
        "bytes": {"reg": 12 * n, "smem": 0, "l2": 12 * n, "hbm": 12 * n},
    }


def gemm_traffic(
    p: dict[str, Any], m: int, n: int, k: int, bm: int, bn: int, tm: int, tn: int
) -> dict[str, Any]:
    """
    C = A B in FP32, tiled: each thread block computes a bm x bn tile of C,
    staging A and B tiles through shared memory; each thread computes a
    tm x tn sub-tile from registers.

    - Global loads: every block reads its bm rows of A and bn columns of B
      once, so A is read n/bn times and B m/bm times; C is written once.
    - HBM: if A and B fit in L2 together, the compulsory bytes (each matrix
      once); otherwise every global load misses L2 (an idealisation, named
      on the site).
    - Shared memory: the staged tiles are written once and, at every k, each
      thread reads tm values of A and tn values of B.
    - Registers: three 4-byte operands per FFMA.
    """
    assert m % bm == 0 and n % bn == 0 and bm % tm == 0 and bn % tn == 0
    flops = 2 * m * n * k
    loads = 4 * (m * k * (n // bn) + k * n * (m // bm))
    store = 4 * m * n
    l2 = loads + store
    compulsory = 4 * (m * k + k * n + m * n)
    fits = 4 * (m * k + k * n) <= p["l2_bytes"]
    hbm = compulsory if fits else l2
    smem_reads = 4 * k * (m * n // (tm * tn)) * (tm + tn)
    return {
        "flops": flops,
        "bytes": {"reg": 12 * m * n * k, "smem": loads + smem_reads, "l2": l2, "hbm": hbm},
        "l2_fits": fits,
    }


KERNELS: dict[str, dict[str, Any]] = {
    "vecadd": {
        "label": "Vector add, 2^28 elements",
        "kind": "elementwise",
        "n": 2**28,
    },
    "gemm16": {
        "label": "GEMM 4096³, 16×16 tiles in shared memory",
        "kind": "gemm",
        "m": 4096, "n": 4096, "k": 4096, "bm": 16, "bn": 16, "tm": 1, "tn": 1,
    },
    "gemm128": {
        "label": "GEMM 4096³, 128×128 tiles, 8×8 per thread",
        "kind": "gemm",
        "m": 4096, "n": 4096, "k": 4096, "bm": 128, "bn": 128, "tm": 8, "tn": 8,
    },
}


def kernel_traffic(p: dict[str, Any], kernel: dict[str, Any]) -> dict[str, Any]:
    if kernel["kind"] == "elementwise":
        return elementwise_traffic(kernel["n"])
    return gemm_traffic(
        p, kernel["m"], kernel["n"], kernel["k"], kernel["bm"], kernel["bn"], kernel["tm"], kernel["tn"]
    )


def kernel_time(p: dict[str, Any], kernel: dict[str, Any]) -> dict[str, Any]:
    """
    The hierarchical-roofline estimate: each level and the ALUs work in
    parallel, so the kernel takes as long as its slowest one. `util` is each
    level's busy fraction over that time.
    """
    d = derived(p)
    t = kernel_traffic(p, kernel)
    times = {lv: t["bytes"][lv] / d["bw"][lv] for lv in LEVELS}
    compute = t["flops"] / p["peak_fp32"]
    total = compute
    bound = "compute"
    for lv in LEVELS:
        if times[lv] > total:
            total = times[lv]
            bound = lv
    util = {lv: times[lv] / total for lv in LEVELS}
    ai = {lv: (t["flops"] / t["bytes"][lv] if t["bytes"][lv] > 0 else None) for lv in LEVELS}
    return {
        "flops": t["flops"],
        "bytes": t["bytes"],
        "times": times,
        "compute": compute,
        "total": total,
        "bound": bound,
        "util": util,
        "compute_util": compute / total,
        "ai": ai,
        "achieved": t["flops"] / total,
    }


def flow_steps(p: dict[str, Any], kernel: dict[str, Any], steps: int = 60) -> list[dict[str, Any]]:
    """
    The memory-hierarchy animation: at step i of `steps`, the kernel is i/steps
    of the way through, and every level has moved that fraction of its bytes
    (a pipelined kernel moves data at every level at once).
    """
    kt = kernel_time(p, kernel)
    out = []
    for i in range(steps + 1):
        out.append(
            {
                "i": i,
                "t": kt["total"] * i / steps,
                "moved": {lv: kt["bytes"][lv] * i / steps for lv in LEVELS},
                "flops": kt["flops"] * i / steps,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Roofline
# ---------------------------------------------------------------------------


def attainable(peak: float, bw: float, ai: float) -> float:
    """min(peak, ai x bw): the roofline."""
    mem = ai * bw
    return mem if mem < peak else peak


def roofline_point(p: dict[str, Any], ai: float, engine: str) -> dict[str, Any]:
    peak = p["peak_fp32"] if engine == "fp32" else p["peak_tensor"]
    bw = p["hbm_bw"]
    ridge = peak / bw
    perf = attainable(peak, bw, ai)
    return {
        "ai": ai,
        "perf": perf,
        "ridge": ridge,
        "bound": "memory" if ai < ridge else "compute",
        "frac_peak": perf / peak,
    }


SWEEP_TILES = (1, 2, 4, 8, 16, 32, 64, 128)


def roofline_sweep(p: dict[str, Any], size: int = 4096) -> list[dict[str, Any]]:
    """
    Square GEMM, block tile b x b with 16 x 16 threads (b <= 16: one output
    per thread; larger b: b/16 x b/16 per thread). One state per tile size:
    the HBM arithmetic intensity and where it sits on the FP32 roofline.
    """
    out = []
    for b in SWEEP_TILES:
        t = b // 16 if b > 16 else 1
        g = gemm_traffic(p, size, size, size, b, b, t, t)
        ai = g["flops"] / g["bytes"]["hbm"]
        pt = roofline_point(p, ai, "fp32")
        out.append(
            {
                "tile": b,
                "per_thread": t,
                "flops": g["flops"],
                "hbm": g["bytes"]["hbm"],
                "ai": ai,
                "perf": pt["perf"],
                "bound": pt["bound"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# SIMT: a warp meets a branch
# ---------------------------------------------------------------------------

FULL_MASK = 0xFFFFFFFF


def xorshift32(x: int) -> int:
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


def lane_data(seed: int) -> list[int]:
    """32 pseudo-random values in 0..99, one per lane (xorshift32)."""
    x = seed if seed != 0 else 1
    out = []
    for _ in range(32):
        x = xorshift32(x)
        out.append(x % 100)
    return out


def branch_mask(cond: str, k: int, seed: int = 2024) -> int:
    """The lanes for which the branch condition is true, as a 32-bit mask."""
    m = 0
    data = lane_data(seed) if cond == "data" else []
    for lane in range(32):
        if cond == "lt":
            t = lane < k
        elif cond == "mod":
            t = lane % k == 0
        elif cond == "uniform":
            # blockIdx.x < k: the same answer for every lane of the warp
            t = 0 < k
        elif cond == "data":
            t = data[lane] < k
        else:
            raise ValueError(cond)
        if t:
            m |= 1 << lane
    return m


def popcount(x: int) -> int:
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def simt_steps(cond: str, k: int, len_a: int, len_b: int, len_c: int, seed: int = 2024) -> dict[str, Any]:
    """
    if (cond) { A: len_a instructions } else { B: len_b instructions }
    then C: len_c instructions after reconvergence.
    The warp issues each instruction once for the lanes that take it; a path
    no lane takes is skipped.
    """
    taken = branch_mask(cond, k, seed)
    not_taken = FULL_MASK ^ taken
    steps: list[dict[str, Any]] = [
        {"phase": "pre", "j": 0, "mask": FULL_MASK},
        {"phase": "branch", "j": 0, "mask": FULL_MASK},
    ]
    if taken:
        for j in range(len_a):
            steps.append({"phase": "A", "j": j, "mask": taken})
    if not_taken:
        for j in range(len_b):
            steps.append({"phase": "B", "j": j, "mask": not_taken})
    for j in range(len_c):
        steps.append({"phase": "C", "j": j, "mask": FULL_MASK})
    active = 0
    for s in steps:
        active += popcount(s["mask"])
    issued = len(steps)
    return {
        "taken": taken,
        "steps": steps,
        "issued": issued,
        "active_lane_slots": active,
        "efficiency": active / (32 * issued),
        "divergent": taken != 0 and not_taken != 0,
    }


# ---------------------------------------------------------------------------
# Coalescing: 32-byte sectors per warp access
# ---------------------------------------------------------------------------


def coalesce_steps(elem_bytes: int, stride: int, offset: int, sector: int = 32) -> dict[str, Any]:
    """
    Lane i reads elem_bytes at offset + i * stride * elem_bytes. The warp's
    request is served by every 32-byte sector it touches.
    """
    seen: list[int] = []
    steps = []
    for lane in range(32):
        addr = offset + lane * stride * elem_bytes
        first = addr // sector
        last = (addr + elem_bytes - 1) // sector
        lane_sectors = list(range(first, last + 1))
        new = [s for s in lane_sectors if s not in seen]
        seen.extend(new)
        steps.append(
            {"lane": lane, "addr": addr, "sectors": lane_sectors, "new": new, "total": len(seen)}
        )
    lines: list[int] = []
    for s in seen:
        ln = s // 4
        if ln not in lines:
            lines.append(ln)
    requested = 32 * elem_bytes
    fetched = len(seen) * sector
    return {
        "steps": steps,
        "sectors": len(seen),
        "lines": len(lines),
        "requested": requested,
        "fetched": fetched,
        "efficiency": requested / fetched,
    }


# ---------------------------------------------------------------------------
# Shared-memory bank conflicts
# ---------------------------------------------------------------------------


def bank_words(pattern: str, pad: int, stride: int = 1, cols: int = 32) -> list[int]:
    """The 32-bit word each lane addresses in float s[32][cols + pad]."""
    pitch = cols + pad
    out = []
    for lane in range(32):
        if pattern == "row":  # s[0][lane]: a row
            out.append(lane)
        elif pattern == "col":  # s[lane][0]: a column (the transpose read)
            out.append(lane * pitch)
        elif pattern == "stride":  # s_flat[lane * stride]
            out.append(lane * stride)
        elif pattern == "broadcast":  # s[0][lane / 8]: groups of 8 lanes share a word
            out.append(lane // 8)
        else:
            raise ValueError(pattern)
    return out


def bank_steps(pattern: str, pad: int, stride: int = 1, banks: int = 32) -> dict[str, Any]:
    """
    Requests (one lane at a time) then service passes: in each pass every
    bank serves one distinct word; lanes reading the same word share it
    (broadcast). The conflict degree is the number of passes.
    """
    words = bank_words(pattern, pad, stride)
    distinct: list[list[int]] = [[] for _ in range(banks)]
    requests = []
    for lane, w in enumerate(words):
        b = w % banks
        if w not in distinct[b]:
            distinct[b].append(w)
        requests.append({"lane": lane, "word": w, "bank": b, "load": [len(d) for d in distinct]})
    degree = max(1, max(len(d) for d in distinct))
    passes = []
    for pnum in range(degree):
        served = []
        for lane, w in enumerate(words):
            b = w % banks
            if len(distinct[b]) > pnum and distinct[b][pnum] == w:
                served.append(lane)
        passes.append({"pass": pnum, "lanes": served})
    return {"words": words, "requests": requests, "passes": passes, "degree": degree}


# ---------------------------------------------------------------------------
# Occupancy (the rules of cuda_occupancy.h, CC 8.0 / 9.0, carveout at max)
# ---------------------------------------------------------------------------


def _round_up(x: int, y: int) -> int:
    return y * ((x + y - 1) // y)


def occupancy(p: dict[str, Any], threads: int, regs: int, smem: int) -> dict[str, Any]:
    warps_per_block = (threads + p["warp_size"] - 1) // p["warp_size"]
    big = 1 << 30
    # threads (warp slots)
    if threads > p["max_threads_per_block"] or threads < 1:
        lim_warps = 0
    else:
        lim_warps = p["max_warps_per_sm"] // warps_per_block
    # registers: allocated per warp in units of 256, per sub-partition
    regs_per_warp = _round_up(regs * p["warp_size"], p["reg_alloc_unit"])
    regs_per_block = regs_per_warp * warps_per_block
    regs_assumed = regs_per_warp * _round_up(warps_per_block, p["sub_partitions"])
    if (
        p["regs_per_block"] < regs_assumed
        or p["regs_per_block"] < regs_per_block
        or regs > p["max_regs_per_thread"]
    ):
        lim_regs = 0
    elif regs_per_warp > 0:
        per_sub = (p["regs_per_sm"] // p["sub_partitions"]) // regs_per_warp
        lim_regs = (per_sub * p["sub_partitions"]) // warps_per_block
    else:
        lim_regs = big
    # shared memory: request + 1 KB reserved, in units of 128 bytes
    smem_alloc = _round_up(smem + p["smem_reserved"], p["smem_alloc_unit"])
    if smem_alloc > p["smem_per_block"] + p["smem_reserved"]:
        lim_smem = 0
    else:
        lim_smem = p["smem_per_sm"] // smem_alloc
    lim_blocks = p["max_blocks_per_sm"]
    limits = {"warps": lim_warps, "regs": lim_regs, "smem": lim_smem, "blocks": lim_blocks}
    blocks = min(lim_warps, lim_regs, lim_smem, lim_blocks)
    limiters = [k for k in ("warps", "regs", "smem", "blocks") if limits[k] == blocks]
    active_warps = blocks * warps_per_block
    return {
        "warps_per_block": warps_per_block,
        "regs_per_warp": regs_per_warp,
        "regs_per_block": regs_per_block,
        "smem_alloc": smem_alloc,
        "limits": limits,
        "blocks": blocks,
        "limiters": limiters,
        "active_warps": active_warps,
        "occupancy": active_warps / p["max_warps_per_sm"],
    }


def occupancy_steps(p: dict[str, Any], threads: int, regs: int, smem: int) -> dict[str, Any]:
    """Place blocks on one SM, one at a time, until the next does not fit."""
    o = occupancy(p, threads, regs, smem)
    steps = []
    for b in range(o["blocks"] + 1):
        steps.append(
            {
                "blocks": b,
                "warps": b * o["warps_per_block"],
                "regs": b * o["regs_per_block"],
                "smem": b * o["smem_alloc"],
                "rejected": b == o["blocks"],
            }
        )
    return {"result": o, "steps": steps}


# ---------------------------------------------------------------------------
# Load / compute / store timeline per tile
# ---------------------------------------------------------------------------


def tile_timeline(n: int, t_load: float, t_compute: float, t_store: float, buffers: int) -> dict[str, Any]:
    """
    Three engines (copy-in, compute, copy-out), one tile at a time each.
    Tile i's load needs a free buffer: the compute of tile i - buffers must
    have finished. With one buffer nothing overlaps load and compute; with
    two, the next tile loads while this one computes.
    """
    load_end: list[float] = []
    comp_end: list[float] = []
    store_end: list[float] = []
    segs = []
    for i in range(n):
        ls = load_end[i - 1] if i > 0 else 0.0
        if i >= buffers and comp_end[i - buffers] > ls:
            ls = comp_end[i - buffers]
        le = ls + t_load
        cs = le
        if i > 0 and comp_end[i - 1] > cs:
            cs = comp_end[i - 1]
        ce = cs + t_compute
        ss = ce
        if i > 0 and store_end[i - 1] > ss:
            ss = store_end[i - 1]
        se = ss + t_store
        load_end.append(le)
        comp_end.append(ce)
        store_end.append(se)
        segs.append({"tile": i, "load": [ls, le], "compute": [cs, ce], "store": [ss, se]})
    total = store_end[n - 1] if n else 0.0
    busy = n * t_compute
    return {"segments": segs, "total": total, "compute_busy": busy / total if total > 0 else 0.0}
