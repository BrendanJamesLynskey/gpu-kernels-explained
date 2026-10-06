"""
Tests of the reference GPU model against the closed forms and the worked
examples in its sources (CUDA Programming Guide, the whitepapers,
cuda_occupancy.h). Run: python3 -m pytest -q tests/python
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference"))

import gpu_model as g  # noqa: E402

A100 = g.preset("a100")
H100 = g.preset("h100")


# --- presets ---------------------------------------------------------------


@pytest.mark.parametrize("pid", ["a100", "h100"])
def test_every_field_has_a_known_source_and_status(pid):
    fields = dict(g.PRESETS[pid]["fields"], **g.COMMON)
    for key, f in fields.items():
        assert f["src"] in g.SOURCES, key
        assert f["st"] in {"spec", "rule", "measured", "derived", "approximation"}, key
        assert f["ref"], key


def test_a100_clock_matches_its_fp32_peak():
    # 2 flops per FFMA x 6912 cores x 1410 MHz = 19.49 TFLOPS, Table 4 says 19.5
    d = g.derived(A100)
    assert d["fp32_cores"] == 6912
    assert abs(2 * d["fp32_cores"] * d["clock_hz"] / A100["peak_fp32"] - 1) < 0.001


def test_h100_clock_is_derived_from_its_fp32_peak():
    d = g.derived(H100)
    assert 2 * d["fp32_cores"] * d["clock_hz"] == pytest.approx(67e12, rel=1e-12)
    assert 1980e6 < d["clock_hz"] < 1985e6


def test_a100_level_bandwidths():
    bw = g.derived(A100)["bw"]
    assert bw["hbm"] == 1555e9
    assert bw["l2"] == 5120 * 1410e6  # p. 35
    assert bw["smem"] == 108 * 32 * 4 * 1410e6  # 32 banks x 4 B per SM per clock
    assert bw["reg"] == 6 * 19.5e12
    # the ladder is ordered: registers > shared > L2 > HBM
    assert bw["reg"] > bw["smem"] > bw["l2"] > bw["hbm"]
    assert g.derived(H100)["bw"]["reg"] > g.derived(H100)["bw"]["smem"] > g.derived(H100)["bw"]["l2"] > g.derived(H100)["bw"]["hbm"]


def test_ridge_points():
    assert g.derived(A100)["ridge_fp32"] == pytest.approx(19.5e12 / 1555e9)
    assert g.derived(H100)["ridge_fp32"] == pytest.approx(20.0)
    assert g.derived(H100)["ridge_tensor"] == pytest.approx(989.5e12 / 3.35e12)


# --- traffic and time -------------------------------------------------------


def test_elementwise_traffic():
    t = g.elementwise_traffic(1000)
    assert t["flops"] == 1000
    assert t["bytes"] == {"reg": 12000, "smem": 0, "l2": 12000, "hbm": 12000}


def test_gemm_traffic_closed_form():
    m = n = k = 4096
    for b, tt in [(16, 1), (128, 8)]:
        t = g.gemm_traffic(A100, m, n, k, b, b, tt, tt)
        assert t["flops"] == 2 * m * n * k
        # A read n/b times, B read m/b times, C written once
        assert t["bytes"]["l2"] == 4 * (2 * m * n * k // b + m * n)
        assert not t["l2_fits"]  # 2 x 64 MiB > 40 MiB
        assert t["bytes"]["hbm"] == t["bytes"]["l2"]
        assert t["bytes"]["smem"] == 4 * 2 * m * n * k // b + 4 * k * m * n * 2 // tt
        assert t["bytes"]["reg"] == 12 * m * n * k


def test_gemm_that_fits_in_l2_moves_compulsory_bytes():
    t = g.gemm_traffic(A100, 1024, 1024, 1024, 32, 32, 2, 2)
    assert t["l2_fits"]
    assert t["bytes"]["hbm"] == 4 * 3 * 1024 * 1024


def test_kernel_bounds_tell_the_story():
    # vector add waits on HBM; 16x16 tiles without register blocking wait on
    # shared memory; 128x128 tiles with 8x8 per thread reach the ALUs (A100)
    b = {k: g.kernel_time(A100, kern)["bound"] for k, kern in g.KERNELS.items()}
    assert b == {"vecadd": "hbm", "gemm16": "smem", "gemm128": "compute"}
    kt = g.kernel_time(A100, g.KERNELS["gemm128"])
    assert kt["total"] == kt["compute"] == 2 * 4096**3 / 19.5e12
    # on H100 the same 8x8 kernel is just shared-memory bound
    assert g.kernel_time(H100, g.KERNELS["gemm128"])["bound"] == "smem"


def test_kernel_time_is_the_slowest_level_and_utilisations():
    for p in (A100, H100):
        for kern in g.KERNELS.values():
            kt = g.kernel_time(p, kern)
            assert kt["total"] == max([kt["compute"]] + list(kt["times"].values()))
            assert max(list(kt["util"].values()) + [kt["compute_util"]]) == 1.0
            assert kt["achieved"] <= p["peak_fp32"] * (1 + 1e-12)


def test_flow_steps_are_linear_in_time():
    kt = g.kernel_time(A100, g.KERNELS["vecadd"])
    s = g.flow_steps(A100, g.KERNELS["vecadd"], 60)
    assert len(s) == 61
    assert s[0]["moved"]["hbm"] == 0 and s[60]["moved"]["hbm"] == kt["bytes"]["hbm"]
    assert s[30]["t"] == kt["total"] / 2
    assert s[30]["moved"]["l2"] == kt["bytes"]["l2"] / 2


# --- roofline ----------------------------------------------------------------


def test_roofline_is_min_of_peak_and_bandwidth_times_intensity():
    r = g.roofline_point(A100, 1.0, "fp32")
    assert r["perf"] == 1555e9 and r["bound"] == "memory"
    r = g.roofline_point(A100, 100.0, "fp32")
    assert r["perf"] == 19.5e12 and r["bound"] == "compute"
    r = g.roofline_point(A100, 100.0, "tensor")
    assert r["perf"] == 100 * 1555e9 and r["bound"] == "memory"  # below the tensor ridge (200.6)


def test_sweep_intensity_grows_with_the_tile():
    s = g.roofline_sweep(A100)
    # HBM intensity of a b x b tiled GEMM is about b/4 flop/byte (C's store is the rest)
    for st in s:
        n = 4096
        assert st["ai"] == pytest.approx(2 * n**3 / (4 * (2 * n**3 / st["tile"] + n * n)))
    ais = [st["ai"] for st in s]
    assert ais == sorted(ais)
    assert [st["bound"] for st in s].index("compute") == 6  # 64 x 64 crosses A100's FP32 ridge (12.5)


# --- SIMT ------------------------------------------------------------------------


def test_xorshift_known_values():
    # the standard xorshift32 sequence from seed 1 (Marsaglia 2003)
    assert g.xorshift32(1) == 270369
    assert g.xorshift32(270369) == 67634689


def test_simt_divergent_branch_serialises_both_paths():
    r = g.simt_steps("lt", 8, 3, 2, 2)
    phases = [s["phase"] for s in r["steps"]]
    assert phases == ["pre", "branch", "A", "A", "A", "B", "B", "C", "C"]
    assert r["steps"][2]["mask"] == 0xFF
    assert r["steps"][5]["mask"] == 0xFFFFFF00
    assert r["divergent"]
    # 4 full steps + 3 x 8 lanes + 2 x 24 lanes, over 9 issues of 32
    assert r["efficiency"] == (4 * 32 + 3 * 8 + 2 * 24) / (9 * 32)


def test_simt_uniform_branch_skips_the_other_path():
    r = g.simt_steps("uniform", 1, 3, 2, 2)
    assert [s["phase"] for s in r["steps"]].count("B") == 0
    assert r["efficiency"] == 1.0 and not r["divergent"]
    r = g.simt_steps("lt", 32, 3, 2, 2)  # every lane takes A
    assert r["efficiency"] == 1.0


def test_simt_even_odd():
    r = g.simt_steps("mod", 2, 1, 1, 0)
    assert r["taken"] == 0x55555555
    assert r["efficiency"] == (2 * 32 + 16 + 16) / (4 * 32)


# --- coalescing (CUDA guide §2.3.4.1) ----------------------------------------------


def test_consecutive_4_byte_words_use_four_sectors():
    r = g.coalesce_steps(4, 1, 0)
    assert r["sectors"] == 4 and r["fetched"] == 128 and r["efficiency"] == 1.0
    assert r["lines"] == 1


def test_stride_of_32_bytes_or_more_needs_a_sector_per_thread():
    for stride in (8, 16, 32):
        r = g.coalesce_steps(4, stride, 0)
        assert r["sectors"] == 32
        assert r["fetched"] == 1024 and r["efficiency"] == 0.125  # the guide's 12.5%


def test_misaligned_start_touches_one_more_sector():
    r = g.coalesce_steps(4, 1, 4)
    assert r["sectors"] == 5 and r["lines"] == 2
    assert r["efficiency"] == 128 / 160


def test_wide_elements_and_broadcast():
    assert g.coalesce_steps(16, 1, 0)["sectors"] == 16  # float4: 512 bytes, 100%
    assert g.coalesce_steps(16, 1, 0)["efficiency"] == 1.0
    r = g.coalesce_steps(4, 0, 0)  # every lane reads the same word
    assert r["sectors"] == 1 and r["efficiency"] == 4.0  # 128 bytes requested, 32 fetched


# --- bank conflicts (CUDA guide §2.3.4.2) ---------------------------------------------


def test_guide_strided_examples():
    assert g.bank_steps("stride", 0, 1)["degree"] == 1
    assert g.bank_steps("stride", 0, 2)["degree"] == 2  # "two-way bank conflict"
    assert g.bank_steps("stride", 0, 3)["degree"] == 1


def test_any_odd_stride_is_conflict_free_and_even_strides_conflict():
    for s in range(1, 34):
        d = g.bank_steps("stride", 0, s)["degree"]
        assert d == math.gcd(s, 32)


def test_transpose_column_is_32_way_and_padding_fixes_it():
    assert g.bank_steps("col", 0)["degree"] == 32
    assert g.bank_steps("col", 1)["degree"] == 1  # s[32][33]
    assert g.bank_steps("row", 0)["degree"] == 1
    assert g.bank_steps("col", 2)["degree"] == 2


def test_broadcast_is_conflict_free():
    r = g.bank_steps("broadcast", 0)
    assert r["degree"] == 1 and len(r["passes"][0]["lanes"]) == 32


def test_every_lane_is_served_exactly_once():
    for pat, pad, st in [("col", 0, 1), ("stride", 0, 4), ("col", 2, 1), ("broadcast", 0, 1)]:
        r = g.bank_steps(pat, pad, st)
        served = sorted(l for ps in r["passes"] for l in ps["lanes"])
        assert served == list(range(32))


# --- occupancy (CUDA guide §2.3.7, cuda_occupancy.h) -----------------------------------


def test_guide_threads_examples():
    # 768 threads per block: 2 blocks, (768 x 2) / 2048 = 75%
    assert g.occupancy(A100, 768, 32, 0)["occupancy"] == 0.75
    # 32 threads per block: the 32-block limit, 50%
    o = g.occupancy(A100, 32, 16, 0)
    assert o["blocks"] == 32 and o["limiters"] == ["blocks"] and o["occupancy"] == 0.5


def test_register_limit():
    # 64 regs/thread: 2048 per warp, 16384 per sub-partition -> 8 warps each, 32 per SM
    o = g.occupancy(A100, 256, 64, 0)
    assert o["regs_per_warp"] == 2048
    assert o["limits"]["regs"] == 4 and o["occupancy"] == 0.5
    # register count rounds up to the 256-register warp granule: 40 -> 1280 per warp
    assert g.occupancy(A100, 128, 40, 0)["regs_per_warp"] == 1280
    # 255 registers: one 1024-thread block cannot fit
    assert g.occupancy(A100, 1024, 255, 0)["blocks"] == 0
    assert g.occupancy(A100, 128, 256, 0)["blocks"] == 0  # over 255 per thread


def test_shared_memory_limit_includes_the_reserved_kilobyte():
    o = g.occupancy(A100, 256, 32, 48 * 1024)
    assert o["smem_alloc"] == 49 * 1024
    assert o["limits"]["smem"] == 164 // 49  # 3
    # the per-block maximum (163 KB) fits once; one byte more does not
    assert g.occupancy(A100, 256, 32, 163 * 1024)["blocks"] == 1
    assert g.occupancy(A100, 256, 32, 163 * 1024 + 1)["blocks"] == 0
    assert g.occupancy(H100, 256, 32, 227 * 1024)["blocks"] == 1
    # rounded up to 128 bytes
    assert g.occupancy(A100, 256, 32, 1)["smem_alloc"] == 1024 + 128


def test_occupancy_steps_end_with_the_rejected_block():
    s = g.occupancy_steps(A100, 256, 64, 49152)
    assert len(s["steps"]) == s["result"]["blocks"] + 1
    assert s["steps"][-1]["rejected"] and not s["steps"][0]["rejected"]
    last = s["steps"][-1]
    assert last["smem"] <= A100["smem_per_sm"] < last["smem"] + s["result"]["smem_alloc"]


# --- tile timeline ------------------------------------------------------------------------


def test_single_buffer_is_serial_for_load_and_compute():
    r = g.tile_timeline(6, 2.0, 3.0, 1.0, 1)
    # load and compute alternate; each store overlaps the next tile's load
    assert r["total"] == 6 * (2.0 + 3.0) + 1.0


def test_double_buffer_hides_the_loads_behind_compute():
    r = g.tile_timeline(6, 2.0, 3.0, 1.0, 2)
    assert r["total"] == 2.0 + 6 * 3.0 + 1.0
    # load-bound case: the loads set the pace
    r = g.tile_timeline(8, 4.5, 1.25, 0.5, 2)
    assert r["total"] == 8 * 4.5 + 1.25 + 0.5
