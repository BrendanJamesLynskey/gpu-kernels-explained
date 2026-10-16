"""
Tests of chapter 12's model (reference/blackwell.py): the tensor-core
presets' sources, the ridge points, the block-scaled formats against their
definitions, and the pipeline's schedule against its closed forms and its
own invariants. Run: python3 -m pytest -q tests/python
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference"))

import blackwell as b  # noqa: E402
import gpu_model as g  # noqa: E402


@pytest.mark.parametrize("pid", b.TC_ORDER)
def test_every_field_is_sourced(pid):
    for key, f in b.TC_PRESETS[pid]["fields"].items():
        assert f["src"] in b.SOURCES or f["src"] in g.SOURCES, key
        assert f["st"] in {"spec", "rule", "measured", "derived", "none"}, key
        assert (f["v"] is None) == (f["st"] == "none"), key


def test_a100_h100_are_the_cuda_core_presets():
    for pid in ("a100", "h100"):
        p, t = g.preset(pid), b.tc_preset(pid)
        assert t["peak_bf16"] == p["peak_tensor"]
        assert t["peak_int8"] == p["peak_int8_tensor"]
        assert t["hbm_bw"] == p["hbm_bw"] and t["sms"] == p["sms"]


def test_b200_matches_the_sibling_sites():
    # the Racks and Accelerators sites' B200 (the GPU of a GB200 NVL72)
    t = b.tc_preset("b200")
    assert (t["peak_bf16"], t["peak_fp8"], t["peak_int8"]) == (2.5e15, 5e15, 5e15)
    assert (t["hbm_bw"], t["hbm_bytes"]) == (8e12, 186e9)
    assert t["peak_fp4"] == 2 * t["peak_fp8"] == 4 * t["peak_bf16"]


def test_ridges():
    r = {pid: b.ridges(pid) for pid in b.TC_ORDER}
    assert r["a100"]["bf16"] == pytest.approx(312e12 / 1555e9)
    assert r["b200"]["bf16"] == 312.5 and r["b200"]["fp4"] == 1250
    assert r["a100"]["fp8"] is None and r["h100"]["fp4"] is None
    assert r["a100"]["bf16"] < r["h100"]["bf16"] < r["b200"]["bf16"]


def test_block_formats():
    # OCP MX: 32 elements share an 8-bit E8M0 scale; NVFP4: 16 share an E4M3 scale
    assert b.bits_per_value("mxfp4") == 4.25
    assert b.bits_per_value("nvfp4") == 4.5
    assert b.bits_per_value("mxfp8") == 8.25
    assert b.weight_bytes("bf16", 8192, 8192) == 8192 * 8192 * 2
    assert b.weight_bytes("nvfp4", 8192, 8192) == 8192 * 8192 / 2 + 8192 * 512


def test_stages():
    s = b.stage_bytes(128, 256, 64, 2)
    assert s == 48 * 1024
    assert b.max_stages(228 * 1024, s) == 4
    assert b.max_stages(164 * 1024, s) == 3


def _steady_runs():
    for st in range(1, 7):
        for ld in (2, 4, 12, 24):
            for iss in (1, 2, 5):
                for mma in (2, 4, 6):
                    yield st, ld, iss, mma


@pytest.mark.parametrize("st,ld,iss,mma", list(_steady_runs()))
def test_steady_state_interval(st, ld, iss, mma):
    """Within one long tile, S consecutive MMAs take max(S mma, S issue, load + mma)."""
    s = b.pipeline(1, 60, st, 2, ld, iss, mma, 0)
    want = max(st * mma, st * iss, ld + mma)
    ms = s["mmaStart"]
    for g0 in range(30, 60 - st):
        assert ms[g0 + st] - ms[g0] == want
    assert s["steady"] == -(-want // st)


@pytest.mark.parametrize("over", [{}, {"accs": 1}, {"stages": 1}, {"epi": 30}, {"load": 24, "stages": 6}])
def test_invariants(over):
    s = b.pipeline_default(**over)
    p = s["params"]
    n = p["tiles"] * p["ksteps"]
    for gg in range(n):
        assert s["mmaStart"][gg] >= s["loadDone"][gg]  # data before the MMA
        if gg:
            assert s["mmaStart"][gg] >= s["mmaDone"][gg - 1]  # one tensor core
            assert s["loadStart"][gg] >= s["loadStart"][gg - 1] + p["issue"]
        if gg >= p["stages"]:  # a slot is reused only once it is free
            assert s["loadStart"][gg] >= s["mmaDone"][gg - p["stages"]]
    for f in b.frames(s):
        assert sum(1 for x in f["slots"] if x[0] == "mma") <= 1
    assert s["total"] >= s["busy"]


def test_double_buffered_accumulators_hide_the_epilogue():
    one = b.pipeline_default(accs=1)
    two = b.pipeline_default(accs=2)
    p = b.PIPE_DEFAULT
    # with two buffers and an epilogue shorter than a tile's MMAs, the tensor
    # core never waits: the run is the first load, the MMAs and the last epilogue
    assert p["epi"] <= p["ksteps"] * two["steady"]
    assert two["stall"] == 0
    assert two["total"] == p["load"] + two["busy"] + p["epi"]
    # with one, every tile but the last waits for its predecessor's epilogue
    assert one["total"] > two["total"]
