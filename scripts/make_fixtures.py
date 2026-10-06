"""
Writes the site's model data and the parity fixtures from the Python
reference (reference/gpu_model.py):

  src/data/presets.json          the presets, sources and kernels the site shows
  tests/fixtures/gpu_fixtures.json  reference results the TypeScript port must
                                 reproduce exactly (tests/unit/model.test.ts)

    python3 scripts/make_fixtures.py          # write both
    python3 scripts/make_fixtures.py --check  # fail if either is out of date (CI)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))

import gpu_model as g  # noqa: E402

PRESET_IDS = ("a100", "h100")

# The parameter grids the fixtures cover (the widgets' controls stay inside them).
AI_GRID = [2.0**e for e in range(-4, 11)] + [0.25, 1.5, 12.5, 20.0, 31.508]
OCC_THREADS = [32, 64, 96, 128, 192, 256, 384, 512, 640, 768, 1024, 1056]
OCC_REGS = [16, 24, 32, 40, 48, 64, 72, 96, 128, 168, 200, 255, 256]
OCC_SMEM = [0, 1024, 4096, 8192, 16384, 32768, 49152, 65536, 102400, 166912, 200000, 232448]
COALESCE = [(eb, st, off) for eb in (1, 2, 4, 8, 16) for st in (0, 1, 2, 3, 4, 8, 16, 32, 33) for off in (0, 4, 16, 64)]
BANKS = (
    [("row", pad, 1) for pad in range(0, 5)]
    + [("col", pad, 1) for pad in range(0, 5)]
    + [("stride", 0, s) for s in range(1, 34)]
    + [("broadcast", 0, 1)]
)
SIMT = (
    [("lt", k) for k in (0, 1, 8, 16, 31, 32)]
    + [("mod", k) for k in (1, 2, 4, 8, 32)]
    + [("uniform", k) for k in (0, 1)]
    + [("data", k) for k in (0, 10, 50, 90, 100)]
)
SIMT_LENS = [(3, 2, 2), (1, 1, 1), (5, 0, 3), (0, 4, 1)]
TIMELINES = [
    (1, 2.0, 3.0, 1.0, 1),
    (6, 2.0, 3.0, 1.0, 1),
    (6, 2.0, 3.0, 1.0, 2),
    (6, 2.0, 3.0, 1.0, 3),
    (8, 4.5, 1.25, 0.5, 2),
    (8, 0.1, 0.7, 0.3, 2),
]
# the overlap animation's scenarios (chapter 10): tiles, load, compute, store
TIMELINE_SCENARIOS = [(6, 2.0, 3.0, 1.0), (6, 3.0, 1.5, 1.0), (6, 1.0, 1.0, 1.0)]
SOFTMAX_SEEDS = (127, 142, 101)
DEQUANT_SEEDS = (11, 21, 31)
DEQUANT_SCALES = (0.0625, 0.125, 0.25)
SOFTMAX_BLOCKS = (2, 4, 8, 16)
FLASH = [(n, d, b) for n in (512, 1024) for d in (64, 128) for b in (64, 128)]
ATTN = [(n, d, b) for n in (1024, 2048, 4096, 8192, 16384) for d in (64, 128) for b in (64, 128)]


def site_data() -> dict:
    return {
        "sources": g.SOURCES,
        "common": g.COMMON,
        "presets": [g.PRESETS[pid] for pid in PRESET_IDS],
        "kernels": g.KERNELS,
        "sweepTiles": list(g.SWEEP_TILES),
        "gemmVariants": g.GEMM_VARIANTS,
        "gemmOrder": list(g.GEMM_ORDER),
        "gemmMarch": g.GEMM_MARCH,
        "splitK": {"shape": g.SPLITK_SHAPE, "splits": list(g.SPLITK_SPLITS)},
        "quantFormats": g.QUANT_FORMATS,
        "quantOrder": list(g.QUANT_ORDER),
        "quantBatches": list(g.QUANT_BATCHES),
        "quantShape": g.QUANT_SHAPE,
    }


def fixtures() -> dict:
    out: dict = {"presets": {}, "simt": [], "coalesce": [], "banks": [], "timeline": []}
    for pid in PRESET_IDS:
        p = g.preset(pid)
        occ = []
        for t in OCC_THREADS:
            for r in OCC_REGS:
                for s in OCC_SMEM:
                    occ.append({"threads": t, "regs": r, "smem": s, "out": g.occupancy(p, t, r, s)})
        out["presets"][pid] = {
            "flat": p,
            "derived": g.derived(p),
            "kernels": {kid: g.kernel_time(p, k) for kid, k in g.KERNELS.items()},
            "flow": {kid: g.flow_steps(p, k, 60) for kid, k in g.KERNELS.items()},
            "roofline": [
                {"ai": ai, "engine": e, "out": g.roofline_point(p, ai, e)}
                for ai in AI_GRID
                for e in ("fp32", "tensor")
            ],
            "sweep": g.roofline_sweep(p),
            "occupancy": occ,
            "gemmVariants": {v: g.gemm_variant(p, v) for v in g.GEMM_ORDER},
            "splitK": g.split_k_sweep(p),
            "quant": {f: g.quant_sweep(p, f) for f in g.QUANT_ORDER},
            "occupancySteps": [
                {"threads": t, "regs": r, "smem": s, "out": g.occupancy_steps(p, t, r, s)}
                for (t, r, s) in [(256, 32, 0), (256, 64, 49152), (1024, 64, 0), (128, 255, 0), (64, 32, 102400), (32, 16, 0)]
            ],
        }
    for cond, k in SIMT:
        for a, b, c in SIMT_LENS:
            out["simt"].append({"cond": cond, "k": k, "lens": [a, b, c], "out": g.simt_steps(cond, k, a, b, c)})
    out["laneData"] = g.lane_data(2024)
    for eb, st, off in COALESCE:
        out["coalesce"].append({"eb": eb, "stride": st, "offset": off, "out": g.coalesce_steps(eb, st, off)})
    for pat, pad, st in BANKS:
        out["banks"].append({"pattern": pat, "pad": pad, "stride": st, "out": g.bank_steps(pat, pad, st)})
    for n, tl, tc, ts, b in TIMELINES:
        out["timeline"].append({"args": [n, tl, tc, ts, b], "out": g.tile_timeline(n, tl, tc, ts, b)})
    out["timelineSteps"] = []
    for n, tl, tc, ts in TIMELINE_SCENARIOS:
        for b in (1, 2, 3):
            t = g.tile_timeline(n, tl, tc, ts, b)
            out["timelineSteps"].append({"args": [n, tl, tc, ts, b], "timeline": t, "steps": g.timeline_steps(t)})
    out["gemmMarch"] = {v: g.gemm_march(v) for v in g.GEMM_ORDER}
    out["reduce"] = {k: g.reduce_steps(k) for k in g.REDUCE_KINDS}
    out["softmax"] = [
        {"seed": sd, "block": b, "out": g.online_softmax(g.softmax_inputs(16, sd), b)}
        for sd in SOFTMAX_SEEDS
        for b in SOFTMAX_BLOCKS
    ]
    out["flash"] = [{"args": [n, d, b, b], "out": g.flash_steps(n, d, b, b)} for n, d, b in FLASH]
    out["attention"] = [{"args": [n, d, b, b], "out": g.attention_traffic(n, d, b, b)} for n, d, b in ATTN]
    out["dequant"] = [
        {"seed": sd, "scale": sc, "out": g.dequant_steps(sd, sc)} for sd in DEQUANT_SEEDS for sc in DEQUANT_SCALES
    ]
    return out


def dump(obj: dict, indent: int | None = 1) -> str:
    return json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False) + "\n"


# (generator, JSON indent): the fixtures are compact, the site data readable.
TARGETS = {
    ROOT / "src" / "data" / "presets.json": (site_data, 1),
    ROOT / "tests" / "fixtures" / "gpu_fixtures.json": (fixtures, None),
}


def main() -> int:
    check = "--check" in sys.argv
    stale = []
    for path, (fn, indent) in TARGETS.items():
        text = dump(fn(), indent)
        if check:
            if not path.exists() or path.read_text() != text:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"wrote {path.relative_to(ROOT)} ({len(text):,} bytes)")
    if stale:
        print("out of date (run python3 scripts/make_fixtures.py):", ", ".join(stale))
        return 1
    if check:
        print("fixtures and site data are up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
