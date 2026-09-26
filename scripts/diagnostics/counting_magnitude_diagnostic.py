"""Does any CONTINUOUS readout scale with circle count, even though the
binarized top-5% KC-code overlap (counting_diagnostic.py) looked flat
across n=1..4? That overlap check discards magnitude information by
binarizing to a fixed-size top-k set; x_teach_output's actual value-mode
regression training uses continuous (unthresholded, weighted) KC
activation instead -- see its pre_act/err computation in fly_api.py. A
flat Jaccard result does not by itself rule out a usable magnitude
signal; check directly here before concluding the counting pivot failed
too.

Multiple reps per count (default 5) since a single random circle layout
per count is noisy -- we want the TREND across many layouts of the same
count, not one sample's activation level.

Usage:
    python scripts/diagnostics/counting_magnitude_diagnostic.py --size 32 --hops 3 --n-max 4 --reps 8
"""
import sys
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))       # .../my_experiments
for _p in (SCRIPT_DIR, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np

from fly_api import FlyBrainAPI
from agents.shared.paths import FLY_ROOT
from encoders.shapes.circle_shapes import circle_bitmap


def main(size, hops, n_max, radius, reps, seed):
    fly = FlyBrainAPI(mode="full", path=FLY_ROOT)
    fly.enable_scaling()
    rng = np.random.default_rng(seed)

    ns, vis, alpn, kc_raw = [], [], [], []
    for n in range(n_max + 1):
        for _ in range(reps):
            bm = circle_bitmap(n, rng, size=size, radius=radius)
            out = fly.step(image=bm, hops=hops)
            idx, val, _ = fly.encode(image=bm)
            h = fly._forward_pure(idx, val, hops=hops, thr=0.0)
            ns.append(n)
            vis.append(out.get("VIS_mean", 0.0))
            alpn.append(out.get("ALPN_mean", 0.0))
            kc_raw.append(float(h[fly.KC].mean()))  # continuous, NOT top-k thresholded

    ns = np.asarray(ns, dtype=float)
    print(f"=== continuous readout vs. circle count, n={reps} reps/count ===\n")
    for name, vals in (("VIS_mean", vis), ("ALPN_mean", alpn), ("KC_mean_raw (unthresholded)", kc_raw)):
        vals = np.asarray(vals, dtype=float)
        corr = float(np.corrcoef(ns, vals)[0, 1]) if vals.std() > 0 else float("nan")
        per_count = ", ".join(f"n={n}: {vals[ns == n].mean():.5f}" for n in range(n_max + 1))
        print(f"{name}\n  corr with count = {corr:.3f}\n  means: {per_count}\n")

    print("A correlation clearly above ~0.5-0.6 with a roughly monotonic per-count "
          "mean is a usable magnitude signal for x_teach_output's regression mode, "
          "even if the binarized top-5% code looked flat. Near-zero/non-monotonic "
          "here across all three readouts means the earlier pessimistic result "
          "holds and counting likely needs a different fix (bigger radius/n_max "
          "spread for a stronger ink-total signal, or deeper hops) rather than "
          "being fundamentally viable at these settings.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--n-max", type=int, default=4)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(size=a.size, hops=a.hops, n_max=a.n_max, radius=a.radius, reps=a.reps, seed=a.seed)
