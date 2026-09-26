"""Diagnostic: how much do different digits' top-5% KC codes overlap at
the KC layer itself, BEFORE any x_add_output/x_teach_output training?

Forward passes only (fly.encode + fly._forward_pure), no train() and no
x_teach_output -- cheap (a handful of full-mode step()-equivalent calls,
not hundreds), so run this FIRST at any candidate resolution before
committing to a full scripts/shapes/run_shapes.py run at that resolution.

Why this matters: connectome_shape_agent.py's x_teach_output only changes
the INCOMING edges of the new output nodes -- it cannot fix two digits
that already produce near-identical KC codes upstream. If two digits'
codes overlap heavily here, no amount of output-layer training will ever
separate them; that's a representation-layer problem, not a training
problem, and only a change that affects the KC code itself (resolution,
a different encoding, or a different task like counting) can address it.
Same diagnostic-before-trusting discipline as tictactoe_diagnostic.py's
cross-board interference check.

Usage:
    python encoders/shapes/shapes_diagnostic.py --size 16
    python encoders/shapes/shapes_diagnostic.py --size 32
    python encoders/shapes/shapes_diagnostic.py --size 64
Compare the resulting overlap matrices across sizes before picking one.
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
from encoders.shapes.digit_shapes_encoder import digit_bitmap, SEGMENTS


def kc_code(fly, bitmap, hops, kc_frac=0.05):
    idx, val, _ = fly.encode(image=bitmap)
    h = fly._forward_pure(idx, val, hops=hops, thr=0.0)
    kcs = h[fly.KC]
    k = max(1, int(len(kcs) * kc_frac))
    thr = np.sort(kcs)[-k]
    return set(np.where(kcs >= thr)[0].tolist())


def main(size, hops, digits=None):
    fly = FlyBrainAPI(mode="full", path=FLY_ROOT)
    fly.enable_scaling()
    digits = sorted(digits) if digits else sorted(SEGMENTS)
    codes = {d: kc_code(fly, digit_bitmap(d, size=size), hops=hops) for d in digits}

    print(f"=== size={size}x{size}, hops={hops}: KC code sizes ===")
    for d in digits:
        print(f"  digit {d}: {len(codes[d])} active KCs")

    print(f"\n=== pairwise Jaccard overlap (0=disjoint, 1=identical KC codes) ===")
    header = "     " + " ".join(f"  d{d} " for d in digits)
    print(header)
    worst = (-1.0, None, None)
    for d1 in digits:
        row = []
        for d2 in digits:
            inter = len(codes[d1] & codes[d2])
            union = len(codes[d1] | codes[d2])
            j = inter / union if union else 0.0
            row.append(j)
            if d1 != d2 and j > worst[0]:
                worst = (j, d1, d2)
        print(f"  d{d1}  " + " ".join(f"{v:.2f}  " for v in row))

    print(f"\nHighest off-diagonal overlap: digits {worst[1]} and {worst[2]} at "
          f"{worst[0]:.2f} -- if this is >~0.5, expect predict() to confuse "
          f"this pair regardless of how much output-layer training you do; "
          f"if it's low but the trained model still confuses this pair, the "
          f"problem is more likely in x_teach_output's training dynamics, "
          f"not the representation.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--hops", type=int, default=2,
                    help="full mode defaults to 2 (calibrated for the olfactory "
                         "ORN->ALPN->KC path); README.md notes vision needs 3-4 "
                         "hops to meaningfully reach KC -- try --hops 3 or 4 here "
                         "before changing resolution/font/task")
    ap.add_argument("--kc-frac", type=float, default=0.05)
    ap.add_argument("--digits", type=int, nargs="+", default=None,
                    help="test a specific subset instead of all of SEGMENTS, "
                         "e.g. --digits 0 1 7 8 to try a maximally-distinct set")
    a = ap.parse_args()
    main(size=a.size, hops=a.hops, digits=a.digits)
