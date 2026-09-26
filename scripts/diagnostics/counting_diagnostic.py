"""Same diagnostic idea as shapes_diagnostic.py, adapted for counts instead
of digit identities: check KC-code overlap between different circle-counts
BEFORE running a full training loop. Forward passes only, no training.

Usage:
    python scripts/diagnostics/counting_diagnostic.py --size 32 --hops 3 --n-max 4
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


def kc_code(fly, bitmap, hops, kc_frac=0.05):
    idx, val, _ = fly.encode(image=bitmap)
    h = fly._forward_pure(idx, val, hops=hops, thr=0.0)
    kcs = h[fly.KC]
    k = max(1, int(len(kcs) * kc_frac))
    thr = np.sort(kcs)[-k]
    return set(np.where(kcs >= thr)[0].tolist())


def main(size, hops, n_max, radius, seed):
    fly = FlyBrainAPI(mode="full", path=FLY_ROOT)
    fly.enable_scaling()
    rng = np.random.default_rng(seed)
    counts = list(range(n_max + 1))
    codes = {n: kc_code(fly, circle_bitmap(n, rng, size=size, radius=radius), hops=hops)
             for n in counts}

    print(f"=== size={size}x{size}, hops={hops}, radius={radius}: KC code sizes ===")
    for n in counts:
        print(f"  n={n}: {len(codes[n])} active KCs")

    print(f"\n=== pairwise Jaccard overlap (0=disjoint, 1=identical) ===")
    print("     " + " ".join(f"  n{n} " for n in counts))
    for n1 in counts:
        row = []
        for n2 in counts:
            inter = len(codes[n1] & codes[n2])
            union = len(codes[n1] | codes[n2])
            row.append(inter / union if union else 0.0)
        print(f"  n{n1}  " + " ".join(f"{v:.2f}  " for v in row))

    print("\nLook for a roughly MONOTONIC pattern (n and n+1 overlap more than n "
          "and n+2, etc.) -- that's the signature of a genuine graded magnitude "
          "code, as opposed to the flat near-1.0-everywhere pattern seen for "
          "digit identity.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--n-max", type=int, default=4)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(size=a.size, hops=a.hops, n_max=a.n_max, radius=a.radius, seed=a.seed)
