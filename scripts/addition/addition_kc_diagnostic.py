"""KC-code overlap diagnostic for the addition experiments -- same idea as
MBValueCache.overlap_stats() (grid-world/tic-tac-toe) and shapes_
diagnostic.py (digit shape-identity): directly checks whether hops=1 KC
codes for different (a,b,c) triples are distinguishable, instead of
inferring a representational ceiling indirectly from training curves.

Three checks:
  1. Random-triple baseline -- how much overlap is just generically
     typical for this encoding (reference point for the other two).
  2. Diff sweep -- for a FIXED (a,b), how does overlap with the correct
     triple's code change as c moves away from the true sum? If overlap
     stays high even at diff=+-1 (a near-miss -- exactly the negative-
     example distribution used in training), no amount of training can
     separate "correct" from "off by one".
  3. (two-digit only) Carry vs no-carry -- a matched pair with the same
     tens digit where units crosses 10 or doesn't. If the codes barely
     differ, the representation isn't registering the one thing that
     makes carrying discrete.

    python addition_kc_diagnostic.py                 # single-digit
    python addition_kc_diagnostic.py --two-digit
"""
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR), os.path.dirname(os.path.dirname(_THIS_DIR))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np

from fly_api import FlyBrainAPI
from agents.shared.paths import FLY_ROOT
from encoders.addition.addition_encoder import AdditionEncoder
from encoders.addition.addition_encoder_2digit import TwoDigitAdditionEncoder


def kc_code(brain, vec, hops=1, kc_frac=0.05):
    idx, val, _ = brain.encode(odor=vec)
    h = brain._forward_pure_mb(idx, val, hops=hops, thr=0.0)
    kcs = h[brain.KC]
    k = max(1, int(len(kcs) * kc_frac))
    top = set(np.argsort(kcs)[-k:].tolist())
    return top, int((kcs > 0).sum())


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def run_diff_sweep(brain, encoder, op_max, n_pairs=20, seed=0):
    rng = np.random.default_rng(seed)
    diffs = [0, 1, -1, 2, -2, 3, -3]
    overlaps = {d: [] for d in diffs}
    empty_count = 0
    for _ in range(n_pairs):
        a = int(rng.integers(0, op_max + 1))
        b = int(rng.integers(0, op_max + 1))
        correct = a + b
        c_max = 2 * op_max
        code0, active0 = kc_code(brain, encoder.encode(a, b, correct))
        if active0 == 0:
            empty_count += 1
        for d in diffs:
            c = correct - d  # diff = a+b-c = d  ->  c = correct-d
            if c < 0 or c > c_max:
                continue
            code_c, _ = kc_code(brain, encoder.encode(a, b, c))
            overlaps[d].append(jaccard(code0, code_c))
    print(f"  (empty codes at diff=0: {empty_count}/{n_pairs})")
    print(f"  {'diff':>5} {'mean_jaccard':>13} {'n':>4}")
    for d in diffs:
        vals = overlaps[d]
        if vals:
            print(f"  {d:>5} {np.mean(vals):>13.3f} {len(vals):>4}")


def run_carry_check(brain, encoder, n_pairs=20, seed=1):
    """Matched pair, same tens digit of both operands, units digit of the
    second operand chosen so one case crosses 10 and the other doesn't --
    isolates the carry itself from everything else."""
    rng = np.random.default_rng(seed)
    overlaps = []
    for _ in range(n_pairs):
        au = int(rng.integers(1, 9))          # 1..8
        bu_nocarry = int(rng.integers(0, 10 - au))
        bu_carry = int(rng.integers(10 - au, 10))
        at = int(rng.integers(0, 4))
        bt = int(rng.integers(0, 4))
        a = at * 10 + au
        b_nc, b_c = bt * 10 + bu_nocarry, bt * 10 + bu_carry
        code_nc, _ = kc_code(brain, encoder.encode(a, b_nc, a + b_nc))
        code_c, _ = kc_code(brain, encoder.encode(a, b_c, a + b_c))
        overlaps.append(jaccard(code_nc, code_c))
    print(f"  carry vs no-carry (matched, correct answer either way): "
          f"mean_jaccard={np.mean(overlaps):.3f} (n={len(overlaps)})")


def run_random_baseline(brain, encoder, op_max, n=60, seed=2):
    rng = np.random.default_rng(seed)
    codes = []
    for _ in range(n):
        a = int(rng.integers(0, op_max + 1))
        b = int(rng.integers(0, op_max + 1))
        c = int(rng.integers(0, 2 * op_max + 1))
        code, _ = kc_code(brain, encoder.encode(a, b, c))
        codes.append(code)
    vals = [jaccard(codes[i], codes[j]) for i in range(len(codes)) for j in range(i + 1, len(codes))]
    print(f"  random-triple baseline: mean_jaccard={np.mean(vals):.3f} "
          f"p95={np.quantile(vals, 0.95):.3f} (n_pairs={len(vals)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-digit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    two_digit = args.two_digit
    op_max = 49 if two_digit else 9
    brain = FlyBrainAPI(mode="mb", path=FLY_ROOT)
    live_path = os.path.join(FLY_ROOT, "my_experiments", "results", "gridworld", "live_alpn_indices.npy")
    encoder = (TwoDigitAdditionEncoder(live_path, seed=args.seed) if two_digit
              else AdditionEncoder(live_path, n_max=op_max, seed=args.seed))

    print(f"=== KC-code overlap diagnostic: {'two-digit' if two_digit else 'single-digit'} ===\n")

    print("1. Random-triple baseline overlap (any triples, unrelated):")
    run_random_baseline(brain, encoder, op_max, seed=args.seed)

    print("\n2. Overlap vs residual diff (fixed a,b; c moved away from the correct sum;")
    print("   diff=0 is self-overlap, =1.000 by construction, shown as the reference point):")
    run_diff_sweep(brain, encoder, op_max, seed=args.seed)

    if two_digit:
        print("\n3. Carry vs no-carry (matched a, same tens digit, units crossing 10 or not):")
        run_carry_check(brain, encoder, seed=args.seed)


if __name__ == "__main__":
    main()
