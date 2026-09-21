"""Run this BEFORE building/trusting anything else in the tic-tac-toe
experiment. Same discipline as the bandit's innate-bias check and the
grid-world encoding checks -- verify the API's actual behavior on this new
encoding before writing agents against assumed behavior.

Checks two things:

1. Innate bias -- do different untrained boards get very different
   MB_pref values with zero training? (mirrors the bandit's innate odor
   bias finding; if large, calibrate_baseline() in the connectome agent
   is required, not optional).

2. Cross-board interference -- after training hard on ONE board, how much
   do OTHER, unrelated boards' values move? Whole-graph plasticity bled
   across stimuli in both the bandit and grid-world experiments; with 9
   shared cell-index-groups here (vs. one distinct pattern per state
   there) it is likely worse. This number tells you whether the
   compositional encoding is viable at all before you sink time into the
   three agents.

Prints CELL_DRIVE / GROUP_SIZE guidance based on what it finds. Nothing
here is wired up to the real fly-brain library's exact call signatures --
check fly_api.py's actual step()/train() kwargs (hops=, mode=, etc.) match
what's used below before running.
"""
import os
import random
import sys

# make sure fly_api.py / the encoders package are importable whether they
# live inside my_experiments/ (this script's own directory) or one level
# up at the repo root (this script's parent directory) -- covers both
# layouts without you having to move files around.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from fly_api import FlyBrainAPI
from encoders import board_odor_encoder as enc

N_SAMPLE_BOARDS = 40
HOPS = 1


def random_legal_board(max_depth=4):
    marks = [0] * 9
    to_move = 1
    for _ in range(random.randint(0, max_depth)):
        empties = [i for i, v in enumerate(marks) if v == 0]
        if not empties:
            break
        marks[random.choice(empties)] = to_move
        to_move *= -1
    return tuple(marks)


def main():
    fly = FlyBrainAPI(mode="mb")
    # dedupe: random_legal_board() will produce repeats at shallow depths
    # (the empty board especially), so this is usually < N_SAMPLE_BOARDS
    boards = list({random_legal_board() for _ in range(N_SAMPLE_BOARDS)})

    # 1. innate bias
    print("=== innate bias (untrained MB_pref per board) ===")
    baseline_vals = {b: fly.step(odor=enc.board_to_vector(b), hops=HOPS)["MB_pref"] for b in boards}
    vals = list(baseline_vals.values())
    print(f"n={len(vals)}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
          f"min={min(vals):.4f}  max={max(vals):.4f}")

    # How much of that spread can the same feature set the connectome
    # agent's calibrate_baseline() actually fits (9 marks + 36 pairwise
    # products, NOT the raw 685-dim vector) explain? High R^2 -> the
    # baseline will generalize to boards it never saw during calibration.
    # Low R^2 -> real bias will leak through uncorrected on unseen boards.
    X = np.array([enc.board_baseline_features(b) for b in boards])
    y = np.array(vals)
    X_aug = np.hstack([X, np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    resid = y - X_aug @ coef
    ss_res, ss_tot = float(np.sum(resid**2)), float(np.sum((y - y.mean())**2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"linear-baseline fit: R^2={r2:.3f}  residual std={np.std(resid):.4f}")
    print("-> if R^2 is well below ~0.9, the connectome agent's value "
          "estimates will carry real leftover bias between boards even "
          "after calibrate_baseline -- worth increasing the calibration "
          "sample size or checking whether the residual is concentrated "
          "in a particular kind of board (e.g. near-win boards) before "
          "trusting comparisons between close-valued afterstates.\n")

    # 2. cross-board interference -- the trained board is FORCED into the
    # probe set (a random sample could miss it entirely, which defeats the
    # whole comparison), and the degenerate empty board (all-zero vector,
    # mathematically guaranteed delta=0.0 regardless of training) is
    # excluded from the "unrelated" candidates.
    print("=== cross-board interference ===")
    nonempty = [b for b in boards if any(m != 0 for m in b)]
    if not nonempty:
        print("no non-empty boards sampled -- rerun (increase N_SAMPLE_BOARDS "
              "or max_depth in random_legal_board)")
        return
    train_board = nonempty[0]
    other_candidates = [b for b in nonempty if b != train_board]
    probe_boards = [train_board] + random.sample(other_candidates, min(4, len(other_candidates)))

    before = {b: fly.step(odor=enc.board_to_vector(b), hops=HOPS)["MB_pref"] for b in probe_boards}
    for _ in range(10):
        fly.train(odor=enc.board_to_vector(train_board), reward=1.0, hops=HOPS)
    after = {b: fly.step(odor=enc.board_to_vector(b), hops=HOPS)["MB_pref"] for b in probe_boards}

    trained_delta = after[train_board] - before[train_board]
    for b in probe_boards:
        tag = "TRAINED BOARD" if b == train_board else "unrelated"
        d = after[b] - before[b]
        frac = (f"  ({100 * d / trained_delta:+.0f}% of trained delta)"
                if b != train_board and trained_delta != 0 else "")
        print(f"{tag:14s} board={b}  before={before[b]:+.4f}  after={after[b]:+.4f}  delta={d:+.4f}{frac}")
    print("-> the %-of-trained-delta column on unrelated boards is the real "
          "number to look at. If it's a large fraction (say >20-30%), "
          "cross-board interference will dominate learning at tic-tac-toe's "
          "scale and the encoding (GROUP_SIZE / CELL_DRIVE / which cells "
          "share indices) needs rethinking before running the full experiment.")


if __name__ == "__main__":
    main()
