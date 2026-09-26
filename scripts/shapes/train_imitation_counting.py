"""Tier 3 for counting: imitation-style dense supervision, same paradigm
shift as train_imitation_tictactoe.py -- instead of one random example
per round (Tier 1/2's online RL-flavored loop), directly and repeatedly
train on a fixed, generated pool of examples per count, then evaluate on
a FRESH held-out pool (never trained on) to get a genuine generalization
number instead of a memorization one.

Applicability note (answering "is this even applicable here"): tic-tac-
toe's imitation tier worked because a ground-truth *label* (the minimax-
optimal move) exists for every state independent of any reward signal.
Counting has the same property -- the true count is definitionally known
for every generated image, no RL/exploration needed to obtain it -- so
yes, this transfers directly and arguably fits BETTER here than it did
for tic-tac-toe, since Tier 1/2 here already train on ground truth every
round anyway (there's no meaningful reward-vs-imitation distinction the
way there was for tic-tac-toe's win/loss signal). The real lever this
tier isolates is DENSITY/REPETITION/ORDERING of supervision, not the
presence of a label -- so toggle SHUFFLE/SLEEP the same way
train_imitation_tictactoe.py did and compare against Tier 1/2's numbers
on the SAME held-out evaluation protocol below.

    python scripts/shapes/train_imitation_counting.py
    python scripts/shapes/train_imitation_counting.py --shuffle --sleep-every 40
"""
import sys
import os
import csv
import time
import random
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_d = _HERE
for _ in range(6):
    if _d not in sys.path:
        sys.path.insert(0, _d)
    _d = os.path.dirname(_d)

import numpy as np

from encoders.shapes.circle_shapes import circle_bitmap
from agents.shapes.connectome_count_agent import ConnectomeCountAgent

N_MAX = 4
IMG_SIZE = 32
RADIUS = 3
LAYOUTS_PER_COUNT = 40   # distinct random circle layouts per count, for training
HELD_OUT_PER_COUNT = 20  # fresh layouts per count, NEVER trained on, for eval
REPS_PER_LAYOUT = 3
PROGRESS_EVERY = 100


def _find_results_dir():
    d = _HERE
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "results")):
            return os.path.join(d, "results", "counting")
        d = os.path.dirname(d)
    return os.path.join(os.path.dirname(os.path.dirname(_HERE)), "results", "counting")


def build_pool(rng, n_per_count, size, radius):
    pool = []
    for n in range(N_MAX + 1):
        for _ in range(n_per_count):
            pool.append((n, circle_bitmap(n, rng, size=size, radius=radius)))
    return pool


def evaluate(agent, held_out):
    correct = abs_err_sum = 0
    for count, bitmap in held_out:
        pred, _ = agent.predict(bitmap)
        correct += int(pred == count)
        abs_err_sum += abs(pred - count)
    n = len(held_out)
    return correct / n, abs_err_sum / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    ap.add_argument("--sleep-every", type=int, default=None,
                    help="e.g. 40 to enable periodic sleep(); None=off")
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--trials", type=int, default=3)
    a = ap.parse_args()

    results_dir = _find_results_dir()
    os.makedirs(results_dir, exist_ok=True)

    train_rng = np.random.default_rng(0)
    holdout_rng = np.random.default_rng(999)  # different seed: disjoint layouts

    train_pool = build_pool(train_rng, LAYOUTS_PER_COUNT, IMG_SIZE, RADIUS)
    held_out = build_pool(holdout_rng, HELD_OUT_PER_COUNT, IMG_SIZE, RADIUS)

    if a.shuffle:
        random.Random(0).shuffle(train_pool)

    agent = ConnectomeCountAgent(n_max=N_MAX, teach_eta=a.eta)
    calib_bitmaps = [circle_bitmap(n, np.random.default_rng(1), size=IMG_SIZE, radius=RADIUS)
                      for n in range(N_MAX + 1) for _ in range(4)]
    agent.calibrate_baseline(calib_bitmaps)  # BEFORE any training

    pre_acc, pre_mae = evaluate(agent, held_out)
    print(f"untrained baseline on held-out set: accuracy={pre_acc:.3f} MAE={pre_mae:.3f}")

    suffix = f"_shuffle{a.shuffle}_sleep{a.sleep_every}"
    csv_path = os.path.join(results_dir, f"imitation_counting_results{suffix}.csv")
    t0 = time.time()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "held_out_accuracy", "held_out_mae", "output_drift_pct"])
        w.writerow([-1, pre_acc, pre_mae, 0.0])

        for i, (count, bitmap) in enumerate(train_pool):
            for _ in range(REPS_PER_LAYOUT):
                agent.train_on(bitmap, count, trials=a.trials)

            if a.sleep_every and (i + 1) % a.sleep_every == 0:
                agent.sleep()

            if (i + 1) % PROGRESS_EVERY == 0 or i == len(train_pool) - 1:
                acc, mae = evaluate(agent, held_out)
                elapsed = time.time() - t0
                print(f"{i + 1}/{len(train_pool)} layouts, elapsed={elapsed:.0f}s "
                      f"held_out_accuracy={acc:.3f} held_out_mae={mae:.3f} "
                      f"output_drift={agent.output_drift_pct():.2f}%")
                w.writerow([i, acc, mae, f"{agent.output_drift_pct():.4f}"])
                f.flush()

    final_acc, final_mae = evaluate(agent, held_out)
    print(f"\nfinal held-out accuracy: {final_acc:.3f}  MAE: {final_mae:.3f}  "
          f"(untrained baseline was {pre_acc:.3f} / {pre_mae:.3f})")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
