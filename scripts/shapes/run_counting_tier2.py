"""Tier 2 runner: connectome only, using ConnectomeCountAgentV2's
error-adaptive teaching (see that file's docstring). Baselines are not
re-run here -- same flagged asymmetry as tic-tac-toe's Tier 2, and their
Tier-1 numbers from run_counting.py's own CSV are the comparison point.

    python scripts/shapes/run_counting_tier2.py
    python scripts/shapes/run_counting_tier2.py --tol 0.5 --max-extra-trials 6
"""
import sys
import os
import csv
import time
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_d = _HERE
for _ in range(6):
    if _d not in sys.path:
        sys.path.insert(0, _d)
    _d = os.path.dirname(_d)

import numpy as np

from encoders.shapes.circle_shapes import sample_random, circle_bitmap
from agents.shapes.connectome_count_agent_v2 import ConnectomeCountAgentV2

N_ROUNDS = 400
N_MAX = 4
IMG_SIZE = 32
RADIUS = 3
SLEEP_EVERY = 50
PROGRESS_EVERY = 20
N_CALIB_PER_COUNT = 4


def _find_results_dir():
    d = _HERE
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "results")):
            return os.path.join(d, "results", "counting")
        d = os.path.dirname(d)
    return os.path.join(os.path.dirname(os.path.dirname(_HERE)), "results", "counting")


RESULTS_DIR = _find_results_dir()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--tol", type=float, default=0.5, help="stop extra passes once |pred-count|<=tol")
    ap.add_argument("--max-extra-trials", type=int, default=6)
    ap.add_argument("--trials-per-pass", type=int, default=2)
    a = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    agent = ConnectomeCountAgentV2(n_max=N_MAX, teach_eta=a.eta, tol=a.tol,
                                   max_extra_trials=a.max_extra_trials,
                                   trials_per_pass=a.trials_per_pass)
    calib_rng = np.random.default_rng(1)
    calib_bitmaps = [circle_bitmap(n, calib_rng, size=IMG_SIZE, radius=RADIUS)
                      for n in range(N_MAX + 1) for _ in range(N_CALIB_PER_COUNT)]
    agent.calibrate_baseline(calib_bitmaps)

    csv_path = os.path.join(RESULTS_DIR, "counting_results_tier2.csv")
    t0 = time.time()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "count", "predicted", "correct", "abs_err",
                    "output_drift_pct", "passes_used"])

        for rnd in range(N_ROUNDS):
            count, bitmap = sample_random(rng, n_min=0, n_max=N_MAX, size=IMG_SIZE, radius=RADIUS)
            pred, _ = agent.predict(bitmap)
            correct = int(pred == count)
            abs_err = abs(pred - count)
            passes = agent.train_on(bitmap, count)
            w.writerow([rnd, count, pred, correct, abs_err,
                        f"{agent.output_drift_pct():.4f}", passes])
            f.flush()

            if (rnd + 1) % SLEEP_EVERY == 0:
                agent.sleep()

            if (rnd + 1) % PROGRESS_EVERY == 0 or rnd == N_ROUNDS - 1:
                elapsed = time.time() - t0
                eta_s = elapsed / (rnd + 1) * (N_ROUNDS - rnd - 1)
                print(f"round {rnd + 1}/{N_ROUNDS} elapsed={elapsed:.0f}s eta={eta_s:.0f}s "
                      f"mean_passes(last20)={agent.mean_passes(20):.2f}")

    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
