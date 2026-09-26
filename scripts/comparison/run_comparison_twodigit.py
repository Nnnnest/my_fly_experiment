"""Runner for Method B: true two-digit comparison (separate tens/units
ALPN groups, see two_digit_pair_encoder.py's module docstring for why
this is a structurally harder task than Method A's single-knob encoding,
not just a wider number range).

Reuses ConnectomeCompareAgent/LinearCompareAgent/MLPCompareAgent
unchanged -- only the encoder differs, since TwoDigitPairEncoder shares
the same .encode(a, b) interface as NumberPairEncoder.

    python scripts/comparison/run_comparison_twodigit.py
    python scripts/comparison/run_comparison_twodigit.py --no-train
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

from encoders.comparison.two_digit_pair_encoder import TwoDigitPairEncoder, sample_two_digit_pair
from agents.comparison.connectome_compare_agent import ConnectomeCompareAgent
from agents.comparison.baseline_compare_agents import LinearCompareAgent, MLPCompareAgent

N_ROUNDS = 400
N_MIN, N_MAX = 10, 99
N_CALIB = 60  # more than Method A's 40 -- wider (a,b,a*b) space to fit a baseline over
SLEEP_EVERY = 50
PROGRESS_EVERY = 20


def _find_dir_containing(name):
    d = _HERE
    for _ in range(6):
        if os.path.exists(os.path.join(d, name)):
            return d
        d = os.path.dirname(d)
    raise FileNotFoundError(f"couldn't find '{name}' by walking up from {_HERE}")


LIVE_ALPN_PATH = os.path.join(_find_dir_containing("results"), "results", "gridworld", "live_alpn_indices.npy")
RESULTS_DIR = os.path.join(_find_dir_containing("results"), "results", "comparison")

if not os.path.exists(LIVE_ALPN_PATH):
    from agents.shared.mb_value_cache import ensure_live_hops1
    os.makedirs(os.path.dirname(LIVE_ALPN_PATH), exist_ok=True)
    ensure_live_hops1(LIVE_ALPN_PATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train", action="store_true",
                    help="skip connectome train_on() -- check whether tens/units "
                         "encoding alone (no learned weighting) already solves it, "
                         "same diagnostic as Method A's --no-train")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    encoder = TwoDigitPairEncoder(LIVE_ALPN_PATH)
    connectome = ConnectomeCompareAgent(encoder)

    calib_rng = np.random.default_rng(1)
    calib_pairs = [sample_two_digit_pair(calib_rng, N_MIN, N_MAX) for _ in range(N_CALIB)]
    connectome.calibrate_baseline(calib_pairs)  # BEFORE any train_on call

    agents = {
        "connectome": connectome,
        "linear": LinearCompareAgent(),
        "mlp": MLPCompareAgent(),
    }

    fname = "comparison_results_twodigit_notrain.csv" if args.no_train else "comparison_results_twodigit.csv"
    csv_path = os.path.join(RESULTS_DIR, fname)
    t0 = time.time()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "agent", "a", "b", "abs_diff", "predicted", "correct", "drift_pct"])

        for rnd in range(N_ROUNDS):
            a, b = sample_two_digit_pair(rng, N_MIN, N_MAX)
            correct_ans = "A" if a > b else "B"
            abs_diff = abs(a - b)

            for name, agent in agents.items():
                pred, _ = agent.predict(a, b)
                correct = int(pred == correct_ans)
                drift = f"{agent.drift_pct():.4f}" if hasattr(agent, "drift_pct") else ""
                w.writerow([rnd, name, a, b, abs_diff, pred, correct, drift])
                if name == "connectome" and args.no_train:
                    pass
                else:
                    agent.train_on(a, b)
            f.flush()

            if hasattr(agents["connectome"], "sleep") and (rnd + 1) % SLEEP_EVERY == 0:
                agents["connectome"].sleep()

            if (rnd + 1) % PROGRESS_EVERY == 0 or rnd == N_ROUNDS - 1:
                elapsed = time.time() - t0
                eta = elapsed / (rnd + 1) * (N_ROUNDS - rnd - 1)
                print(f"round {rnd + 1}/{N_ROUNDS} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
