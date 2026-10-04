"""Runner for N-digit comparison (general version of
run_comparison_twodigit.py, via MultiDigitPairEncoder). Same pattern as
run_comparison.py's baseline fix: logs a round=-1 pre-training baseline
(zero train_on calls, all three agents) before the main loop, so you get
the genuine before-any-learning number instead of only the 0-19 block
average.

    python scripts/comparison/run_comparison_multidigit.py --n-digits 2
    python scripts/comparison/run_comparison_multidigit.py --n-digits 3
    python scripts/comparison/run_comparison_multidigit.py --n-digits 3 --no-train
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

from encoders.comparison.multi_digit_pair_encoder import MultiDigitPairEncoder, sample_n_digit_pair
from agents.comparison.connectome_compare_agent import ConnectomeCompareAgent
from agents.comparison.baseline_compare_agents import LinearCompareAgent, MLPCompareAgent

N_ROUNDS = 400
N_CALIB = 60
N_BASELINE_EVAL = 60
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
    ap.add_argument("--n-digits", type=int, default=2)
    ap.add_argument("--no-train", action="store_true",
                    help="skip connectome train_on() -- same diagnostic as the "
                         "single/two-digit versions")
    args = ap.parse_args()
    n_digits = args.n_digits

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    encoder = MultiDigitPairEncoder(LIVE_ALPN_PATH, n_digits=n_digits)
    print(f"n_digits={n_digits}: {2 * n_digits} ALPN groups, "
          f"{encoder.group_size} cells/group (of {len(np.load(LIVE_ALPN_PATH))} live total)")
    connectome = ConnectomeCompareAgent(encoder)

    calib_rng = np.random.default_rng(1)
    calib_pairs = [sample_n_digit_pair(calib_rng, n_digits) for _ in range(N_CALIB)]
    connectome.calibrate_baseline(calib_pairs)  # BEFORE any train_on call

    agents = {
        "connectome": connectome,
        "linear": LinearCompareAgent(),
        "mlp": MLPCompareAgent(),
    }

    fname = f"comparison_results_{n_digits}digit{'_notrain' if args.no_train else ''}.csv"
    csv_path = os.path.join(RESULTS_DIR, fname)
    t0 = time.time()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "agent", "a", "b", "abs_diff", "predicted", "correct", "drift_pct"])

        # pre-training baseline, zero train_on calls, all agents
        baseline_rng = np.random.default_rng(2)
        baseline_pairs = [sample_n_digit_pair(baseline_rng, n_digits) for _ in range(N_BASELINE_EVAL)]
        baseline_correct = {name: 0 for name in agents}
        for a, b in baseline_pairs:
            correct_ans = "A" if a > b else "B"
            abs_diff = abs(a - b)
            for name, agent in agents.items():
                pred, _ = agent.predict(a, b)
                correct = int(pred == correct_ans)
                baseline_correct[name] += correct
                drift = f"{agent.drift_pct():.4f}" if hasattr(agent, "drift_pct") else ""
                w.writerow([-1, name, a, b, abs_diff, pred, correct, drift])
        print(f"pre-training baseline (n={N_BASELINE_EVAL} pairs, zero train_on calls):")
        for name in agents:
            print(f"  {name}: accuracy={baseline_correct[name] / N_BASELINE_EVAL:.3f}")
        f.flush()

        for rnd in range(N_ROUNDS):
            a, b = sample_n_digit_pair(rng, n_digits)
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
