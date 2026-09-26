"""Main runner for the number-comparison ("solve the inequality")
experiment. mode='mb', plain step()/train() -- no vision, no x_zone --
directly reusing the bandit's validated mechanism instead of the
full-mode regression path that failed for counting.

Logs abs_diff = |a-b| alongside correctness, since (like the bandit's
close-probability arms) pairs with a small gap should be intrinsically
harder than pairs with a large one -- worth checking whether accuracy
degrades with closeness the way you'd expect from a real graded-magnitude
comparison, as opposed to being flat (which would suggest it's not
really using the magnitude at all).

    python scripts/comparison/run_comparison.py
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

from encoders.comparison.number_pair_encoder import NumberPairEncoder, sample_pair
from agents.comparison.connectome_compare_agent import ConnectomeCompareAgent
from agents.comparison.baseline_compare_agents import LinearCompareAgent, MLPCompareAgent

N_ROUNDS = 400
N_CALIB = 40
SLEEP_EVERY = 50
PROGRESS_EVERY = 20


def _find_dir_containing(name):
    """Walk up from this file to find an ancestor that has `name` as a
    child (file or dir) -- same robustness reasoning as the sys.path fix:
    don't hard-code an exact directory depth."""
    d = _HERE
    for _ in range(6):
        if os.path.exists(os.path.join(d, name)):
            return d
        d = os.path.dirname(d)
    raise FileNotFoundError(f"couldn't find '{name}' by walking up from {_HERE}")


LIVE_ALPN_PATH = os.path.join(_find_dir_containing("results"), "results", "gridworld", "live_alpn_indices.npy")
RESULTS_DIR = os.path.join(_find_dir_containing("results"), "results", "comparison")

if not os.path.exists(LIVE_ALPN_PATH):
    # fall back to computing it fresh, same helper run_gridworld_large_v2.py
    # uses (agents/shared/mb_value_cache.py's ensure_live_hops1) -- avoids a
    # second hard-coded path guess if your layout differs from the above.
    from agents.shared.mb_value_cache import ensure_live_hops1
    os.makedirs(os.path.dirname(LIVE_ALPN_PATH), exist_ok=True)
    ensure_live_hops1(LIVE_ALPN_PATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train", action="store_true",
                    help="skip connectome train_on() entirely -- diagnostic to check "
                         "whether the baseline-corrected sign is already near-perfect "
                         "from the physical encoding alone, with zero learning")
    ap.add_argument("--n-max", type=int, default=9,
                    help="numbers sampled from [1, n_max] -- sweep this (19/49/99) to "
                         "find where resolution starts to degrade accuracy, especially "
                         "for close pairs where the volume-knob gap shrinks")
    args = ap.parse_args()
    n_max = args.n_max

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    encoder = NumberPairEncoder(LIVE_ALPN_PATH, n_max=n_max)
    connectome = ConnectomeCompareAgent(encoder)

    calib_rng = np.random.default_rng(1)
    calib_pairs = [sample_pair(calib_rng, n_max=n_max) for _ in range(N_CALIB)]
    connectome.calibrate_baseline(calib_pairs)  # BEFORE any train_on call

    agents = {
        "connectome": connectome,
        "linear": LinearCompareAgent(),
        "mlp": MLPCompareAgent(),
    }

    tag = f"_nmax{n_max}" if n_max != 9 else ""
    fname = f"comparison_results{tag}_notrain.csv" if args.no_train else f"comparison_results{tag}.csv"
    csv_path = os.path.join(RESULTS_DIR, fname)
    t0 = time.time()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "agent", "a", "b", "abs_diff", "predicted", "correct", "drift_pct"])

        for rnd in range(N_ROUNDS):
            a, b = sample_pair(rng, n_max=n_max)
            correct_ans = "A" if a > b else "B"
            abs_diff = abs(a - b)

            for name, agent in agents.items():
                pred, _ = agent.predict(a, b)
                correct = int(pred == correct_ans)
                drift = f"{agent.drift_pct():.4f}" if hasattr(agent, "drift_pct") else ""
                w.writerow([rnd, name, a, b, abs_diff, pred, correct, drift])
                if name == "connectome" and args.no_train:
                    pass  # diagnostic mode: never train the connectome
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
