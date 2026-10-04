"""Breaking-point sweep for number comparison (Method A, single-knob
encoding): runs all three agents across an increasing range of n_max
values to find where each one's accuracy starts to degrade, and
specifically who breaks FIRST.

Tracks, per n_max and per agent:
  - baseline_accuracy: pre-training (round=-1 style), zero train_on calls
  - overall_accuracy / last_W_accuracy: same as run_comparison.py
  - gap1_accuracy: accuracy on the hardest case specifically (adjacent
    numbers, |a-b|=1) -- expected to be the first thing to crack as the
    "volume knob" gets more crowded, even while overall accuracy still
    looks fine (large-gap pairs stay easy regardless of n_max)

Shorter rounds-per-n_max than run_comparison.py's 400 (default 200) to
keep total sweep runtime reasonable across several n_max values --
override with --rounds if you want the full 400 at each point.

    python scripts/comparison/sweep_nmax.py
    python scripts/comparison/sweep_nmax.py --n-max-values 9 19 49 99 199 499 999
    python scripts/comparison/sweep_nmax.py --rounds 400 --n-max-values 99 299 999
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

N_CALIB = 40
N_BASELINE_EVAL = 60
GAP1_MIN_N = 20  # minimum number of gap=1 samples to trust that accuracy number


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


def run_one(n_max, n_rounds, window, seed=0):
    rng = np.random.default_rng(seed)
    encoder = NumberPairEncoder(LIVE_ALPN_PATH, n_max=n_max, seed=seed)
    connectome = ConnectomeCompareAgent(encoder)

    calib_rng = np.random.default_rng(seed + 1)
    calib_pairs = [sample_pair(calib_rng, n_max=n_max) for _ in range(N_CALIB)]
    connectome.calibrate_baseline(calib_pairs)

    agents = {
        "connectome": connectome,
        "linear": LinearCompareAgent(seed=seed),
        "mlp": MLPCompareAgent(seed=seed),
    }

    # pre-training baseline, zero train_on calls
    baseline_rng = np.random.default_rng(seed + 2)
    baseline_pairs = [sample_pair(baseline_rng, n_max=n_max) for _ in range(N_BASELINE_EVAL)]
    baseline_correct = {name: 0 for name in agents}
    for a, b in baseline_pairs:
        correct_ans = "A" if a > b else "B"
        for name, agent in agents.items():
            pred, _ = agent.predict(a, b)
            baseline_correct[name] += int(pred == correct_ans)
    baseline_acc = {name: baseline_correct[name] / N_BASELINE_EVAL for name in agents}

    # main training loop
    history = {name: [] for name in agents}  # (round, correct, abs_diff)
    for rnd in range(n_rounds):
        a, b = sample_pair(rng, n_max=n_max)
        correct_ans = "A" if a > b else "B"
        abs_diff = abs(a - b)
        for name, agent in agents.items():
            pred, _ = agent.predict(a, b)
            correct = int(pred == correct_ans)
            history[name].append((rnd, correct, abs_diff))
            agent.train_on(a, b)

    results = {}
    for name in agents:
        rows = history[name]
        overall = np.mean([c for _, c, _ in rows])
        last = [c for r, c, _ in rows if r >= n_rounds - window]
        last_acc = np.mean(last) if last else float("nan")
        gap1 = [c for _, c, d in rows if d == 1]
        gap1_acc = np.mean(gap1) if len(gap1) >= GAP1_MIN_N else float("nan")
        results[name] = dict(baseline_accuracy=baseline_acc[name], overall_accuracy=overall,
                             last_accuracy=last_acc, gap1_accuracy=gap1_acc,
                             gap1_n=len(gap1))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max-values", type=int, nargs="+",
                    default=[9, 19, 49, 99, 199, 499, 999])
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "sweep_nmax_results.csv")
    t0 = time.time()

    rows = []
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_max", "agent", "baseline_accuracy", "overall_accuracy",
                    "last_accuracy", "gap1_accuracy", "gap1_n"])

        for n_max in args.n_max_values:
            t1 = time.time()
            results = run_one(n_max, args.rounds, args.window, seed=args.seed)
            for name, r in results.items():
                w.writerow([n_max, name, f"{r['baseline_accuracy']:.4f}",
                            f"{r['overall_accuracy']:.4f}", f"{r['last_accuracy']:.4f}",
                            f"{r['gap1_accuracy']:.4f}" if not np.isnan(r['gap1_accuracy']) else "",
                            r['gap1_n']])
                rows.append((n_max, name, r))
            f.flush()
            print(f"n_max={n_max} done in {time.time() - t1:.0f}s -- "
                  + "  ".join(f"{name}: last={r['last_accuracy']:.2f} "
                              f"gap1={r['gap1_accuracy']:.2f}(n={r['gap1_n']})"
                              for name, r in results.items()))

    print(f"\ntotal elapsed {time.time() - t0:.0f}s. wrote {csv_path}")

    # quick summary: for each agent, first n_max where last_accuracy drops
    # below 0.8 (an arbitrary but reasonable "starting to break" threshold)
    print("\n=== first n_max where last_accuracy < 0.80 (None = never, in this sweep) ===")
    by_agent = {}
    for n_max, name, r in rows:
        by_agent.setdefault(name, []).append((n_max, r["last_accuracy"]))
    for name, pts in by_agent.items():
        pts.sort()
        broke_at = next((n for n, acc in pts if acc < 0.80), None)
        print(f"  {name}: {broke_at}")


if __name__ == "__main__":
    main()
