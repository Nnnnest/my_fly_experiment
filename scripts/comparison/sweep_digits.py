"""Breaking-point sweep for N-digit (positional, separate-group) number
comparison: runs all three agents across an increasing digit count to
find where each one's accuracy starts to degrade, and who breaks FIRST.

Unlike sweep_nmax.py (Method A, single intensity knob), this is the
harder, positional-encoding version -- raw total current never gives the
answer away at any digit count, so any degradation here should be closer
to a genuine "ran out of ability to combine digits correctly" signal
rather than a resolution artifact. Logs group_size (live ALPN cells per
digit-position group) alongside results, since that shrinks as n_digits
grows and is a real confound -- see multi_digit_pair_encoder.py's
module docstring.

    python scripts/comparison/sweep_digits.py
    python scripts/comparison/sweep_digits.py --digit-values 1 2 3 4 5
    python scripts/comparison/sweep_digits.py --rounds 400 --digit-values 2 3 4
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

N_CALIB = 60
N_BASELINE_EVAL = 60
GAP1_MIN_N = 20


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


def run_one(n_digits, n_rounds, window, seed=0):
    rng = np.random.default_rng(seed)
    encoder = MultiDigitPairEncoder(LIVE_ALPN_PATH, n_digits=n_digits, seed=seed)
    connectome = ConnectomeCompareAgent(encoder)

    calib_rng = np.random.default_rng(seed + 1)
    calib_pairs = [sample_n_digit_pair(calib_rng, n_digits) for _ in range(N_CALIB)]
    connectome.calibrate_baseline(calib_pairs)

    agents = {
        "connectome": connectome,
        "linear": LinearCompareAgent(seed=seed),
        "mlp": MLPCompareAgent(seed=seed),
    }

    baseline_rng = np.random.default_rng(seed + 2)
    baseline_pairs = [sample_n_digit_pair(baseline_rng, n_digits) for _ in range(N_BASELINE_EVAL)]
    baseline_correct = {name: 0 for name in agents}
    for a, b in baseline_pairs:
        correct_ans = "A" if a > b else "B"
        for name, agent in agents.items():
            pred, _ = agent.predict(a, b)
            baseline_correct[name] += int(pred == correct_ans)
    baseline_acc = {name: baseline_correct[name] / N_BASELINE_EVAL for name in agents}

    history = {name: [] for name in agents}
    for rnd in range(n_rounds):
        a, b = sample_n_digit_pair(rng, n_digits)
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
    return results, encoder.group_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digit-values", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "sweep_digits_results.csv")
    t0 = time.time()

    rows = []
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_digits", "group_size", "agent", "baseline_accuracy", "overall_accuracy",
                    "last_accuracy", "gap1_accuracy", "gap1_n"])

        for n_digits in args.digit_values:
            t1 = time.time()
            try:
                results, group_size = run_one(n_digits, args.rounds, args.window, seed=args.seed)
            except ValueError as e:
                print(f"n_digits={n_digits}: SKIPPED ({e})")
                continue
            for name, r in results.items():
                w.writerow([n_digits, group_size, name, f"{r['baseline_accuracy']:.4f}",
                            f"{r['overall_accuracy']:.4f}", f"{r['last_accuracy']:.4f}",
                            f"{r['gap1_accuracy']:.4f}" if not np.isnan(r['gap1_accuracy']) else "",
                            r['gap1_n']])
                rows.append((n_digits, name, r))
            f.flush()
            print(f"n_digits={n_digits} (group_size={group_size}) done in {time.time() - t1:.0f}s -- "
                  + "  ".join(f"{name}: baseline={r['baseline_accuracy']:.2f} "
                              f"last={r['last_accuracy']:.2f} "
                              f"gap1={r['gap1_accuracy']:.2f}(n={r['gap1_n']})"
                              for name, r in results.items()))

    print(f"\ntotal elapsed {time.time() - t0:.0f}s. wrote {csv_path}")

    print("\n=== first n_digits where last_accuracy < 0.80 (None = never, in this sweep) ===")
    by_agent = {}
    for n_digits, name, r in rows:
        by_agent.setdefault(name, []).append((n_digits, r["last_accuracy"]))
    for name, pts in by_agent.items():
        pts.sort()
        broke_at = next((n for n, acc in pts if acc < 0.80), None)
        print(f"  {name}: {broke_at}")


if __name__ == "__main__":
    main()
