"""Addition experiment runner (spec.txt rung (b) / summary_addendum2.txt
section 15): binary judgment 'is C == A + B?', mode='mb', plain
step()/train(). Single-digit only (n_max=9, no carrying) for this first
run -- widen n_max, then move to true two-digit positional operands with
carrying, only after this baseline result comes in.

Always run the --no-train control first (built into main() below, per
section 15's explicit instruction -- this is what caught Method A's
suspicious first result being real-but-modest bias, not perfect learning).

    python run_addition.py                      # n_max=9, 400 rounds
    python run_addition.py --n-max 19            # wider range
    python run_addition.py --no-train-only       # just the control
"""
import argparse
import csv
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR), os.path.dirname(os.path.dirname(_THIS_DIR))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from agents.addition.connectome_addition_agent import ConnectomeAdditionAgent
from agents.addition.baseline_addition_agents import LinearAdditionAgent, MLPAdditionAgent

RESULTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "..", "results", "addition")
N_TEST = 200
N_ROUNDS = 400
BLOCK = 20


def sample_triple(rng, n_max, positive):
    a = int(rng.integers(0, n_max + 1))
    b = int(rng.integers(0, n_max + 1))
    correct = a + b
    if positive:
        return a, b, correct, True
    c_max = 2 * n_max
    if rng.random() < 0.5:  # near-miss negative
        delta = int(rng.choice([-3, -2, -1, 1, 2, 3]))
        c = min(max(correct + delta, 0), c_max)
        if c == correct:
            c = correct + 1 if correct < c_max else correct - 1
    else:  # anywhere-else negative
        c = int(rng.integers(0, c_max + 1))
        if c == correct:
            c = correct + 1 if correct < c_max else correct - 1
    return a, b, c, False


def sample_batch(rng, n_max, n, pos_frac=0.5):
    return [sample_triple(rng, n_max, rng.random() < pos_frac) for _ in range(n)]


def accuracy(agent, triples):
    hits = sum(agent.predict(a, b, c) == label for a, b, c, label in triples)
    return hits / max(1, len(triples))


def run_no_train_control(n_max, seed=0):
    rng = np.random.default_rng(seed)
    calib = [(t[0], t[1], t[2]) for t in sample_batch(rng, n_max, 300)]
    test = sample_batch(rng, n_max, N_TEST)
    agent = ConnectomeAdditionAgent(n_max=n_max, seed=seed)
    agent.calibrate_baseline(calib)
    acc = accuracy(agent, test)
    print(f"[no-train control] n_max={n_max}: connectome accuracy = {acc:.3f} (chance = 0.500)")
    return acc


def run_training(n_max, n_rounds, seed=0, tag="single_digit"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    calib = [(t[0], t[1], t[2]) for t in sample_batch(rng, n_max, 300)]

    conn = ConnectomeAdditionAgent(n_max=n_max, seed=seed)
    conn.calibrate_baseline(calib)
    agents = {"connectome": conn, "linear": LinearAdditionAgent(seed=seed),
              "mlp": MLPAdditionAgent(seed=seed)}

    csv_path = os.path.join(RESULTS_DIR, f"addition_results_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "agent", "a", "b", "c", "label", "pred", "correct", "drift_pct"])
        block_hits = {name: [] for name in agents}
        for rnd in range(n_rounds):
            a, b, c, label = sample_triple(rng, n_max, rng.random() < 0.5)
            for name, agent in agents.items():
                pred = agent.predict(a, b, c)
                agent.update(a, b, c, label)
                drift = conn.drift_pct() if name == "connectome" else ""
                writer.writerow([rnd, name, a, b, c, label, pred, pred == label, drift])
                block_hits[name].append(pred == label)
            if (rnd + 1) % BLOCK == 0:
                f.flush()
                line = f"  round {rnd+1}/{n_rounds}: "
                for name in agents:
                    line += f"{name}={np.mean(block_hits[name][-BLOCK:]):.2f} "
                line += f"drift={conn.drift_pct():.2f}%"
                print(line)

    print(f"wrote {csv_path}")
    for name in agents:
        print(f"  {name}: overall={np.mean(block_hits[name]):.3f} "
              f"last{BLOCK}={np.mean(block_hits[name][-BLOCK:]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=9)
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-train-only", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    run_no_train_control(args.n_max, seed=args.seed)
    if args.no_train_only:
        return
    run_training(args.n_max, args.rounds, seed=args.seed, tag=args.tag or f"nmax{args.n_max}")


if __name__ == "__main__":
    main()
