"""Two-digit (positional, carrying) addition runner. Same discipline as
run_addition.py: --no-train control first, then training.

Operands sampled 0..49 (sum stays 0..99, two digits, no third group
needed) but units digits are NOT restricted -- carrying (units sum >= 10)
happens naturally and often, since it's the one genuinely discrete part of
this task.

    python run_addition2.py
    python run_addition2.py --rounds 800
    python run_addition2.py --no-train-only
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

from agents.addition.connectome_addition2_agent import ConnectomeAddition2Agent
from agents.addition.baseline_addition2_agents import LinearAddition2Agent, MLPAddition2Agent

RESULTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "..", "results", "addition")
OP_MAX = 49  # a, b in 0..49 -> sum always 0..98, two digits, real carrying
N_TEST = 200
N_ROUNDS = 400
BLOCK = 20


def sample_triple(rng, positive):
    a = int(rng.integers(0, OP_MAX + 1))
    b = int(rng.integers(0, OP_MAX + 1))
    correct = a + b
    c_max = 2 * OP_MAX
    if positive:
        return a, b, correct, True
    r = rng.random()
    if r < 0.4:  # near-miss
        delta = int(rng.choice([-3, -2, -1, 1, 2, 3]))
        c = min(max(correct + delta, 0), c_max)
    elif r < 0.7:  # digit-swap distractor (wrong carry: swap tens/units)
        ct, cu = correct // 10, correct % 10
        c = cu * 10 + ct
    else:  # anywhere else
        c = int(rng.integers(0, c_max + 1))
    if c == correct:
        c = correct + 1 if correct < c_max else correct - 1
    return a, b, c, False


def sample_batch(rng, n, pos_frac=0.5):
    return [sample_triple(rng, rng.random() < pos_frac) for _ in range(n)]


def accuracy(agent, triples):
    hits = sum(agent.predict(a, b, c) == label for a, b, c, label in triples)
    return hits / max(1, len(triples))


def run_no_train_control(seed=0):
    rng = np.random.default_rng(seed)
    calib = [(t[0], t[1], t[2]) for t in sample_batch(rng, 300)]
    test = sample_batch(rng, N_TEST)
    agent = ConnectomeAddition2Agent(seed=seed)
    agent.calibrate_baseline(calib)
    acc = accuracy(agent, test)
    print(f"[no-train control] two-digit: connectome accuracy = {acc:.3f} (chance = 0.500)")
    return acc


def run_training(n_rounds, seed=0, tag="two_digit"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    calib = [(t[0], t[1], t[2]) for t in sample_batch(rng, 300)]

    conn = ConnectomeAddition2Agent(seed=seed)
    conn.calibrate_baseline(calib)
    agents = {"connectome": conn, "linear": LinearAddition2Agent(seed=seed),
              "mlp": MLPAddition2Agent(seed=seed)}

    csv_path = os.path.join(RESULTS_DIR, f"addition_results_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "agent", "a", "b", "c", "label", "pred", "correct", "drift_pct"])
        block_hits = {name: [] for name in agents}
        for rnd in range(n_rounds):
            a, b, c, label = sample_triple(rng, rng.random() < 0.5)
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
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-train-only", action="store_true")
    ap.add_argument("--tag", default="two_digit")
    args = ap.parse_args()

    run_no_train_control(seed=args.seed)
    if args.no_train_only:
        return
    run_training(args.rounds, seed=args.seed, tag=args.tag)


if __name__ == "__main__":
    main()
