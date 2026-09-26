"""Single runner for all three addition tiers (basic / gated / imitation),
both digit widths, multiple seeds -- reduces the noise seen in earlier
single-seed, train-set-is-eval-set runs two ways:
  1. accuracy is measured on a FIXED held-out eval set (not the online
     training examples), evaluated periodically -- decouples the training
     signal from the reported metric.
  2. multiple seeds are run and aggregated (mean +/- std), so agent
     differences can be judged against seed-to-seed spread, not read off
     one trajectory.

    python run_addition_experiment.py
    python run_addition_experiment.py --tiers tier1 tier3 --digit-widths two
    python run_addition_experiment.py --seeds 0 1 2 3 4 --rounds 1200
"""
import argparse
import csv
import os
import random
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR), os.path.dirname(os.path.dirname(_THIS_DIR))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from addition_common import sample_batch, sample_triple, accuracy
from agents.addition.connectome_addition_agent import ConnectomeAdditionAgent
from agents.addition.connectome_addition2_agent import ConnectomeAddition2Agent
from agents.addition.addition_gate_utils import GatedAdditionAgent, calibrate_own_effect
from agents.addition.baseline_addition_agents import LinearAdditionAgent, MLPAdditionAgent
from agents.addition.baseline_addition2_agents import LinearAddition2Agent, MLPAddition2Agent

RESULTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "..", "results", "addition")


def make_agents(two_digit, seed):
    if two_digit:
        return (ConnectomeAddition2Agent(seed=seed), LinearAddition2Agent(seed=seed),
                MLPAddition2Agent(seed=seed), 49)
    return (ConnectomeAdditionAgent(n_max=9, seed=seed), LinearAdditionAgent(seed=seed),
            MLPAdditionAgent(seed=seed), 9)


def run_tier1(seed, two_digit, op_max, calib, eval_set, rounds, eval_every, writer):
    conn, lin, mlp, _ = make_agents(two_digit, seed)
    conn.calibrate_baseline(calib)
    agents = {"connectome": conn, "linear": lin, "mlp": mlp}
    rng = np.random.default_rng(seed + 1000)
    for rnd in range(1, rounds + 1):
        a, b, c, label = sample_triple(rng, op_max, rng.random() < 0.5, two_digit)
        for agent in agents.values():
            agent.update(a, b, c, label)
        if rnd % eval_every == 0 or rnd == rounds:
            for name, agent in agents.items():
                acc = accuracy(agent, eval_set)
                drift = agent.drift_pct() if name == "connectome" else ""
                writer.writerow([seed, "tier1", "two" if two_digit else "single", name, rnd, acc, drift])
    return agents


def run_tier2(seed, two_digit, op_max, calib, eval_set, rounds, eval_every, margin_frac, writer):
    conn, lin, mlp, _ = make_agents(two_digit, seed)
    conn.calibrate_baseline(calib)
    rng = np.random.default_rng(seed + 3000)
    probe = [(t[0], t[1], t[2]) for t in sample_batch(rng, op_max, 8, two_digit)]
    gated = GatedAdditionAgent(conn, margin_frac=margin_frac)
    gated.calibrate_gate(probe)
    agents = {"connectome": gated, "linear": lin, "mlp": mlp}
    rng = np.random.default_rng(seed + 1000)
    for rnd in range(1, rounds + 1):
        a, b, c, label = sample_triple(rng, op_max, rng.random() < 0.5, two_digit)
        for agent in agents.values():
            agent.update(a, b, c, label)
        if rnd % eval_every == 0 or rnd == rounds:
            for name, agent in agents.items():
                acc = accuracy(agent, eval_set)
                drift = agent.drift_pct() if name == "connectome" else ""
                writer.writerow([seed, "tier2", "two" if two_digit else "single", name, rnd, acc, drift])
    return agents


def run_tier3(seed, two_digit, op_max, calib, eval_set, reps, pool_size, eval_every,
              sleep_every, writer):
    conn, lin, mlp, _ = make_agents(two_digit, seed)
    conn.calibrate_baseline(calib)
    agents = {"connectome": conn, "linear": lin, "mlp": mlp}
    rng = np.random.default_rng(seed + 2000)
    train_pool = sample_batch(rng, op_max, pool_size, two_digit)
    step = 0
    for rep in range(reps):
        order = list(range(len(train_pool)))
        random.Random(seed + rep).shuffle(order)
        for i in order:
            a, b, c, label = train_pool[i]
            for agent in agents.values():
                agent.update(a, b, c, label)
            step += 1
            if sleep_every and step % sleep_every == 0:
                conn.sleep()
            if step % eval_every == 0:
                for name, agent in agents.items():
                    acc = accuracy(agent, eval_set)
                    drift = agent.drift_pct() if name == "connectome" else ""
                    writer.writerow([seed, "tier3", "two" if two_digit else "single", name, step, acc, drift])
    return agents


TIER_FNS = {"tier1": run_tier1, "tier2": run_tier2, "tier3": run_tier3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["tier1", "tier2", "tier3"],
                    choices=["tier1", "tier2", "tier3"])
    ap.add_argument("--digit-widths", nargs="+", default=["single", "two"],
                    choices=["single", "two"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=800)       # tier1/tier2
    ap.add_argument("--reps", type=int, default=3)             # tier3
    ap.add_argument("--pool-size", type=int, default=200)      # tier3
    ap.add_argument("--sleep-every", type=int, default=None)   # tier3
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--eval-size", type=int, default=300)
    ap.add_argument("--margin-frac", type=float, default=0.05)  # tier2
    ap.add_argument("--tag", default="all_tiers")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"addition_results_{args.tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "tier", "digit_width", "agent", "round", "eval_acc", "drift_pct"])

        for dw in args.digit_widths:
            two_digit = dw == "two"
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                op_max = 49 if two_digit else 9
                calib = [(t[0], t[1], t[2]) for t in sample_batch(rng, op_max, 300, two_digit)]
                eval_set = sample_batch(rng, op_max, args.eval_size, two_digit)

                for tier in args.tiers:
                    print(f"=== {dw}-digit / {tier} / seed={seed} ===")
                    if tier == "tier1":
                        run_tier1(seed, two_digit, op_max, calib, eval_set, args.rounds,
                                 args.eval_every, writer)
                    elif tier == "tier2":
                        run_tier2(seed, two_digit, op_max, calib, eval_set, args.rounds,
                                 args.eval_every, args.margin_frac, writer)
                    else:
                        run_tier3(seed, two_digit, op_max, calib, eval_set, args.reps,
                                 args.pool_size, args.eval_every, args.sleep_every, writer)
                    f.flush()

    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
