"""Tier 2 -- 'assisted': same task as Tier 1, but the connectome trains
through GatedAdditionAgent (skips train() once its own judgment already
agrees with the label beyond a calibrated margin) -- same idea as
tic-tac-toe/grid-world's Tier 2 gate. linear/mlp are UNCHANGED (no gate
built for them, same reasoning as tic-tac-toe's write-up: their own
update rules already have a built-in decay/error-proportional step).

    python run_addition_tier2.py                  # single-digit
    python run_addition_tier2.py --two-digit       # two-digit
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

from addition_common import sample_batch, sample_triple
from agents.addition.connectome_addition_agent import ConnectomeAdditionAgent
from agents.addition.connectome_addition2_agent import ConnectomeAddition2Agent
from agents.addition.addition_gate_utils import GatedAdditionAgent
from agents.addition.baseline_addition_agents import LinearAdditionAgent, MLPAdditionAgent
from agents.addition.baseline_addition2_agents import LinearAddition2Agent, MLPAddition2Agent

RESULTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "..", "results", "addition")
N_ROUNDS = 800
BLOCK = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-digit", action="store_true")
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin-frac", type=float, default=0.05)
    args = ap.parse_args()

    two_digit = args.two_digit
    op_max = 49 if two_digit else 9
    tag = "tier2_two_digit" if two_digit else "tier2_nmax9"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    calib = [(t[0], t[1], t[2]) for t in sample_batch(rng, op_max, 300, two_digit)]
    probe = [(t[0], t[1], t[2]) for t in sample_batch(rng, op_max, 8, two_digit)]

    if two_digit:
        inner = ConnectomeAddition2Agent(seed=args.seed)
        lin, mlp = LinearAddition2Agent(seed=args.seed), MLPAddition2Agent(seed=args.seed)
    else:
        inner = ConnectomeAdditionAgent(n_max=op_max, seed=args.seed)
        lin, mlp = LinearAdditionAgent(seed=args.seed), MLPAdditionAgent(seed=args.seed)

    inner.calibrate_baseline(calib)
    conn = GatedAdditionAgent(inner, margin_frac=args.margin_frac)
    conn.calibrate_gate(probe)
    agents = {"connectome": conn, "linear": lin, "mlp": mlp}

    csv_path = os.path.join(RESULTS_DIR, f"addition_results_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "agent", "a", "b", "c", "label", "pred", "correct",
                         "drift_pct", "n_trained", "n_skipped"])
        block_hits = {name: [] for name in agents}
        for rnd in range(args.rounds):
            a, b, c, label = sample_triple(rng, op_max, rng.random() < 0.5, two_digit)
            for name, agent in agents.items():
                pred = agent.predict(a, b, c)
                agent.update(a, b, c, label)
                if name == "connectome":
                    drift, nt, ns = conn.drift_pct(), conn.n_trained, conn.n_skipped
                else:
                    drift = nt = ns = ""
                writer.writerow([rnd, name, a, b, c, label, pred, pred == label, drift, nt, ns])
                block_hits[name].append(pred == label)
            if (rnd + 1) % BLOCK == 0:
                f.flush()
                line = f"  round {rnd+1}/{args.rounds}: "
                for name in agents:
                    line += f"{name}={np.mean(block_hits[name][-BLOCK:]):.2f} "
                line += f"drift={conn.drift_pct():.2f}% trained={conn.n_trained} skipped={conn.n_skipped}"
                print(line)

    print(f"wrote {csv_path}")
    for name in agents:
        print(f"  {name}: overall={np.mean(block_hits[name]):.3f} "
              f"last{BLOCK}={np.mean(block_hits[name][-BLOCK:]):.3f}")


if __name__ == "__main__":
    main()
