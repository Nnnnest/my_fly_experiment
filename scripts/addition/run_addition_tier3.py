"""Tier 3 -- 'imitation': the connectome is shown a FIXED pool of
(a,b,c,label) triples repeatedly (dense supervised training on known-
correct labels), rather than one online random example per round -- same
paradigm shift as train_imitation_tictactoe.py. Evaluated on a pool never
trained on (held-out generalization) AND on the training pool itself
(memorization) -- unlike tic-tac-toe, the triple space here isn't small
enough to enumerate exhaustively, so 'full coverage' isn't available;
held-out accuracy is the closer analogue.

    python run_addition_tier3.py                  # single-digit
    python run_addition_tier3.py --two-digit
    python run_addition_tier3.py --sleep-every 50  # periodic fly.sleep()
"""
import argparse
import os
import random
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR), os.path.dirname(os.path.dirname(_THIS_DIR))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from addition_common import sample_batch, accuracy
from agents.addition.connectome_addition_agent import ConnectomeAdditionAgent
from agents.addition.connectome_addition2_agent import ConnectomeAddition2Agent

POOL_SIZE = 200
HELDOUT_SIZE = 200
REPS_PER_TRIPLE = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-digit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep-every", type=int, default=None)
    ap.add_argument("--reps", type=int, default=REPS_PER_TRIPLE)
    args = ap.parse_args()

    two_digit = args.two_digit
    op_max = 49 if two_digit else 9
    rng = np.random.default_rng(args.seed)

    calib = [(t[0], t[1], t[2]) for t in sample_batch(rng, op_max, 300, two_digit)]
    train_pool = sample_batch(rng, op_max, POOL_SIZE, two_digit)
    heldout_pool = sample_batch(rng, op_max, HELDOUT_SIZE, two_digit)

    agent = (ConnectomeAddition2Agent(seed=args.seed) if two_digit
             else ConnectomeAdditionAgent(n_max=op_max, seed=args.seed))
    agent.calibrate_baseline(calib)

    print(f"[before] train-pool acc={accuracy(agent, train_pool):.3f} "
          f"held-out acc={accuracy(agent, heldout_pool):.3f}")

    order = list(range(len(train_pool)))
    step = 0
    for rep in range(args.reps):
        random.Random(args.seed + rep).shuffle(order)
        for i in order:
            a, b, c, label = train_pool[i]
            agent.update(a, b, c, label)
            step += 1
            if args.sleep_every and step % args.sleep_every == 0:
                agent.sleep()

    print(f"[after {args.reps} reps, sleep_every={args.sleep_every}]")
    print(f"  train-pool acc={accuracy(agent, train_pool):.3f} (memorization)  "
          f"held-out acc={accuracy(agent, heldout_pool):.3f} (generalization)")
    print(f"  drift={agent.drift_pct():.2f}%")


if __name__ == "__main__":
    main()
