"""python -m scripts.diagnostics.heuristic_ties"""
import random

import numpy as np

from core import paths
from core.runner import RunSpec
from opponents.opponents import heuristic_opponent
from agents.shared.tictactoe_play import play, train_policy, greedy_policy, opp_policy
from scripts.tictactoe.run_tictactoe import make_agent


def tie_random_policy(agent):
    def pol(board, legal):
        vals = []
        for a in legal:
            b = list(board)
            b[a] = 1
            vals.append(agent.value(tuple(b)))
        top = max(vals)
        return random.choice([a for a, v in zip(legal, vals) if v >= top - 1e-9])
    return pol


def evaluate(pol, n=200):
    out = {1: [], -1: []}
    for g in range(n):
        m = 1 if g % 2 == 0 else -1
        _, r = play({m: pol, -m: opp_policy(heuristic_opponent, -m)})
        out[m].append(r[m])
    return {("first" if m == 1 else "second"): (np.mean([x == 0 for x in v]), np.mean([x == -1 for x in v]))
            for m, v in out.items()}


for sym in (False, True):
    spec = RunSpec("t", "t", "qlearning", "x", 0, {"sym": sym, "lr": 0.05})
    agent = make_agent(spec)
    eps, n = 0.1, 500
    for ep in range(n):
        m = 1 if ep % 2 == 0 else -1
        after, res = play({m: train_policy(agent, eps), -m: opp_policy(heuristic_opponent, -m)})
        agent.update_episode(after[m], res[m])
        eps *= (0.01 / 0.1) ** (1 / n)
    print(f"sym={sym}  first-legal tie-break (draw, loss) by role:", evaluate(greedy_policy(agent)))
    print(f"sym={sym}  random tie-break      (draw, loss) by role:", evaluate(tie_random_policy(agent)))
