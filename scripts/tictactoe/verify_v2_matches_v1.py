"""python -m scripts.tictactoe.verify_v2_matches_v1"""
import random

import numpy as np

from core import paths
from agents.tictactoe.connectome_tictactoe_agent import ConnectomeTicTacToeAgent
from agents.tictactoe.connectome_tictactoe_agent_v2 import ConnectomeTicTacToeAgentV2
from agents.shared.tictactoe_play import play, train_policy, opp_policy
from opponents.opponents import random_opponent
from scripts.tictactoe.run_tictactoe import early_boards


def run(agent, boards, n=50):
    agent.calibrate_baseline(boards)
    for ep in range(n):
        random.seed(1000 + ep)
        m = 1 if ep % 2 == 0 else -1
        after, res = play({m: train_policy(agent, 0.1), -m: opp_policy(random_opponent, -m)})
        agent.update_episode(after[m], res[m])
    return agent.fly.wM.copy()


boards = early_boards(0)
w1 = run(ConnectomeTicTacToeAgent(), boards)
w2 = run(ConnectomeTicTacToeAgentV2(gate="none"), boards)
w3 = run(ConnectomeTicTacToeAgentV2(gate="none"), boards)      # same process, pool reset
d12, d23 = float(np.abs(w1 - w2).max()), float(np.abs(w2 - w3).max())
print(f"v1 vs v2: {d12}   v2 vs v2 after reset: {d23}")
assert d12 < 1e-4 and d23 == 0.0
print("OK")
