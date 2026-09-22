# scripts/tictactoe/verify_v2_matches_v1.py
import random
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agents.tictactoe.connectome_tictactoe_agent import ConnectomeTicTacToeAgent
from agents.tictactoe.connectome_tictactoe_agent_v2 import ConnectomeTicTacToeAgentV2
from scripts.tictactoe.run_tictactoe import sample_early_boards, play_episode
from opponents.opponents import random_opponent

random.seed(0)
boards = sample_early_boards(300)

a1 = ConnectomeTicTacToeAgent(); a1.calibrate_baseline(boards)
a2 = ConnectomeTicTacToeAgentV2(gate="none"); a2.calibrate_baseline(boards)

for ep in range(50):
    random.seed(1000 + ep)
    afterstates, outcome = play_episode(a1, random_opponent, ep % 2 == 0, 0.1)
    a1.update_episode(afterstates, outcome)
    random.seed(1000 + ep)
    afterstates2, outcome2 = play_episode(a2, random_opponent, ep % 2 == 0, 0.1)
    a2.update_episode(afterstates2, outcome2)

diff = float(np.abs(a1.fly.wM - a2.fly.wM).max())
print(f"max weight diff after 50 episodes: {diff}")
assert diff < 1e-4, "v2 gate='none' does NOT reproduce v1 -- do not trust Tier 2 results until this passes"
print("OK: v2 (gate=none) reproduces v1 exactly")
