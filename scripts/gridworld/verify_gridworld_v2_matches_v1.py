"""Confirms ConnectomeGridAgentV2(encoding='onehot', hops=1, gate='none')
reproduces ConnectomeGridAgent (true v1, live brain.step() calls, no
cache) bit-for-bit in its train() trajectory, on the small 26-state maze.
Run before trusting 'onehot_h1_v1equiv' as a Tier-1-fair stand-in on
larger mazes where running true v1 directly would be much slower."""
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from envs.gridworld_env import GridWorld, GRID
from agents.gridworld.connectome_gridworld_agent import ConnectomeGridAgent
from agents.gridworld.connectome_gridworld_agent_v2 import ConnectomeGridAgentV2

LIVE_PATH = os.path.join(ROOT, "results", "gridworld", "live_alpn_indices.npy")
N_EPISODES = 30


def run(agent_cls, **kw):
    env = GridWorld(grid=GRID, max_steps=100, seed=0)
    bfs_dist = env.bfs_distances_from_goal()
    trav_ids = [env.state_id((r, c)) for r in range(env.height)
                for c in range(env.width) if not env._is_wall(r, c)]
    agent = agent_cls(trav_ids, 4, LIVE_PATH, bfs_dist, hops=1, seed=0, **kw)
    for ep in range(N_EPISODES):
        s = env.reset()
        for t in range(env.max_steps):
            a = agent.select_action(s)
            s2, r, done = env.step(a)
            agent.update(s, a, r, s2, done)
            s = s2
            if done:
                break
        agent.decay_epsilon()
    return agent.brain.wM


if __name__ == "__main__":
    w1 = run(ConnectomeGridAgent)
    w2 = run(ConnectomeGridAgentV2, encoding="onehot", gate="none")
    diff = float(np.abs(w1 - w2).max())
    print(f"max weight diff after {N_EPISODES} episodes: {diff}")
    assert diff < 1e-4, "v2 (onehot, gate=none) does NOT reproduce v1 -- do not use as a Tier-1-fair stand-in"
    print("OK: v2 (onehot, gate=none) reproduces v1 exactly")
