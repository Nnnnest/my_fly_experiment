import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents"))
from envs.gridworld_env import GridWorld, ACTION_NAMES
from agents.connectome_gridworld_agent import ConnectomeGridAgent
from forced_sweep import forced_exploration_sweep
import numpy as np

env = GridWorld()
LIVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_alpn_indices.npy")

sweep_order = env.traversable_states()
agent = ConnectomeGridAgent(
    env.n_states, 4, LIVE_PATH,
    grid_width=env.width, goal_pos=env.goal,
    min_pulls=5, seed=0,
)
forced_exploration_sweep(env, agent, min_pulls=5, n_passes=1, seed=0)

print("state (pos)      | best action | value by action                | trained_order_rank")
for rank, pos in enumerate(sweep_order):
    s = env.state_id(pos)
    vals = agent._corrected_values(s)
    best = ACTION_NAMES[int(np.argmax(vals))]
    val_str = "  ".join(f"{ACTION_NAMES[a]}={vals[a]:7.2f}" for a in range(4))
    print(f"{str(pos):17s} | {best:11s} | {val_str} | {rank}")

print(f"\ngoal is at {env.goal}")
