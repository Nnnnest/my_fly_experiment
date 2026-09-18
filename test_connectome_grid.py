import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents"))
from envs.gridworld_env import GridWorld
from agents.connectome_gridworld_agent import ConnectomeGridAgent
import time

from forced_sweep import forced_exploration_sweep

LIVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_alpn_indices.npy")

print("constructing agent (this does n_states * n_actions baseline probes)...")
t0 = time.time()

env = GridWorld()
agent = ConnectomeGridAgent(
    env.n_states, 4, LIVE_PATH,
    grid_width=env.width, goal_pos=env.goal,
    min_pulls=5, seed=0,
)
forced_exploration_sweep(env, agent, min_pulls=5, n_passes=1, seed=0)
# now agent.counts[s, a] >= min_pulls for every traversable state, every action —
# select_action's under_explored branch will never trigger again, so the
# episodic loop below is pure exploit/epsilon-explore from a fully-informed start

print(f"done in {time.time()-t0:.1f}s")

N_PILOT_EPISODES = 30

print(f"\nrunning {N_PILOT_EPISODES} episodes, logging every step...")

results = []

for ep in range(N_PILOT_EPISODES):
    s = env.reset()
    total_r = 0
    t0 = time.time()
    for t in range(env.max_steps):
        a = agent.select_action(s)
        s2, r, done = env.step(a)
        agent.update(s, a, r, s2, done)
        total_r += r
        s = s2
        if done:
            break
    explored_pairs = (agent.counts >= agent.min_pulls).sum()
    total_pairs = agent.n_states * agent.n_actions

    results.append({"episode": ep, "steps": t + 1, "reward": total_r, "reached_goal": env.pos == env.goal}) 

import pandas as pd
df = pd.DataFrame(results)
df["rolling_success"] = df["reached_goal"].rolling(10, min_periods=1).mean()
df["rolling_steps"] = df["steps"].rolling(10, min_periods=1).mean()
print(df[["episode", "steps", "reached_goal", "rolling_success", "rolling_steps"]].to_string(index=False))
