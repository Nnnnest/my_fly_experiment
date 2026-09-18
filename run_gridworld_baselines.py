import sys, os, csv
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from envs.gridworld_env import GridWorld
from agents.qlearning_gridworld_agent import QLearningGridAgent
from agents.mlp_gridworld_agent import MLPGridAgent

N_EPISODES = 300
SEEDS = [0, 1, 2]

env_probe = GridWorld()
n_states, n_actions = env_probe.n_states, 4

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "gridworld_baselines.csv")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["agent", "seed", "episode", "steps", "total_reward", "reached_goal"])

    for seed in SEEDS:
        for agent_name, agent in [
            ("qlearning", QLearningGridAgent(n_states, n_actions, seed=seed)),
            ("mlp", MLPGridAgent(n_states, n_actions, seed=seed)),
        ]:
            env = GridWorld(seed=seed)
            for ep in range(N_EPISODES):
                s = env.reset()
                total_r = 0.0
                for t in range(env.max_steps):
                    a = agent.select_action(s)
                    s2, r, done = env.step(a)
                    agent.update(s, a, r, s2, done)
                    s = s2
                    total_r += r
                    if done:
                        break
                writer.writerow([agent_name, seed, ep, t + 1, total_r, env.pos == env.goal])
            print(f"{agent_name} seed={seed} done")

print(f"Wrote {CSV_PATH}")
