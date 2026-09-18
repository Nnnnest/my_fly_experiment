import sys, os, csv, time
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "agents"))
from envs.gridworld_env import GridWorld
from envs.maze_generator import generate_maze
from agents.qlearning_gridworld_agent import QLearningGridAgent
from agents.mlp_gridworld_agent import MLPGridAgent
from agents.connectome_gridworld_agent import ConnectomeGridAgent

N_EPISODES = 400
SEEDS = [0, 1, 2]
MIN_PULLS = 5
GAMMA = 0.95
HOPS = 2
PROGRESS_EVERY = 10
LIVE_PATH_HOPS2 = os.path.join(SCRIPT_DIR, "live_alpn_indices_hops2.npy")

LARGE_MAZE = generate_maze(19, 13, extra_connections=0.1, seed=0)
MAX_STEPS = 350

def shape_reward(raw_r, bfs_dist, s, s2, gamma=GAMMA):
    return raw_r + (bfs_dist[s] - gamma * bfs_dist[s2])

env_probe = GridWorld(grid=LARGE_MAZE, max_steps=MAX_STEPS)
n_states, n_actions = env_probe.n_states, 4
n_traversable = sum(1 for row in env_probe.grid for c in row if c != "#")
pairs_needed = n_traversable * 4

print("=== LARGE MAZE — connectome runs at hops=2, NOT directly comparable to the hops=1 tier ===")
print(f"traversable={n_traversable}, pairs_needed={pairs_needed}, live_budget=468, max_steps={MAX_STEPS}")
if pairs_needed > 468:
    raise SystemExit(f"pairs_needed ({pairs_needed}) exceeds live hops=2 budget (468) — regenerate a smaller maze")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "gridworld_results_large_hops2.csv")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["maze", "agent", "seed", "episode", "steps", "total_reward", "reached_goal"])

    for seed in SEEDS:
        env = GridWorld(grid=LARGE_MAZE, max_steps=MAX_STEPS, seed=seed)
        bfs_dist = env.bfs_distances_from_goal()

        traversable_state_ids = [
            env.state_id((r, c))
            for r in range(env.height) for c in range(env.width)
            if not env._is_wall(r, c)
        ]

        agents = [
            ("qlearning", QLearningGridAgent(n_states, n_actions, min_pulls=MIN_PULLS, seed=seed)),
            ("mlp", MLPGridAgent(n_states, n_actions, min_pulls=MIN_PULLS, seed=seed)),
            ("connectome", ConnectomeGridAgent(
                traversable_state_ids, n_actions, LIVE_PATH_HOPS2, bfs_dist,
                hops=HOPS, min_pulls=MIN_PULLS, seed=seed,
            )),
        ]

        for agent_name, agent in agents:
            t_start = time.time()
            recent_successes = []
            for ep in range(N_EPISODES):
                s = env.reset()
                total_r = 0.0
                for t in range(env.max_steps):
                    a = agent.select_action(s)
                    s2, r, done = env.step(a)
                    if agent_name == "connectome":
                        agent.update(s, a, r, s2, done)
                    else:
                        shaped_r = shape_reward(r, bfs_dist, s, s2)
                        agent.update(s, a, shaped_r, s2, done)
                    s = s2
                    total_r += r
                    if done:
                        break
                agent.decay_epsilon()
                reached = env.pos == env.goal
                recent_successes.append(reached)
                writer.writerow(["large_hops2", agent_name, seed, ep, t + 1, total_r, reached])

                if (ep + 1) % PROGRESS_EVERY == 0 or ep == N_EPISODES - 1:
                    elapsed = time.time() - t_start
                    rate = sum(recent_successes[-PROGRESS_EVERY:]) / min(PROGRESS_EVERY, len(recent_successes))
                    eps_per_sec = (ep + 1) / elapsed if elapsed > 0 else float("inf")
                    remaining = (N_EPISODES - (ep + 1)) / eps_per_sec if eps_per_sec > 0 else 0
                    print(f"    [{agent_name} seed={seed}] ep {ep+1}/{N_EPISODES}, "
                          f"recent_success={rate:.2f}, elapsed={elapsed:.1f}s, "
                          f"est_remaining={remaining:.1f}s")
                f.flush()

            print(f"    counts range: min={agent.counts.min()}, max={agent.counts.max()}, "
                  f"pairs over 3x min_pulls: {(agent.counts > 3*MIN_PULLS).sum()}/{agent.counts.size}")
            print(f"  {agent_name} seed={seed} done in {time.time()-t_start:.1f}s")

print(f"Wrote {CSV_PATH}")
