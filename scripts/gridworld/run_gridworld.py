import sys, os, csv, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from envs.gridworld_env import GridWorld, GRID
from envs.maze_generator import generate_maze
from agents.gridworld.qlearning_gridworld_agent import QLearningGridAgent
from agents.gridworld.mlp_gridworld_agent import MLPGridAgent
from agents.gridworld.connectome_gridworld_agent import ConnectomeGridAgent

N_EPISODES = 300
SEEDS = [0, 1, 2]
MIN_PULLS = 5
GAMMA = 0.95
PROGRESS_EVERY = 10   # print a progress line every N episodes
LIVE_PATH_HOPS1 = os.path.join(ROOT,"results", "gridworld", "live_alpn_indices.npy")

MAZES = {
    "original_26": {"grid": GRID, "max_steps": 100},
    "medium_72": {"grid": generate_maze(15, 11, extra_connections=0.1, seed=0), "max_steps": 250},
}

def shape_reward(raw_r, bfs_dist, s, s2, gamma=GAMMA):
    return raw_r + (bfs_dist[s] - gamma * bfs_dist[s2])

RESULTS_DIR = os.path.join(ROOT, "results", "gridworld")
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "gridworld_results_multimaze.csv")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["maze", "agent", "seed", "episode", "steps", "total_reward", "reached_goal"])

    for maze_name, cfg in MAZES.items():
        env_probe = GridWorld(grid=cfg["grid"], max_steps=cfg["max_steps"])
        n_states, n_actions = env_probe.n_states, 4
        n_traversable = sum(1 for row in env_probe.grid for c in row if c != "#")
        print(f"=== maze={maze_name}, traversable={n_traversable}, pairs_needed={n_traversable*4}, max_steps={cfg['max_steps']} ===")

        for seed in SEEDS:
            env = GridWorld(grid=cfg["grid"], max_steps=cfg["max_steps"], seed=seed)
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
                    traversable_state_ids, n_actions, LIVE_PATH_HOPS1, bfs_dist,
                    hops=1, min_pulls=MIN_PULLS, seed=seed,
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
                    writer.writerow([maze_name, agent_name, seed, ep, t + 1, total_r, reached])

                    if (ep + 1) % PROGRESS_EVERY == 0 or ep == N_EPISODES - 1:
                        elapsed = time.time() - t_start
                        rate = sum(recent_successes[-PROGRESS_EVERY:]) / min(PROGRESS_EVERY, len(recent_successes))
                        eps_per_sec = (ep + 1) / elapsed if elapsed > 0 else float("inf")
                        remaining = (N_EPISODES - (ep + 1)) / eps_per_sec if eps_per_sec > 0 else 0
                        print(f"    [{agent_name} seed={seed}] ep {ep+1}/{N_EPISODES}, "
                              f"recent_success={rate:.2f}, elapsed={elapsed:.1f}s, "
                              f"est_remaining={remaining:.1f}s")
                    f.flush()

                print(f"  {agent_name} seed={seed} done in {time.time()-t_start:.1f}s")

print(f"Wrote {CSV_PATH}")
