"""Tier 3 (imitation) for grid world -- teaches the connectome the
BFS-shortest-path-optimal action directly at every reachable state,
instead of learning it through trial-and-error shaped reward (Tiers
1/1-opt/2). Same fixed reward=1.0/punish=1.0 sign-gate convention as
every other train() call in this project. Mirrors
train_imitation_tictactoe.py's approach (label the optimal move, label
one alternative as wrong, repeat)."""
import argparse, os, random, sys, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.gridworld_env import GridWorld, ACTIONS, GRID
from envs.maze_generator import generate_maze
from agents.gridworld.connectome_gridworld_agent_v2 import ConnectomeGridAgentV2
from agents.shared.mb_value_cache import ensure_live_hops1

REPS_PER_STATE = 3
LIVE_PATH = os.path.join(ROOT, "results", "gridworld", "live_alpn_indices.npy")
MODELS_DIR = os.path.join(ROOT, "models", "gridworld")
SUMMARY_CSV = os.path.join(ROOT, "results", "gridworld", "tier3_imitation_summary.csv")


def optimal_and_others(sid, env, bfs_dist):
    """optimal = actions whose resulting cell is strictly closer to goal;
    other = every remaining legal action (including bumping into a wall)."""
    r, c = divmod(sid, env.width)
    d0 = bfs_dist[sid]
    optimal, other = [], []
    for a, (dr, dc) in ACTIONS.items():
        nr, nc = r + dr, c + dc
        nsid = sid if env._is_wall(nr, nc) else env.state_id((nr, nc))
        (optimal if bfs_dist[nsid] < d0 else other).append(a)
    return optimal, other


def policy_accuracy(agent, env, trav_ids, bfs_dist):
    ok = n = 0
    for sid in trav_ids:
        r, c = divmod(sid, env.width)
        if (r, c) == env.goal:
            continue
        dr, dc = ACTIONS[agent.greedy_action(sid)]
        nr, nc = r + dr, c + dc
        nsid = sid if env._is_wall(nr, nc) else env.state_id((nr, nc))
        ok += bfs_dist[nsid] < bfs_dist[sid]
        n += 1
    return ok / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maze", choices=["small_26", "medium_72", "large_113"], default="large_113")
    ap.add_argument("--shuffle", dest="shuffle", action="store_true", default=True)
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-overlap", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.maze == "small_26":
        grid, max_steps = GRID, 100
    elif args.maze == "medium_72":
        grid, max_steps = generate_maze(15, 11, extra_connections=0.1, seed=0), 250
    else:
        grid, max_steps = generate_maze(19, 13, extra_connections=0.1, seed=0), 350

    env = GridWorld(grid=grid, max_steps=max_steps, seed=args.seed)
    bfs_dist = env.bfs_distances_from_goal()
    trav_ids = [env.state_id((r, c)) for r in range(env.height)
                for c in range(env.width) if not env._is_wall(r, c)]
    n_states = len(trav_ids)
    print(f"maze={args.maze} traversable={n_states} pairs={n_states * 4}")

    ensure_live_hops1(LIVE_PATH)
    agent = ConnectomeGridAgentV2(
        trav_ids, 4, LIVE_PATH, bfs_dist, hops=1, gate="none", encoding="khot",
        k=args.k, max_overlap=args.max_overlap, seed=args.seed)

    rng = random.Random(args.seed)
    order = list(range(n_states))
    if args.shuffle:
        rng.shuffle(order)

    t0 = time.time()
    n_trained = n_skipped = 0
    for i, s in enumerate(order):
        sid = trav_ids[s]
        optimal, other = optimal_and_others(sid, env, bfs_dist)
        if not optimal or not other:
            n_skipped += 1
            continue
        for _ in range(REPS_PER_STATE):
            best_a, other_a = rng.choice(optimal), rng.choice(other)
            agent._train(agent.encoder.encode(s, best_a), reward=1.0)
            agent._train(agent.encoder.encode(s, other_a), punish=1.0)
            n_trained += 2
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{n_states} states done ({time.time()-t0:.0f}s, {n_trained} train() calls)")

    acc = policy_accuracy(agent, env, trav_ids, bfs_dist)
    secs = time.time() - t0
    print(f"\ndone in {secs:.0f}s. {n_skipped} states had no improving action (skipped). "
          f"greedy-policy accuracy: {acc:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    suffix = "_shuffle" if args.shuffle else ""
    weights_path = os.path.join(MODELS_DIR, f"connectome_imitation_{args.maze}{suffix}_weights.npz")
    agent.brain.save_weights(weights_path)
    print(f"saved weights to {weights_path}")

    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    write_header = not os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a") as f:
        if write_header:
            f.write("maze,shuffle,seed,n_states,n_trained,n_skipped,policy_accuracy,seconds\n")
        f.write(f"{args.maze},{args.shuffle},{args.seed},{n_states},{n_trained},{n_skipped},{acc:.4f},{secs:.1f}\n")
    print(f"appended to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
