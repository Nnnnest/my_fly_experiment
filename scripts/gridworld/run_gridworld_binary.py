"""Gridworld with the connectome's BINARY feedback given to the baselines.

In the main experiment Q-learning and the MLP receive numeric potential-based shaping, r + (d(s) - 0.95 d(s')),
while the connectome receives a binary target (reward if the move reduces the shortest-path distance or reaches the
goal, otherwise punishment). Here Q-learning and the MLP receive the same binary signal as a reward of +1 / -1.
Everything else (mazes, start rule, episodes, epsilon schedule, forced exploration, seeds, agents' code) is the main
experiment's: make_agent, policy_accuracy and summarize are imported from run_gridworld.py.

Hyperparameters were tuned for the shaped reward, whose scale differs, so a tuning pass is provided (seeds 1000-1004,
mazes 26/72/113). Choose the best mean overall_success per agent; if the best value sits at the edge of a grid, extend
the grid; if flat, take the interior plateau value. Then set ALPHA_Q_BIN / LR_MLP_BIN below, commit, and run the final.

    python -m scripts.gridworld.run_gridworld_binary --smoke
    python -m scripts.gridworld.run_gridworld_binary --tuning --workers 14
    python -m analysis.binary_maze_tables --tuning
    (edit ALPHA_Q_BIN / LR_MLP_BIN if needed, commit)
    python -m scripts.gridworld.run_gridworld_binary --workers 14
"""
import numpy as np

from core import paths  # noqa: F401
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from envs.gridworld_env import GridWorld
from scripts.gridworld.run_gridworld import (MAZES, get_maze, make_agent, policy_accuracy, summarize,
                                             GAMMA, MIN_PULLS, ALPHA_Q, LR_MLP, TUNE_ALPHA_Q, TUNE_LR_MLP)

EXPERIMENT = "gridworld_binary"
ALPHA_Q_BIN, LR_MLP_BIN = ALPHA_Q, 0.003

def binary_reward(r, done, s, s2, bfs):
    """+1 if the move reaches the goal or reduces the shortest-path distance, else -1 (wall bumps included).
    Same criterion as ConnectomeGridAgentV2.update (target_reward)."""
    goal = bool(done and r > 0)
    return 1.0 if (goal or bfs[s2] < bfs[s]) else -1.0


def run_one(spec, log):
    cfg = spec.config
    env = GridWorld(grid=get_maze(cfg["maze"]), max_steps=cfg["max_steps"], seed=spec.seed)
    bfs = env.bfs_distances_from_goal()
    trav_ids = [env.state_id((r, c)) for r in range(env.height) for c in range(env.width)
                if not env._is_wall(r, c)]
    starts = [p for p in env.traversable_states() if p != env.goal]
    agent = make_agent(spec, env, trav_ids, bfs)
    start_rng = np.random.default_rng([spec.seed, 7])        # same start sequence as the main experiment
    succ, eff = [], []
    for ep in range(cfg["episodes"]):
        s = env.reset()
        if cfg["start"] == "rs":
            pos = starts[int(start_rng.integers(len(starts)))]
            env.set_pos(pos)
            s = env.state_id(pos)
        opt = int(bfs[s])
        for t in range(env.max_steps):
            a = agent.select_action(s)
            s2, r, done = env.step(a)
            agent.update(s, a, binary_reward(r, done, s, s2, bfs), s2, done)
            s = s2
            if done:
                break
        agent.decay_epsilon()
        reached = env.pos == env.goal
        succ.append(reached)
        if reached:
            eff.append(opt / max(t + 1, 1))
        log.log(ep, "reached", reached)
        log.log(ep, "steps", t + 1)
        log.log(ep, "optimal_steps", opt)
        if (ep + 1) % 10 == 0 or ep == cfg["episodes"] - 1:
            log.log(ep, "policy_acc", policy_accuracy(agent, env, trav_ids, bfs), split="eval")
    return {"overall_success": float(np.mean(succ)), "path_efficiency": float(np.mean(eff)) if eff else 0.0,
            "final_policy_acc": policy_accuracy(agent, env, trav_ids, bfs), **summarize(succ)}


def final_variants():
    return [("qlearning", "binary", {"lr": ALPHA_Q_BIN}), ("mlp", "binary", {"lr": LR_MLP_BIN})]


TUNE_LR_MLP_BIN = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]   # extended downward: best of the first grid was its lower edge


def tuning_variants():
    return ([("qlearning", f"a{a}", {"lr": a}) for a in TUNE_ALPHA_Q] +
            [("mlp", f"lr{a}", {"lr": a}) for a in TUNE_LR_MLP_BIN])


def main():
    args = common_parser(__doc__).parse_args()
    if args.smoke:
        tasks = [("maze26", "rs")]
    elif args.tuning:
        tasks = [("maze26", "rs"), ("maze72", "rs"), ("maze113", "rs")]
    else:
        tasks = [(m, "rs") for m in MAZES] + [("maze113", "fs")]
    variants = tuning_variants() if args.tuning else final_variants()
    specs = []
    for maze, start in tasks:
        _, _, max_steps, episodes = MAZES[maze]
        episodes = 30 if args.smoke else episodes
        for agent, tier, extra in variants:
            for seed in resolve_seeds(args):
                cfg = {"maze": maze, "start": start, "max_steps": max_steps, "episodes": episodes,
                       "gamma": GAMMA, "min_pulls": MIN_PULLS, "feedback": "binary", **extra}
                specs.append(RunSpec(EXPERIMENT, f"{maze}_{start}", agent, tier, seed, cfg))
    run_all(specs, run_one, args,
            cost=lambda s: s.config["episodes"] * s.config["max_steps"] * (10 if s.agent == "mlp" else 1))


if __name__ == "__main__":
    main()
