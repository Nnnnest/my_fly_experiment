"""Gridworld. Mazes maze26/72/113/316, random start (rs); fixed start (fs) only for maze113.
    python -m scripts.gridworld.run_gridworld --smoke
    python -m scripts.gridworld.run_gridworld --tuning
    python -m scripts.gridworld.run_gridworld"""
from functools import lru_cache

import numpy as np

from core import paths
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from envs.gridworld_env import GridWorld, ACTIONS, GRID
from envs.maze_generator import generate_maze, count_traversable
from agents.gridworld.qlearning_gridworld_agent import QLearningGridAgent
from agents.gridworld.mlp_gridworld_agent import MLPGridAgent
from agents.gridworld.connectome_gridworld_agent_v2 import ConnectomeGridAgentV2
from agents.gridworld.connectome_kc_delta_agent import ConnectomeKCDeltaAgent

EXPERIMENT = "gridworld"
GAMMA, MIN_PULLS, WINDOW, POL_EVERY = 0.95, 5, 20, 10
# tuned on tuning seeds (see --tuning); overwrite after tuning:
ALPHA_Q, LR_MLP, ALPHA_KC, ALPHA_RAND, MARGIN_ERROR, MARGIN_RANK = 0.1, 1e-2, 1.0, 1.0, 0.05, 0.02
TUNE_ALPHA_Q = [0.05, 0.1, 0.3, 0.5]
TUNE_LR_MLP = [3e-3, 1e-2, 3e-2]
TUNE_ALPHA_DELTA = [0.1, 0.2, 0.5, 1.0, 1.5]
TUNE_MARGIN = [0.01, 0.02, 0.05, 0.2]

# name -> (width, height, max_steps, episodes); maze26 is the hand-made GRID
MAZES = {"maze26": (None, None, 100, 300), "maze72": (15, 11, 250, 300),
         "maze113": (19, 13, 350, 400), "maze316": (31, 21, 500, 600)}


@lru_cache(maxsize=None)
def get_maze(name):
    w, h = MAZES[name][:2]
    return GRID if w is None else generate_maze(w, h, extra_connections=0.1, seed=0)


def make_agent(spec, env, trav_ids, bfs):
    cfg, seed, live = spec.config, spec.seed, paths.LIVE_ALPN
    if spec.agent == "qlearning":
        return QLearningGridAgent(env.n_states, 4, alpha=cfg["lr"], min_pulls=MIN_PULLS, seed=seed)
    if spec.agent == "mlp":
        return MLPGridAgent(env.n_states, 4, lr=cfg["lr"], min_pulls=MIN_PULLS, seed=seed)
    if spec.agent == "connectome":
        return ConnectomeGridAgentV2(trav_ids, 4, live, bfs, hops=1, min_pulls=MIN_PULLS, seed=seed, **cfg["conn"])
    return ConnectomeKCDeltaAgent(trav_ids, 4, live, min_pulls=MIN_PULLS, seed=seed, alpha=cfg["lr"],
                                  features="kc" if spec.agent == "kc_delta" else "random")


def greedy(agent, sid):
    if hasattr(agent, "greedy_action"):
        return agent.greedy_action(sid)
    if hasattr(agent, "q"):
        return int(np.argmax(agent.q[sid]))
    import torch
    with torch.no_grad():
        return int(torch.argmax(agent.net(agent._one_hot_batch([sid]))[0]).item())


def policy_accuracy(agent, env, trav_ids, bfs):
    ok = n = 0
    for sid in trav_ids:
        r, c = divmod(sid, env.width)
        if (r, c) == env.goal:
            continue
        dr, dc = ACTIONS[greedy(agent, sid)]
        nr, nc = r + dr, c + dc
        nsid = sid if env._is_wall(nr, nc) else env.state_id((nr, nc))
        ok += bfs[nsid] < bfs[sid]
        n += 1
    return ok / max(n, 1)


def summarize(succ):
    x = np.asarray(succ, float)
    if len(x) < WINDOW:
        return {"peak20": float(x.mean()), "final20": float(x.mean()), "first_ep_ge90": None, "collapsed": False}
    roll = np.convolve(x, np.ones(WINDOW) / WINDOW, mode="valid")
    hit = np.where(roll >= 0.9)[0]
    peak, final = float(roll.max()), float(roll[-1])
    return {"peak20": peak, "final20": final, "first_ep_ge90": int(hit[0]) + WINDOW if len(hit) else None,
            "collapsed": bool(peak >= 0.5 and peak - final > 0.3)}


def run_one(spec, log):
    cfg = spec.config
    env = GridWorld(grid=get_maze(cfg["maze"]), max_steps=cfg["max_steps"], seed=spec.seed)
    bfs = env.bfs_distances_from_goal()
    trav_ids = [env.state_id((r, c)) for r in range(env.height) for c in range(env.width)
                if not env._is_wall(r, c)]
    starts = [p for p in env.traversable_states() if p != env.goal]
    agent = make_agent(spec, env, trav_ids, bfs)
    start_rng = np.random.default_rng([spec.seed, 7])        # same start sequence for every agent of a seed
    raw_reward = spec.agent == "connectome"                  # gated agent derives its target from bfs itself
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
            rr = r if raw_reward else r + (bfs[s] - GAMMA * bfs[s2])
            agent.update(s, a, rr, s2, done)
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
        if hasattr(agent, "n_trained"):
            log.log(ep, "n_trained", agent.n_trained)
            log.log(ep, "n_skipped", getattr(agent, "n_skipped", 0))
        if (ep + 1) % POL_EVERY == 0 or ep == cfg["episodes"] - 1:
            log.log(ep, "policy_acc", policy_accuracy(agent, env, trav_ids, bfs), split="eval")
    out = {"overall_success": float(np.mean(succ)), "path_efficiency": float(np.mean(eff)) if eff else 0.0,
           "final_policy_acc": policy_accuracy(agent, env, trav_ids, bfs), **summarize(succ)}
    if hasattr(agent, "n_trained"):
        out["n_trained"], out["n_skipped"] = int(agent.n_trained), int(getattr(agent, "n_skipped", 0))
    if hasattr(agent, "drift_pct"):
        out["final_drift_pct"] = agent.drift_pct()
    return out


def conn(tier, **kw):
    return ("connectome", tier, {"conn": kw})


def final_variants(pairs):
    v = [("qlearning", "fair", {"lr": ALPHA_Q}), ("mlp", "fair", {"lr": LR_MLP})]
    if pairs <= len(paths.live_alpn()):
        v.append(conn("fair", encoding="onehot", gate="none"))
    v += [conn("opt", encoding="khot", gate="none"), conn("gate_visit", encoding="khot", gate="visit"),
          conn("gate_error", encoding="khot", gate="error", margin_frac=MARGIN_ERROR),
          conn("gate_rank", encoding="khot", gate="rank", margin_frac=MARGIN_RANK),
          ("kc_delta", "diagnostic", {"lr": ALPHA_KC}), ("rand_delta", "diagnostic", {"lr": ALPHA_RAND})]
    return v


def tuning_variants():
    return ([("qlearning", f"a{a}", {"lr": a}) for a in TUNE_ALPHA_Q] +
            [("mlp", f"lr{a}", {"lr": a}) for a in TUNE_LR_MLP] +
            [("kc_delta", f"a{a}", {"lr": a}) for a in TUNE_ALPHA_DELTA] +
            [("rand_delta", f"a{a}", {"lr": a}) for a in TUNE_ALPHA_DELTA] +
            [conn(f"gate_error_m{m}", encoding="khot", gate="error", margin_frac=m) for m in TUNE_MARGIN] +
            [conn(f"gate_rank_m{m}", encoding="khot", gate="rank", margin_frac=m) for m in TUNE_MARGIN])


def main():
    args = common_parser(__doc__).parse_args()
    if args.smoke:
        tasks = [("maze26", "rs")]
    elif args.tuning:
        tasks = [("maze26", "rs"), ("maze72", "rs"), ("maze113", "rs")]
    else:
        tasks = [(m, "rs") for m in MAZES] + [("maze113", "fs")]
    specs = []
    for maze, start in tasks:
        _, _, max_steps, episodes = MAZES[maze]
        episodes = 30 if args.smoke else episodes
        pairs = 4 * count_traversable(get_maze(maze))
        variants = tuning_variants() if args.tuning else final_variants(pairs)
        for agent, tier, extra in variants:
            for seed in resolve_seeds(args):
                cfg = {"maze": maze, "start": start, "max_steps": max_steps, "episodes": episodes,
                       "gamma": GAMMA, "min_pulls": MIN_PULLS, **extra}
                specs.append(RunSpec(EXPERIMENT, f"{maze}_{start}", agent, tier, seed, cfg))
    run_all(specs, run_one, args,
        cost=lambda s: s.config["episodes"] * s.config["max_steps"] * (10 if s.agent == "mlp" else 1))


if __name__ == "__main__":
    main()
