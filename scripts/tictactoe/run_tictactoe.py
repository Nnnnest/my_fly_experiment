"""Tic-tac-toe. Tasks: rl_random, rl_heuristic, rl_selfplay (2000 episodes) and imitation.
    python -m scripts.tictactoe.run_tictactoe --smoke
    python -m scripts.tictactoe.run_tictactoe --tuning --workers 8
    python -m scripts.tictactoe.run_tictactoe --workers 8"""
import random
from functools import lru_cache

import numpy as np

from core import paths
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from envs.tictactoe_env import TicTacToe
from opponents.opponents import random_opponent, heuristic_opponent
from agents.tictactoe.qlearning_tictactoe_agent import QLearningTicTacToeAgent
from agents.tictactoe.mlp_tictactoe_agent import MLPTicTacToeAgent
from agents.tictactoe.connectome_tictactoe_agent_v2 import ConnectomeTicTacToeAgentV2
from agents.shared.symmetry_wrapper import SymmetryWrapper
from agents.shared.tictactoe_play import play, train_policy, greedy_policy, opp_policy
from agents.shared.tictactoe_solver import enumerate_reachable_states, minimax
from agents.shared.tictactoe_eval import full_coverage_accuracy, to_perspective, afterstate

EXPERIMENT = "tictactoe"
MIN_PULLS, EPS0, EPS1 = 5, 0.1, 0.01
EPISODES, EVAL_EVERY, N_EVAL, DRIFT_EVERY, REPS = 2000, 250, 200, 50, 3
ALPHA_Q, LR_MLP, MARGIN = 0.1, 3e-3, 0.05
TUNE_ALPHA_Q, TUNE_LR_MLP, TUNE_MARGIN = [0.02, 0.05, 0.1, 0.3], [3e-4, 1e-3, 3e-3, 1e-2], [0.01, 0.02, 0.05, 0.2]
OPP = {"random": random_opponent, "heuristic": heuristic_opponent}


@lru_cache(maxsize=None)
def get_states():
    return enumerate_reachable_states()


def early_boards(seed, n=300, max_depth=4):
    """Calibration sample for the connectome's untrained-baseline fit (before any training)."""
    rng, env, boards = random.Random(seed * 1000 + 5), TicTacToe(), set()
    while len(boards) < n:
        env.reset()
        for _ in range(rng.randint(0, max_depth)):
            env.step(rng.choice(env.legal_actions()))
        boards.add(tuple(env.board))
    return sorted(boards)


def make_agent(spec, slot=0):
    c = spec.config
    if spec.agent == "qlearning":
        inner = QLearningTicTacToeAgent(alpha=c["lr"], min_pulls=MIN_PULLS)
    elif spec.agent == "mlp":
        inner = MLPTicTacToeAgent(lr=c["lr"], min_pulls=MIN_PULLS)
    else:
        inner = ConnectomeTicTacToeAgentV2(min_pulls=MIN_PULLS, gate=c["gate"], seed=spec.seed, slot=slot,
                                           margin_frac=c.get("margin", MARGIN))
    return SymmetryWrapper(inner) if c["sym"] else inner


def eval_vs(agent, opp_fn, n):
    res = []
    for g in range(n):
        m = 1 if g % 2 == 0 else -1
        _, r = play({m: greedy_policy(agent), -m: opp_policy(opp_fn, -m)})
        res.append(r[m])
    return res


def checkpoint(agent, log, step, cfg, last):
    for name, fn in OPP.items():
        res = eval_vs(agent, fn, cfg["n_eval"])
        for key, val in (("win", 1), ("draw", 0), ("loss", -1)):
            last[f"eval_{name}_{key}"] = float(np.mean([r == val for r in res]))
            log.log(step, f"eval_{name}_{key}", last[f"eval_{name}_{key}"], split="eval")
    if step in cfg["cover_at"]:
        last["coverage_acc"] = full_coverage_accuracy(agent, get_states())[0] / 100.0
        log.log(step, "coverage_acc", last["coverage_acc"], split="eval")


def run_rl(spec, log):
    cfg, opp = spec.config, spec.config["opponent"]
    is_conn = spec.agent == "connectome"
    boards = early_boards(spec.seed)
    a1 = make_agent(spec, 0)
    a2 = make_agent(spec, 1) if opp == "selfplay" else None
    if is_conn:
        a1.calibrate_baseline(boards)
        if a2:
            a2.calibrate_baseline(boards)
    eps, decay = EPS0, (EPS1 / EPS0) ** (1 / cfg["episodes"])
    last, outcomes = {}, []
    checkpoint(a1, log, 0, cfg, last)
    for ep in range(cfg["episodes"]):
        m1 = 1 if ep % 2 == 0 else -1
        other = train_policy(a2, eps) if a2 else opp_policy(OPP.get(opp), -m1)
        after, res = play({m1: train_policy(a1, eps), -m1: other})
        a1.update_episode(after[m1], res[m1])
        if a2:
            a2.update_episode(after[-m1], res[-m1])
        outcomes.append(res[m1])
        log.log(ep, "outcome", res[m1])
        if cfg.get("sleep_every") and (ep + 1) % cfg["sleep_every"] == 0:
            a1.sleep()
            if a2:
                a2.sleep()
        eps *= decay
        if is_conn and (ep + 1) % DRIFT_EVERY == 0:
            log.log(ep, "drift_pct", a1.drift_pct())
            log.log(ep, "n_trained", a1.n_trained)
            log.log(ep, "n_skipped", a1.n_skipped)
        if (ep + 1) % cfg["eval_every"] == 0:
            checkpoint(a1, log, ep + 1, cfg, last)
    o = np.array(outcomes)
    w, d, l = (o == 1).astype(float), (o == 0).astype(float), (o == -1).astype(float)
    roll = np.convolve(w, np.ones(100) / 100, mode="valid")
    out = {"win": w.mean(), "draw": d.mean(), "loss": l.mean(), "last100_win": w[-100:].mean(),
           "last100_draw": d[-100:].mean(), "last100_loss": l[-100:].mean(), "peak100_win": roll.max(),
           "collapsed": bool(roll.max() >= 0.3 and roll.max() - roll[-1] > 0.3), **last}
    if is_conn:
        out["final_drift_pct"], out["n_trained"], out["n_skipped"] = a1.drift_pct(), a1.n_trained, a1.n_skipped
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def run_imitation(spec, log):
    cfg = spec.config
    states = get_states()
    order = list(states)
    random.Random(spec.seed).shuffle(order)
    if cfg["max_states"]:
        order = order[:cfg["max_states"]]
    agent = make_agent(spec)
    if spec.agent == "connectome":
        agent.calibrate_baseline(early_boards(spec.seed))
    pre = full_coverage_accuracy(agent, states)[0] / 100.0
    log.log(0, "coverage_acc", pre, split="eval")
    for i, (raw, mover) in enumerate(order):
        _, best_a = minimax(raw, mover)
        mine = to_perspective(raw, mover)
        others = [a for a in range(9) if mine[a] == 0 and a != best_a]
        if others:
            b_after, o_after = afterstate(mine, best_a), afterstate(mine, random.choice(others))
            for _ in range(cfg["reps"]):
                agent.update_episode([b_after], 1)
                agent.update_episode([o_after], -1)
        if cfg.get("sleep_every") and (i + 1) % cfg["sleep_every"] == 0:
            agent.sleep()
    post = full_coverage_accuracy(agent, states)[0] / 100.0
    log.log(len(order), "coverage_acc", post, split="eval")
    out = {"coverage_acc_pre": pre, "coverage_acc": post}
    if spec.agent == "connectome":
        out["final_drift_pct"] = agent.drift_pct()
    return out


def run_one(spec, log):
    return run_imitation(spec, log) if spec.config["mode"] == "imitation" else run_rl(spec, log)


def rl_variants():
    v = []
    for sym, tier in ((False, "fair"), (True, "opt")):
        v += [("qlearning", tier, {"sym": sym, "lr": ALPHA_Q}), ("mlp", tier, {"sym": sym, "lr": LR_MLP}),
              ("connectome", tier, {"sym": sym, "gate": "none"})]
    return v + [("connectome", "opt_sleep100", {"sym": True, "gate": "none", "sleep_every": 100}),
                ("connectome", "gate_error", {"sym": True, "gate": "error", "margin": MARGIN}),
                ("connectome", "gate_visit", {"sym": True, "gate": "visit"})]


def imit_variants():
    return [("qlearning", "fair", {"sym": False, "lr": ALPHA_Q}), ("qlearning", "sym", {"sym": True, "lr": ALPHA_Q}),
            ("mlp", "fair", {"sym": False, "lr": LR_MLP}), ("mlp", "sym", {"sym": True, "lr": LR_MLP}),
            ("connectome", "fair", {"sym": False, "gate": "none"}),
            ("connectome", "sym", {"sym": True, "gate": "none"}),
            ("connectome", "sym_sleep500", {"sym": True, "gate": "none", "sleep_every": 500})]


def tuning_variants(sym):
    v = ([("qlearning", f"a{a}", {"sym": sym, "lr": a}) for a in TUNE_ALPHA_Q] +
         [("mlp", f"lr{a}", {"sym": sym, "lr": a}) for a in TUNE_LR_MLP])
    if sym:
        v += [("connectome", f"gate_error_m{m}", {"sym": True, "gate": "error", "margin": m}) for m in TUNE_MARGIN]
    return v


def main():
    args = common_parser(__doc__).parse_args()
    small = args.smoke
    base = {"episodes": 60 if small else EPISODES, "eval_every": 30 if small else EVAL_EVERY,
            "n_eval": 20 if small else N_EVAL, "reps": REPS, "max_states": 300 if small else None}
    base["cover_at"] = [60] if small else [500, 1000, 1500, 2000]
    jobs = []
    if args.tuning:
        for opp in ("random", "selfplay"):
            for sym in (False, True):
                jobs.append((f"rl_{opp}_{'sym' if sym else 'fair'}", "rl", opp, tuning_variants(sym)))
    else:
        for opp in (["random"] if small else ["random", "heuristic", "selfplay"]):
            jobs.append((f"rl_{opp}", "rl", opp, rl_variants()))
        jobs.append(("imitation", "imitation", "imitation", imit_variants()))
    specs = [RunSpec(EXPERIMENT, task, agent, tier, seed, {**base, "mode": mode, "opponent": opp, **extra})
             for task, mode, opp, variants in jobs for agent, tier, extra in variants
             for seed in resolve_seeds(args)]
    run_all(specs, run_one, args,
            cost=lambda s: (10 if s.config["mode"] == "imitation" else 3) * (5 if s.agent == "connectome" else 1))


if __name__ == "__main__":
    main()
