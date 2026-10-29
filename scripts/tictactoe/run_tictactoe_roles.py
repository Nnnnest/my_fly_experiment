"""Role breakdown against the heuristic opponent (explains the connectome [gate_visit] cell: draw 0.526 / loss 0.474).

The main tic-tac-toe RL run for opponent 'heuristic' is replayed with EXACTLY the main configuration, by calling
run_tictactoe.run_rl itself (the trained agent is captured through a wrapper around make_agent), so every
training trajectory is the main one. After training, the greedy agent is evaluated separately as first mover (X) and
second mover (O) against the heuristic opponent, and its greedy first move and reply to the opponent's opening are
recorded. Nothing consumes random numbers before the end of training, so the replayed runs must reproduce the main
runs' summary values; analysis.roles_tables checks this.

    python -m scripts.tictactoe.run_tictactoe_roles --smoke
    python -m scripts.tictactoe.run_tictactoe_roles --workers 14
"""
import numpy as np

from core import paths  # noqa: F401
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from opponents.opponents import heuristic_opponent
from agents.shared.tictactoe_play import play, greedy_policy, opp_policy
import scripts.tictactoe.run_tictactoe as T

EXPERIMENT = "tictactoe_roles"
ROLES_N = 200

# (agent, tier, extra) exactly as in run_tictactoe.rl_variants()
VARIANTS = [("connectome", "gate_visit", {"sym": True, "gate": "visit"}),
            ("connectome", "opt", {"sym": True, "gate": "none"}),
            ("connectome", "gate_error", {"sym": True, "gate": "error", "margin": T.MARGIN}),
            ("qlearning", "opt", {"sym": True, "lr": T.ALPHA_Q}),
            ("mlp", "opt", {"sym": True, "lr": T.LR_MLP})]


def run_one(spec, log):
    captured, orig = [], T.make_agent

    def spy(s, slot=0):
        agent = orig(s, slot)
        captured.append(agent)
        return agent

    T.make_agent = spy
    try:
        out = T.run_rl(spec, log)             # identical to the main run, including all random draws
    finally:
        T.make_agent = orig
    agent = captured[0]
    pol = greedy_policy(agent)
    n = spec.config["roles_n"]
    for role, name in ((1, "first"), (-1, "second")):
        res = []
        for _ in range(n):
            _, r = play({role: pol, -role: opp_policy(heuristic_opponent, -role)})
            res.append(r[role])
        for key, val in (("win", 1), ("draw", 0), ("loss", -1)):
            out[f"{name}_{key}"] = float(np.mean([x == val for x in res]))
    legal = list(range(9))
    out["first_move_greedy"] = int(agent.best_action((0,) * 9, legal))
    opening = (0, 0, 0, 0, -1, 0, 0, 0, 0)                    # heuristic opponent opens in the centre
    out["second_reply_greedy"] = int(agent.best_action(opening, [a for a in legal if a != 4]))
    return out


def main():
    args = common_parser(__doc__).parse_args()
    if args.tuning:
        raise SystemExit("no tuning for this diagnostic")
    base = {"episodes": 60 if args.smoke else T.EPISODES, "eval_every": 30 if args.smoke else T.EVAL_EVERY,
            "n_eval": 20 if args.smoke else T.N_EVAL, "reps": T.REPS, "max_states": 300 if args.smoke else None}
    base["cover_at"] = [60] if args.smoke else [500, 1000, 1500, 2000]
    specs = [RunSpec(EXPERIMENT, "rl_heuristic", agent, tier, seed,
                     {**base, "mode": "rl", "opponent": "heuristic", "roles_n": 20 if args.smoke else ROLES_N, **extra})
             for agent, tier, extra in VARIANTS for seed in resolve_seeds(args)]
    run_all(specs, run_one, args, cost=lambda s: 5 if s.agent == "connectome" else 1)


if __name__ == "__main__":
    main()
