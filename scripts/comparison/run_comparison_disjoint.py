"""Number comparison with training and evaluation operand pairs strictly disjoint.

Both orientations, (a, b) and (b, a), of every evaluation pair are excluded from the training and calibration
streams. Evaluation pairs are the SAME as in scripts/comparison/run_comparison.py for the same seed (random stream
[seed, 1]), so results are paired with the main runs. The training stream is the main experiment's stream [seed, 3]
with evaluation pairs rejected: it is identical to the main stream whenever no rejection occurs (always the case for
5 and 9 digits).

Readouts use EXACTLY the code of scripts/readouts/run_readouts.py (agents/shared/readouts.py: prep_l2, prep_scaled,
delta_fit, ridge_fit_select with penalty chosen by hold-out ACCURACY; constants LAMBDAS and ALPHA_DELTA imported from
that script) and reproduce its random-generator sequence ([seed, 91], one permutation per ridge fit in the order
kc_1x, kc_5x, raw_1x, raw_5x). Therefore, wherever no pair is rejected (digits_5, digits_9) the delta and ridge accuracies
must equal those of the 'readouts' experiment exactly; analysis.disjoint_cmp_tables checks this.

Agents (tier): connectome (fair, native rule), connectome_notrain (control), linear (fair), mlp (fair),
               delta (kc), ridge (kc_1x, kc_5x, raw_1x, raw_5x).

    python -m scripts.comparison.run_comparison_disjoint --smoke
    python -m scripts.comparison.run_comparison_disjoint --workers 14
"""
import json
import os

import numpy as np

from core import paths
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from core.brain_pool import fresh_brain, value_fn
from agents.comparison.connectome_compare_agent import ConnectomeCompareAgent
from agents.comparison.baseline_compare_agents import LinearCompareAgent, MLPCompareAgent
from agents.shared.readouts import prep_l2, prep_scaled, accuracy, delta_fit, ridge_fit_select
from scripts.comparison.run_comparison import (build, acc, ROUNDS, N_EVAL, N_CALIB, EVAL_EVERY, LR_LINEAR, LR_MLP)
from scripts.readouts.run_readouts import codes_of, LAMBDAS, ALPHA_DELTA

EXPERIMENT = "comparison_disjoint"
TASKS = {"knob_n49": {"kind": "knob", "n_max": 49},
         "digits_2": {"kind": "digits", "n_digits": 2},
         "digits_5": {"kind": "digits", "n_digits": 5},
         "digits_9": {"kind": "digits", "n_digits": 9}}
MAX_MULT = 5
RIDGE_ORDER = [("kc", 1), ("kc", 5), ("raw", 1), ("raw", 5)]     # order of ridge fits in run_readouts.readouts()

# (agent, tier, extra config)
VARIANTS = [("connectome", "fair", {"train": True}),
            ("connectome_notrain", "control", {"train": False}),
            ("linear", "fair", {"train": True, "lr": LR_LINEAR}),
            ("mlp", "fair", {"train": True, "lr": LR_MLP}),
            ("delta", "kc", {"feat": "kc", "mult": 1, "alpha": ALPHA_DELTA}),
            ("ridge", "kc_1x", {"feat": "kc", "mult": 1}),
            ("ridge", "kc_5x", {"feat": "kc", "mult": 5}),
            ("ridge", "raw_1x", {"feat": "raw", "mult": 1}),
            ("ridge", "raw_5x", {"feat": "raw", "mult": 5})]


def make_split(cfg, seed, n_train):
    """Returns enc, scale, eval_pairs, calibration_pairs, train_pairs, forbidden."""
    enc, sample, scale = build(cfg, seed)
    r_eval, r_cal, r_tr = (np.random.default_rng([seed, k]) for k in (1, 2, 3))
    eval_pairs = [sample(r_eval) for _ in range(N_EVAL)]
    forbidden = set(eval_pairs) | {(b, a) for a, b in eval_pairs}

    def draw(r, n):
        out = []
        while len(out) < n:
            p = sample(r)
            if p not in forbidden:
                out.append(p)
        return out

    return enc, scale, eval_pairs, draw(r_cal, N_CALIB), draw(r_tr, n_train), forbidden


def ridge_rng(seed, feat, mult, rounds):
    """Generator state of run_readouts.readouts() just before the ridge fit (feat, mult)."""
    rng = np.random.default_rng([seed, 91])
    for f, m in RIDGE_ORDER:
        if (f, m) == (feat, mult):
            return rng
        rng.permutation(m * rounds)          # ridge_fit_select consumes exactly one permutation of len(X)
    raise ValueError((feat, mult))


def _save_split(cfg, task, seed, eval_pairs, cal, train):
    d = os.path.join(paths.RUNS_DIR, "_splits_smoke" if cfg["smoke"] else "_splits", EXPERIMENT)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{task}_s{seed:04d}.json")
    if os.path.exists(p):
        return
    tmp = f"{p}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump({"eval": eval_pairs, "calibration": cal, "train": train}, f)
    os.replace(tmp, p)          # content is deterministic, so concurrent writers are harmless


def run_one(spec, log):
    cfg, seed, task = spec.config, spec.seed, spec.task
    rounds = cfg["rounds"]
    enc, scale, eval_pairs, cal, train, forbidden = make_split(cfg, seed, MAX_MULT * rounds)
    assert all(p not in forbidden for p in cal + train), "train/calibration stream overlaps evaluation pairs"
    _save_split(cfg, task, seed, eval_pairs, cal, train)

    if spec.agent in ("connectome", "connectome_notrain", "linear", "mlp"):
        stream = train[:rounds]
        if spec.agent.startswith("connectome"):
            agent = ConnectomeCompareAgent(enc)
            agent.calibrate_baseline(cal)                      # BEFORE any training
        elif spec.agent == "linear":
            agent = LinearCompareAgent(lr=cfg["lr"], seed=seed, scale=scale)
        else:
            agent = MLPCompareAgent(lr=cfg["lr"], seed=seed, scale=scale)
        has_drift = hasattr(agent, "drift_pct")

        def evaluate(step):
            a = acc(agent, eval_pairs)
            log.log(step, "eval_acc", a, split="eval")
            if has_drift:
                log.log(step, "drift_pct", agent.drift_pct(), split="eval")
            return a

        pre = evaluate(0)
        ev = {0: pre}
        for t, (a, b) in enumerate(stream):
            if cfg["train"]:
                agent.train_on(a, b)
            if (t + 1) % cfg["eval_every"] == 0:
                ev[t + 1] = evaluate(t + 1)
        out = {"eval_acc_pre": pre, "eval_acc_final": ev[rounds]}
        if has_drift:
            out["final_drift_pct"] = agent.drift_pct()
    else:
        n = rounds * cfg["mult"]
        stream = train[:n]
        fresh_brain()
        cache, _ = value_fn()
        Xr_tr = np.stack([enc.encode(a, b) for a, b in train])           # full 5x stream, as in run_readouts.py
        Xr_ev = np.stack([enc.encode(a, b) for a, b in eval_pairs])
        ytr = np.array([float(a > b) for a, b in train])
        yev = np.array([float(a > b) for a, b in eval_pairs])
        if cfg["feat"] == "kc":
            Xtr, Xev = prep_l2(codes_of(cache, Xr_tr)), prep_l2(codes_of(cache, Xr_ev))
        else:
            Xtr, Xev = prep_scaled(Xr_tr, Xr_ev)
        out = {}
        if spec.agent == "delta":
            w = delta_fit(Xtr[:rounds], ytr[:rounds], cfg["alpha"])
        else:
            w, lam = ridge_fit_select(Xtr[:n], ytr[:n], LAMBDAS, ridge_rng(seed, cfg["feat"], cfg["mult"], rounds))
            out["ridge_lambda"] = float(lam)
        a = accuracy(Xev, yev, w)
        log.log(n, "eval_acc", a, split="eval")
        out["eval_acc_final"] = a
    assert all(p not in forbidden for p in stream)
    out["n_train"] = len(stream)
    out["n_train_unique"] = len(set(stream))
    out["n_overlap_with_eval"] = 0
    return out


def main():
    args = common_parser(__doc__).parse_args()
    if args.tuning:
        raise SystemExit("no tuning here: constants are reused from the main comparison and readout experiments")
    tasks = {k: TASKS[k] for k in ("knob_n49", "digits_2")} if args.smoke else TASKS
    rounds = 60 if args.smoke else ROUNDS
    specs = []
    for task, tcfg in tasks.items():
        for agent, tier, extra in VARIANTS:
            for seed in resolve_seeds(args):
                cfg = {**tcfg, "rounds": rounds, "n_eval": N_EVAL, "n_calib": N_CALIB, "eval_every": EVAL_EVERY,
                       "smoke": bool(args.smoke), "train": True, "lr": None, **extra}
                specs.append(RunSpec(EXPERIMENT, task, agent, tier, seed, cfg))
    run_all(specs, run_one, args, cost=lambda s: 4 if s.agent.startswith("connectome") else 1)


if __name__ == "__main__":
    main()
