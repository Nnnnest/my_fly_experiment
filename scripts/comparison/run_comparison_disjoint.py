"""Number comparison with training and evaluation operand pairs strictly disjoint.

Both orientations, (a, b) and (b, a), of every evaluation pair are excluded from the training and
calibration streams. Evaluation pairs are the SAME as in scripts/comparison/run_comparison.py for the same
seed (random stream [seed, 1]), so the results are paired with the main runs.

Agents (tier): connectome (fair, native rule), connectome_notrain (control), linear (fair), mlp (fair),
               delta (kc: online logistic delta rule on the KC code), ridge (kc_1x, kc_5x, raw_1x, raw_5x).
Constants are reused from the main comparison (LR_LINEAR, LR_MLP) and from the readout experiment
(delta step 3.0); nothing is tuned here.

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
from scripts.comparison.run_comparison import (build, acc, truth, ROUNDS, N_EVAL, N_CALIB, EVAL_EVERY,
                                               LR_LINEAR, LR_MLP)

EXPERIMENT = "comparison_disjoint"
TASKS = {"knob_n49": {"kind": "knob", "n_max": 49},
         "digits_2": {"kind": "digits", "n_digits": 2},
         "digits_5": {"kind": "digits", "n_digits": 5},
         "digits_9": {"kind": "digits", "n_digits": 9}}
ALPHA_DELTA = 3.0
RIDGE_LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
MAX_MULT = 5

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
    """Returns enc, scale, eval_pairs, calibration_pairs, train_pairs, forbidden.
    eval_pairs reproduce run_comparison.py (stream [seed, 1]); calibration and training streams use the
    streams [seed, 2] and [seed, 3] of the main experiment with forbidden pairs rejected."""
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


def _kc_features(cache, enc, pairs):
    V = np.stack([enc.encode(a, b) for a, b in pairs])
    C = cache._codes(V).astype(np.float64)
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    return np.hstack([C, np.ones((len(C), 1))])


def _raw_features(enc, pairs):
    V = np.stack([enc.encode(a, b) for a, b in pairs]).astype(np.float64) / enc.mag
    return np.hstack([V, np.ones((len(V), 1))])


def _labels(pairs):
    return np.array([1.0 if a > b else -1.0 for a, b in pairs])


def fit_delta(X, y, alpha):
    """One online pass of a logistic delta rule over the training stream."""
    w = np.zeros(X.shape[1])
    for x, t in zip(X, (y > 0).astype(float)):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        w += alpha * (t - p) * x
    return w


def fit_ridge(X, y, seed):
    """Ridge in dual form. Penalty chosen by squared error on a 20% hold-out of the readout's own
    training set, then refitted on all examples."""
    rng = np.random.default_rng([seed, 9])
    n = len(X)
    idx = rng.permutation(n)
    k = max(1, int(0.2 * n))
    va, tr = idx[:k], idx[k:]
    Xt, yt, Xv, yv = X[tr], y[tr], X[va], y[va]
    K = Xt @ Xt.T
    best_lam, best_err = RIDGE_LAMS[0], np.inf
    for lam in RIDGE_LAMS:
        alpha = np.linalg.solve(K + lam * np.eye(len(tr)), yt)
        err = float(np.mean((Xv @ (Xt.T @ alpha) - yv) ** 2))
        if err < best_err:
            best_lam, best_err = lam, err
    alpha = np.linalg.solve(X @ X.T + best_lam * np.eye(n), y)
    return X.T @ alpha, best_lam


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
        if cfg["feat"] == "kc":
            Xtr, Xev = _kc_features(cache, enc, stream), _kc_features(cache, enc, eval_pairs)
        else:
            Xtr, Xev = _raw_features(enc, stream), _raw_features(enc, eval_pairs)
        ytr, yev = _labels(stream), _labels(eval_pairs)
        out = {}
        if spec.agent == "delta":
            w = fit_delta(Xtr, ytr, cfg["alpha"])
        else:
            w, lam = fit_ridge(Xtr, ytr, seed)
            out["ridge_lambda"] = float(lam)
        a = float(np.mean((Xev @ w > 0) == (yev > 0)))
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
