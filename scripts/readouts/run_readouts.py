"""Readouts on identical KC codes (revision plan, priority 1).
For each task and seed, on the SAME train stream and SAME held-out set:
  native        the existing connectome agent (online train() on the KC->MBON weights)
  delta         online logistic delta rule on the KC codes (one pass, same stream)
  ridge_kc_1x   offline ridge on the KC codes, trained on the same number of examples as native
  ridge_kc_5x   offline ridge on the KC codes, 5x more examples (does the code contain the information at all?)
  ridge_raw_*   the same two ridge fits on the raw input vectors (what the KC expansion adds)
    python -m scripts.readouts.run_readouts --smoke
    python -m scripts.readouts.run_readouts --tuning --workers 4     # choose ALPHA_DELTA, edit it below, commit
    python -m scripts.readouts.run_readouts --workers 14"""
import numpy as np

from core import paths
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from encoders.addition.addition_encoder import AdditionEncoder
from encoders.addition.addition_encoder_2digit import TwoDigitAdditionEncoder
from agents.shared.addition_common import sample_batch, sample_triple
from agents.shared.readouts import prep_l2, prep_scaled, accuracy, delta_fit, ridge_fit_select
from agents.addition.connectome_addition_agent import ConnectomeAdditionAgent
from agents.comparison.connectome_compare_agent import ConnectomeCompareAgent
from scripts.comparison.run_comparison import build, acc as cmp_acc, N_EVAL as CMP_EVAL, N_CALIB as CMP_CALIB
from scripts.addition.run_addition import accuracy as add_acc, N_EVAL as ADD_EVAL, N_CALIB as ADD_CALIB

EXPERIMENT = "readouts"
ALPHAS = [0.3, 1.0, 3.0, 10.0, 30.0]
ALPHA_DELTA = 3.0          # PLACEHOLDER: set from the --tuning output (best mean over tasks), then commit
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
BIG = 5                    # the data-rich ridge fit uses BIG x the online number of examples

TASKS = {
    "cmp_knob_n49": {"family": "cmp", "kind": "knob", "n_max": 49, "rounds": 400},
    "cmp_digits_2": {"family": "cmp", "kind": "digits", "n_digits": 2, "rounds": 400},
    "cmp_digits_5": {"family": "cmp", "kind": "digits", "n_digits": 5, "rounds": 400},
    "cmp_digits_9": {"family": "cmp", "kind": "digits", "n_digits": 9, "rounds": 400},
    "add_single": {"family": "add", "width": "single", "rounds": 800},
    "add_two": {"family": "add", "width": "two", "rounds": 800},
}


def codes_of(cache, X, chunk=500):
    return np.concatenate([cache._codes(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def readouts(cache, Xr_tr, ytr, Xr_ev, yev, rounds, seed):
    """Xr_*: raw input vectors (685-dim). The KC codes are exactly what the native readout sees."""
    Ktr, Kev = prep_l2(codes_of(cache, Xr_tr)), prep_l2(codes_of(cache, Xr_ev))
    Rtr, Rev = prep_scaled(Xr_tr, Xr_ev)
    rng = np.random.default_rng([seed, 91])
    out = {}
    for a in ALPHAS:
        out[f"acc_delta_a{a}"] = accuracy(Kev, yev, delta_fit(Ktr[:rounds], ytr[:rounds], a))
    out["acc_delta"] = out[f"acc_delta_a{ALPHA_DELTA}"]
    out["alpha_delta"] = ALPHA_DELTA
    for name, (Xt, Xe) in (("kc", (Ktr, Kev)), ("raw", (Rtr, Rev))):
        for tag, n in (("1x", rounds), ("5x", len(Xt))):
            w, lam = ridge_fit_select(Xt[:n], ytr[:n], LAMBDAS, rng)
            out[f"acc_ridge_{name}_{tag}"] = accuracy(Xe, yev, w)
            out[f"lam_ridge_{name}_{tag}"] = float(lam)
    return out


def run_cmp(cfg, seed):
    rounds, big = cfg["rounds"], cfg["big"]
    enc, sample, _ = build(cfg, seed)
    r_eval, r_cal, r_tr = (np.random.default_rng([seed, k]) for k in (1, 2, 3))   # same streams as run_comparison
    eval_pairs = [sample(r_eval) for _ in range(CMP_EVAL)]
    agent = ConnectomeCompareAgent(enc)
    agent.calibrate_baseline([sample(r_cal) for _ in range(CMP_CALIB)])
    train = [sample(r_tr) for _ in range(big * rounds)]          # first `rounds` = the online stream
    out = {"acc_untrained": cmp_acc(agent, eval_pairs)}
    for a, b in train[:rounds]:
        agent.train_on(a, b)
    out["acc_native"] = cmp_acc(agent, eval_pairs)
    Xtr = np.stack([enc.encode(a, b) for a, b in train])
    Xev = np.stack([enc.encode(a, b) for a, b in eval_pairs])
    ytr = np.array([float(a > b) for a, b in train])
    yev = np.array([float(a > b) for a, b in eval_pairs])
    out.update(readouts(agent.cache, Xtr, ytr, Xev, yev, rounds, seed))
    return out


def run_add(cfg, seed):
    rounds, big = cfg["rounds"], cfg["big"]
    two = cfg["width"] == "two"
    op_max = 49 if two else 9
    rngs = {k: np.random.default_rng([seed, i]) for i, k in enumerate(["cal", "eval", "train", "pool", "probe"], 1)}
    eval_set = sample_batch(rngs["eval"], op_max, ADD_EVAL, two)
    live = paths.live_alpn()
    enc = TwoDigitAdditionEncoder(live, seed=seed) if two else AdditionEncoder(live, n_max=9, seed=seed)
    agent = ConnectomeAdditionAgent(enc, "two" if two else "single")
    agent.calibrate_baseline([t[:3] for t in sample_batch(rngs["cal"], op_max, ADD_CALIB, two)])
    train = [sample_triple(rngs["train"], op_max, rngs["train"].random() < 0.5, two) for _ in range(big * rounds)]
    out = {"acc_untrained": add_acc(agent, eval_set)}
    for a, b, c, label in train[:rounds]:
        agent.update(a, b, c, label)
    out["acc_native"] = add_acc(agent, eval_set)
    Xtr = np.stack([enc.encode(*t[:3]) for t in train])
    Xev = np.stack([enc.encode(*t[:3]) for t in eval_set])
    ytr = np.array([float(t[3]) for t in train])
    yev = np.array([float(t[3]) for t in eval_set])
    out.update(readouts(agent.cache, Xtr, ytr, Xev, yev, rounds, seed))
    return out


def run_one(spec, log):
    cfg = spec.config
    return run_cmp(cfg, spec.seed) if cfg["family"] == "cmp" else run_add(cfg, spec.seed)


def main():
    args = common_parser(__doc__).parse_args()
    names = ["cmp_digits_2", "add_single"] if args.smoke else list(TASKS)
    specs = []
    for name in names:
        t = dict(TASKS[name])
        if args.smoke:
            t["rounds"] = 60
        for seed in resolve_seeds(args):
            specs.append(RunSpec(EXPERIMENT, name, "readouts", "all", seed, {**t, "big": BIG}))
    run_all(specs, run_one, args, cost=lambda s: s.config["rounds"] * s.config["big"])


if __name__ == "__main__":
    main()
