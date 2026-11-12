"""Capacity-matched digit-position comparison (essential_experiments_plan.md, experiment 2).
Main experiment: the ~302 live ALPNs are divided among 2*n_digits groups, so per-position capacity falls with n_digits.
Here the connectome is also run with a FIXED number of ALPNs per digit position:
  tier fixed : the main experiment's allocation (reproduces runs/comparison exactly)
  tier g16   : 16 ALPNs per position for every digit length (largest common value up to 9 digits: 18 x 16 = 288)
  tier g30   : 30 ALPNs per position (digit lengths 2-5 only: 10 x 30 = 300)
Agents per tier: connectome (trained), connectome_notrain (control), readouts_cap (ridge / delta on the KC code and on
the raw input, plus KC-activity statistics on the evaluation pairs). Same seeds, evaluation pairs and training
streams as the main comparison experiment, so results pair on (task, seed).
    python -m scripts.comparison.run_comparison_capacity --smoke
    python -m scripts.comparison.run_comparison_capacity --workers 14"""
import numpy as np

from core import paths
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from core.brain_pool import fresh_brain, value_fn
from encoders.comparison.multi_digit_pair_encoder import sample_n_digit_pair
from encoders.comparison.capacity_digit_pair_encoder import CapacityDigitPairEncoder
from agents.shared.kc_native import kc_activity, mask_stats
import scripts.comparison.run_comparison as rc
from scripts.readouts.run_readouts import readouts, codes_of

EXPERIMENT = "comparison_capacity"
_orig_build = rc.build


def build(cfg, seed):
    """Like run_comparison.build; per_position=None keeps the main allocation."""
    if cfg.get("per_position") is None:
        return _orig_build(cfg, seed)
    d = cfg["n_digits"]
    enc = CapacityDigitPairEncoder(paths.live_alpn(), n_digits=d, per_position=cfg["per_position"], seed=seed)
    return enc, (lambda r: sample_n_digit_pair(r, d)), float(10 ** d)


rc.build = build      # run_comparison.run_one resolves build by name at call time


def run_readouts_cap(spec, log):
    cfg, seed, rounds = spec.config, spec.seed, spec.config["rounds"]
    enc, sample, _ = build(cfg, seed)
    r_eval, r_tr = np.random.default_rng([seed, 1]), np.random.default_rng([seed, 3])
    eval_pairs = [sample(r_eval) for _ in range(rc.N_EVAL)]
    train = [sample(r_tr) for _ in range(5 * rounds)]
    fresh_brain()
    cache, _ = value_fn()
    Xtr = np.stack([enc.encode(a, b) for a, b in train])
    Xev = np.stack([enc.encode(a, b) for a, b in eval_pairs])
    ytr = np.array([float(a > b) for a, b in train])
    yev = np.array([float(a > b) for a, b in eval_pairs])
    out = readouts(cache, Xtr, ytr, Xev, yev, rounds, seed)
    st = mask_stats(kc_activity(cache, Xev))
    g = enc.group_size
    out.update(n_alpn_per_position=int(g), n_alpn_total=int(2 * cfg["n_digits"] * g),
               kc_pos_mean=float(st["n_pos"].mean()), kc_read_mean=float(st["n_read"].mean()),
               kc_read_std=float(st["n_read"].std()), kc_read_p05=float(np.quantile(st["n_read"], 0.05)),
               kc_empty_frac=float((st["n_read"] == 0).mean()), n_kc=int(st["n_kc"]), k=int(st["k"]))
    return out


def run_one(spec, log):     # wrapper so that spawned workers import THIS module (and therefore the patch above)
    return run_readouts_cap(spec, log) if spec.agent == "readouts_cap" else rc.run_one(spec, log)


def regimes(d, n_live):
    r = [("fixed", None), ("g16", 16)]
    if 2 * d * 30 <= n_live:
        r.append(("g30", 30))
    return r


def main():
    args = common_parser(__doc__).parse_args()
    if args.tuning:
        raise SystemExit("no tuning mode: constants are those of the main comparison experiment")
    digits = [2, 5] if args.smoke else rc.DIGITS
    rounds = 60 if args.smoke else rc.ROUNDS
    n_live = len(paths.live_alpn())
    specs = []
    for d in digits:
        for tier, g in regimes(d, n_live):
            for agent, trains in (("connectome", True), ("connectome_notrain", False), ("readouts_cap", True)):
                for seed in resolve_seeds(args):
                    cfg = {"kind": "digits", "n_digits": d, "per_position": g, "rounds": rounds, "train": trains,
                           "sleep_every": None, "lr": None, "n_eval": rc.N_EVAL, "eval_every": rc.EVAL_EVERY}
                    specs.append(RunSpec(EXPERIMENT, f"digits_{d}", agent, tier, seed, cfg))
    run_all(specs, run_one, args, cost=lambda s: 4 if s.agent.startswith("connectome") else 2)


if __name__ == "__main__":
    main()
