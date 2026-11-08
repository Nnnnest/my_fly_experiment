"""Does the baseline calibration matter? Native connectome rule with and without calibration.

Tiers (all other settings, streams, seeds and evaluation sets identical):
  full   standard calibration (OLS on label-blind features, subtracted at decision time). Must reproduce the
         'fair' connectome runs of comparison_disjoint / addition_disjoint exactly (checked by analysis.nocalib_tables).
  const  only a constant offset is subtracted: the mean untrained readout over the calibration sample.
  raw    no correction: the decision is the sign of the raw MB_pref.
Tasks use the overlap-free streams: number comparison with disjoint pairs (scripts/comparison/run_comparison_disjoint.py)
and addition with held-out operand pairs (scripts/addition/run_addition_disjoint.py).

    python -m scripts.calibration.run_nocalib --smoke
    python -m scripts.calibration.run_nocalib --workers 14"""
import numpy as np

from core import paths
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from encoders.addition.addition_encoder import AdditionEncoder
from encoders.addition.addition_encoder_2digit import TwoDigitAdditionEncoder
from agents.shared.addition_split import split_pairs, draw
from agents.comparison.connectome_compare_agent import ConnectomeCompareAgent
from agents.addition.connectome_addition_agent import ConnectomeAdditionAgent
import scripts.addition.run_addition as ra
from scripts.addition.run_addition_disjoint import HELD_FRAC, ROUNDS as ADD_ROUNDS
from scripts.comparison.run_comparison import acc as cmp_acc, ROUNDS as CMP_ROUNDS
from scripts.comparison.run_comparison_disjoint import TASKS as CMP_TASKS, make_split

EXPERIMENT = "nocalib"
TIERS = ["full", "const", "raw"]
ADD_TASKS = {"single_online": "single", "two_online": "two"}


def set_offset(agent, raw_cal, tier):
    """Replace the fitted calibration by a constant (const) or by nothing (raw). The last coefficient multiplies the
    constant feature 1.0 in both agents' baseline features, so all other coefficients stay 0."""
    coef = np.zeros_like(np.asarray(agent._coef, float)) if agent._coef is not None else None
    return coef


def run_cmp(spec):
    cfg, seed, tier = spec.config, spec.seed, spec.tier
    rounds = cfg["rounds"]
    enc, scale, eval_pairs, cal, train, _ = make_split(cfg, seed, rounds)
    agent = ConnectomeCompareAgent(enc)
    if tier == "full":
        agent.calibrate_baseline(cal)
    else:
        offset = float(np.mean([agent._raw(a, b) for a, b in cal])) if tier == "const" else 0.0
        agent._coef = np.array([0.0, 0.0, 0.0, offset])          # features [a, b, a*b, 1]
    pre = cmp_acc(agent, eval_pairs)
    for a, b in train[:rounds]:
        agent.train_on(a, b)
    return {"eval_acc_pre": pre, "eval_acc_final": cmp_acc(agent, eval_pairs),
            "final_drift_pct": float(agent.drift_pct())}


def run_add(spec):
    cfg, seed, tier = spec.config, spec.seed, spec.tier
    two = cfg["width"] == "two"
    op_max = 49 if two else 9
    rounds = cfg["rounds"]
    rngs = {k: np.random.default_rng([seed, i]) for i, k in enumerate(["cal", "eval", "train", "pool", "probe"], 1)}
    train_pairs, held_pairs = split_pairs(op_max, seed, HELD_FRAC)
    eval_set = draw(rngs["eval"], held_pairs, ra.N_EVAL, op_max, two)
    cal = [t[:3] for t in draw(rngs["cal"], train_pairs, ra.N_CALIB, op_max, two)]
    train = draw(rngs["train"], train_pairs, rounds, op_max, two)
    live = paths.live_alpn()
    enc = TwoDigitAdditionEncoder(live, seed=seed) if two else AdditionEncoder(live, n_max=9, seed=seed)
    agent = ConnectomeAdditionAgent(enc, cfg["width"])
    if tier == "full":
        agent.calibrate_baseline(cal)
    else:
        offset = float(np.mean(agent._raw_many(cal))) if tier == "const" else 0.0
        coef = np.zeros(len(agent._blind(1, 1, 1)))
        coef[-1] = offset                                         # last feature is the constant 1.0
        agent._coef = coef
    pre = ra.accuracy(agent, eval_set)
    for a, b, c, label in train:
        agent.update(a, b, c, label)
    return {"eval_acc_pre": pre, "eval_acc_final": ra.accuracy(agent, eval_set),
            "final_drift_pct": float(agent.drift_pct())}


def run_one(spec, log):
    out = run_cmp(spec) if spec.config["family"] == "cmp" else run_add(spec)
    log.log(0, "eval_acc", out["eval_acc_pre"], split="eval")
    log.log(spec.config["rounds"], "eval_acc", out["eval_acc_final"], split="eval")
    return out


def main():
    args = common_parser(__doc__).parse_args()
    if args.tuning:
        raise SystemExit("no tuning mode: no hyperparameters are involved")
    cmp_tasks = ["knob_n49"] if args.smoke else list(CMP_TASKS)
    add_tasks = ["single_online"] if args.smoke else list(ADD_TASKS)
    specs = []
    for tier in TIERS:
        for task in cmp_tasks:
            cfg = {**CMP_TASKS[task], "family": "cmp", "rounds": 60 if args.smoke else CMP_ROUNDS,
                   "smoke": bool(args.smoke)}
            specs += [RunSpec(EXPERIMENT, task, "connectome", tier, s, cfg) for s in resolve_seeds(args)]
        for task in add_tasks:
            cfg = {"family": "add", "width": ADD_TASKS[task], "rounds": 60 if args.smoke else ADD_ROUNDS}
            specs += [RunSpec(EXPERIMENT, task, "connectome", tier, s, cfg) for s in resolve_seeds(args)]
    run_all(specs, run_one, args, cost=lambda s: s.config["rounds"])


if __name__ == "__main__":
    main()
