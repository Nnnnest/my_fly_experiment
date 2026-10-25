"""Addition with held-out operand pairs (fixes the item overlap between training and evaluation).
20% of the (a, b) pairs are held out per seed: evaluation uses only held-out pairs, training and calibration only the
others, so no evaluation triple (a, b, c) occurs in training. All agents of the online addition experiment, plus the
readouts on identical KC codes (native, delta rule, ridge), are run on the same disjoint streams.
    python -m scripts.addition.run_addition_disjoint --smoke
    python -m scripts.addition.run_addition_disjoint --workers 14"""
import numpy as np

from core import paths
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from encoders.addition.addition_encoder import AdditionEncoder
from encoders.addition.addition_encoder_2digit import TwoDigitAdditionEncoder
from agents.shared.addition_split import split_pairs, draw
from agents.addition.connectome_addition_agent import ConnectomeAdditionAgent
from agents.addition.addition_gate_utils import GatedAdditionAgent
from agents.addition.baseline_addition_agents import LinearAdditionAgent, MLPAdditionAgent
import scripts.addition.run_addition as ra
from scripts.readouts.run_readouts import readouts

EXPERIMENT = "addition_disjoint"
HELD_FRAC = 0.2
BIG = 5            # readouts: ridge also fitted on BIG x the online number of examples
ROUNDS = 800


def build_agent(spec, two, cal, probe):
    cfg, seed = spec.config, spec.seed
    width = "two" if two else "single"
    if spec.agent.startswith("connectome") or spec.agent == "readouts":
        live = paths.live_alpn()
        enc = TwoDigitAdditionEncoder(live, seed=seed) if two else AdditionEncoder(live, n_max=9, seed=seed)
        agent = ConnectomeAdditionAgent(enc, width)
        agent.calibrate_baseline(cal)                       # label-blind, drawn from TRAINING pairs only
        if cfg.get("gate"):
            agent = GatedAdditionAgent(agent, margin_frac=cfg["margin"])
            agent.calibrate_gate(probe)
        return agent, enc
    if spec.agent == "linear":
        return LinearAdditionAgent(width, cfg["feat"], lr=cfg["lr"], seed=seed, scaled=cfg.get("scaled", True)), None
    return MLPAdditionAgent(width, cfg["feat"], lr=cfg["lr"], seed=seed), None


def run_one(spec, log):
    cfg, seed = spec.config, spec.seed
    two = cfg["width"] == "two"
    op_max = 49 if two else 9
    rounds = cfg["rounds"]
    rngs = {k: np.random.default_rng([seed, i]) for i, k in enumerate(["cal", "eval", "train", "pool", "probe"], 1)}
    train_pairs, held_pairs = split_pairs(op_max, seed, HELD_FRAC)
    eval_set = draw(rngs["eval"], held_pairs, ra.N_EVAL, op_max, two)
    cal = [t[:3] for t in draw(rngs["cal"], train_pairs, ra.N_CALIB, op_max, two)]
    probe = [t[:3] for t in draw(rngs["probe"], train_pairs, 8, op_max, two)]
    train = draw(rngs["train"], train_pairs, cfg["big"] * rounds, op_max, two)
    overlap = {t[:3] for t in train} & {t[:3] for t in eval_set}
    assert not overlap, f"{len(overlap)} evaluation triples occur in the training stream"

    agent, enc = build_agent(spec, two, cal, probe)
    pre = ra.accuracy(agent, eval_set)
    if cfg.get("train", True) and spec.agent != "connectome_notrain":
        for a, b, c, label in train[:rounds]:
            agent.update(a, b, c, label)
    final = ra.accuracy(agent, eval_set)
    out = {"eval_acc_pre": pre, "eval_acc_final": final, "n_held_pairs": len(held_pairs),
           "n_train_pairs": len(train_pairs), "train_eval_overlap": 0}
    if hasattr(agent, "drift_pct"):
        out["final_drift_pct"] = float(agent.drift_pct())
    if spec.agent == "readouts":
        Xtr = np.stack([enc.encode(*t[:3]) for t in train])
        Xev = np.stack([enc.encode(*t[:3]) for t in eval_set])
        ytr = np.array([float(t[3]) for t in train])
        yev = np.array([float(t[3]) for t in eval_set])
        out["acc_untrained"], out["acc_native"] = pre, final
        out.update(readouts(agent.cache, Xtr, ytr, Xev, yev, rounds, seed))
    return out


def main():
    args = common_parser(__doc__).parse_args()
    if args.tuning:
        raise SystemExit("no tuning mode: constants are those of the main addition experiment")
    base = {"rounds": 60 if args.smoke else ROUNDS, "big": BIG, "train": True}
    widths = ["single"] if args.smoke else ["single", "two"]
    variants = ra.variants("online") + [("readouts", "all", {})]
    specs = [RunSpec(EXPERIMENT, f"{w}_online", agent, tier, seed, {**base, "width": w, **extra})
             for w in widths for agent, tier, extra in variants for seed in resolve_seeds(args)]
    run_all(specs, run_one, args,
            cost=lambda s: (4 if s.agent in ("readouts", "connectome") else 1) * s.config["rounds"])


if __name__ == "__main__":
    main()
