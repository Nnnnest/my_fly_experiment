"""Addition ('is C == A+B?'). Tasks: {single,two}_{online,imitation}.
    python -m scripts.addition.run_addition --smoke
    python -m scripts.addition.run_addition --tuning --workers 4
    python -m scripts.addition.run_addition --workers 4"""
import random

import numpy as np

from core import paths
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from encoders.addition.addition_encoder import AdditionEncoder
from encoders.addition.addition_encoder_2digit import TwoDigitAdditionEncoder
from agents.shared.addition_common import sample_batch, sample_triple
from agents.addition.connectome_addition_agent import ConnectomeAdditionAgent
from agents.addition.addition_gate_utils import GatedAdditionAgent
from agents.addition.baseline_addition_agents import LinearAdditionAgent, MLPAdditionAgent

EXPERIMENT = "addition"
ROUNDS, REPS, POOL, N_EVAL, EVAL_EVERY, N_CALIB = 800, 3, 200, 300, 40, 300
LR_LINEAR, LR_MLP, MARGIN, LINEAR_SCALED = 0.05, 3e-3, 0.02, False
TUNE_LR_LINEAR, TUNE_LR_MLP, TUNE_MARGIN = [0.01, 0.02, 0.05, 0.2], [3e-4, 1e-3, 3e-3, 1e-2], [0.01, 0.02, 0.05, 0.2]


def accuracy(agent, triples):
    y = np.array([t[3] for t in triples])
    if hasattr(agent, "predict_many"):
        p = np.asarray(agent.predict_many([t[:3] for t in triples]))
    else:
        p = np.array([agent.predict(*t[:3]) for t in triples])
    return float((p == y).mean())


def make_agent(spec, op_max, two, rngs):
    cfg, seed = spec.config, spec.seed
    if spec.agent.startswith("connectome"):
        live = paths.live_alpn()
        enc = TwoDigitAdditionEncoder(live, seed=seed) if two else AdditionEncoder(live, n_max=9, seed=seed)
        agent = ConnectomeAdditionAgent(enc, "two" if two else "single")
        agent.calibrate_baseline([t[:3] for t in sample_batch(rngs["cal"], op_max, N_CALIB, two)])  # BEFORE training
        if cfg.get("gate"):
            agent = GatedAdditionAgent(agent, margin_frac=cfg["margin"])
            agent.calibrate_gate([t[:3] for t in sample_batch(rngs["probe"], op_max, 8, two)])
        return agent
    if spec.agent == "linear":
        return LinearAdditionAgent("two" if two else "single", cfg["feat"], lr=cfg["lr"], seed=seed,
                                   scaled=cfg.get("scaled", True))
    return MLPAdditionAgent("two" if two else "single", cfg["feat"], lr=cfg["lr"], seed=seed)

def run_one(spec, log):
    cfg, seed = spec.config, spec.seed
    two, op_max = cfg["width"] == "two", 49 if cfg["width"] == "two" else 9
    rngs = {k: np.random.default_rng([seed, i]) for i, k in enumerate(["cal", "eval", "train", "pool", "probe"], 1)}
    eval_set = sample_batch(rngs["eval"], op_max, cfg["n_eval"], two)
    agent = make_agent(spec, op_max, two, rngs)
    has_drift = hasattr(agent, "drift_pct")
    last = {}

    def evaluate(step):
        last["acc"] = accuracy(agent, eval_set)
        log.log(step, "eval_acc", last["acc"], split="eval")
        if has_drift:
            log.log(step, "drift_pct", agent.drift_pct(), split="eval")
        return last["acc"]

    pre = evaluate(0)
    step = 0
    if cfg["mode"] == "online":
        for rnd in range(1, cfg["rounds"] + 1):
            a, b, c, label = sample_triple(rngs["train"], op_max, rngs["train"].random() < 0.5, two)
            if cfg["train"]:
                agent.update(a, b, c, label)
            step = rnd
            if rnd % cfg["eval_every"] == 0:
                evaluate(rnd)
    else:
        pool = sample_batch(rngs["pool"], op_max, cfg["pool"], two)
        for rep in range(cfg["reps"]):
            order = list(range(len(pool)))
            random.Random(seed + rep).shuffle(order)
            for i in order:
                a, b, c, label = pool[i]
                if cfg["train"]:
                    agent.update(a, b, c, label)
                step += 1
                if step % cfg["eval_every"] == 0:
                    evaluate(step)
        log.log(step, "pool_acc", accuracy(agent, pool), split="eval")
    out = {"eval_acc_pre": pre, "eval_acc_final": last["acc"]}
    if cfg["mode"] == "imitation":
        out["pool_acc"] = accuracy(agent, pool)
    if has_drift:
        out["final_drift_pct"] = float(agent.drift_pct())
    if hasattr(agent, "n_trained"):
        out["n_trained"], out["n_skipped"] = agent.n_trained, agent.n_skipped
    return out


def variants(mode):
    v = [("connectome", "fair", {}), 
         ("linear", "raw", {"feat": "raw", "lr": LR_LINEAR, "scaled": LINEAR_SCALED}),
         ("linear", "feat", {"feat": "feat", "lr": LR_LINEAR, "scaled": LINEAR_SCALED}), 
         ("mlp", "raw", {"feat": "raw", "lr": LR_MLP}),
         ("mlp", "feat", {"feat": "feat", "lr": LR_MLP})]
    if mode == "online":
        v += [("connectome", "gate_error", {"gate": True, "margin": MARGIN}),
              ("connectome_notrain", "control", {"train": False})]
    return v


def tuning_variants():
    v = []
    for feat in ("raw", "feat"):
        v += [("linear", f"{feat}_{'s' if sc else 'u'}_lr{lr}", {"feat": feat, "lr": lr, "scaled": sc})
              for sc in (True, False) for lr in TUNE_LR_LINEAR]
        v += [("mlp", f"{feat}_lr{lr}", {"feat": feat, "lr": lr}) for lr in TUNE_LR_MLP]
    return v + [("connectome", f"gate_error_m{m}", {"gate": True, "margin": m}) for m in TUNE_MARGIN]


def main():
    args = common_parser(__doc__).parse_args()
    small = args.smoke
    base = {"rounds": 80 if small else ROUNDS, "reps": 1 if small else REPS, "pool": 40 if small else POOL,
            "n_eval": 60 if small else N_EVAL, "eval_every": EVAL_EVERY, "train": True}
    if args.tuning:
        jobs = [(f"{w}_online", w, "online", tuning_variants()) for w in ("single", "two")]
    elif small:
        jobs = [("single_online", "single", "online", variants("online")),
                ("two_imitation", "two", "imitation", variants("imitation"))]
    else:
        jobs = [(f"{w}_{m}", w, m, variants(m)) for w in ("single", "two") for m in ("online", "imitation")]
    specs = [RunSpec(EXPERIMENT, task, agent, tier, seed, {**base, "width": w, "mode": m, **extra})
             for task, w, m, vs in jobs for agent, tier, extra in vs for seed in resolve_seeds(args)]
    run_all(specs, run_one, args, cost=lambda s: 4 if s.agent.startswith("connectome") else 1)


if __name__ == "__main__":
    main()
