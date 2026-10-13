"""Number comparison. Tasks: knob_n{9,19,49,99} (Method A) and digits_{2,3,4,5,9} (Method B).
    python -m scripts.comparison.run_comparison --smoke
    python -m scripts.comparison.run_comparison"""
import numpy as np

from core import paths
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from encoders.comparison.number_pair_encoder import NumberPairEncoder, sample_pair
from encoders.comparison.multi_digit_pair_encoder import MultiDigitPairEncoder, sample_n_digit_pair
from agents.comparison.connectome_compare_agent import ConnectomeCompareAgent
from agents.comparison.baseline_compare_agents import LinearCompareAgent, MLPCompareAgent

EXPERIMENT = "comparison"
ROUNDS, N_EVAL, EVAL_EVERY, N_CALIB = 400, 200, 20, 100
KNOB_N = [9, 19, 49, 99]
DIGITS = [2, 3, 4, 5, 9]
LR_LINEAR, LR_MLP = 2.0, 3e-3          
TUNE_LINEAR = [0.1, 0.5, 2.0, 8.0]
TUNE_MLP = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
# (agent, tier, trains, sleep_every, lr)
FIXED_VARIANTS = [("connectome", "fair", True, None, None), ("connectome", "sleep50", True, 50, None),
                  ("connectome_notrain", "control", False, None, None)]

def build(cfg, seed):
    live = paths.live_alpn()
    if cfg["kind"] == "knob":
        n = cfg["n_max"]
        return NumberPairEncoder(live, n_max=n, seed=seed), (lambda r: sample_pair(r, n_max=n)), float(n)
    d = cfg["n_digits"]
    return MultiDigitPairEncoder(live, n_digits=d, seed=seed), (lambda r: sample_n_digit_pair(r, d)), float(10 ** d)


def truth(a, b):
    return "A" if a > b else "B"


def acc(agent, pairs):
    return float(np.mean([agent.predict(a, b)[0] == truth(a, b) for a, b in pairs]))


def run_one(spec, log):
    cfg, seed = spec.config, spec.seed
    enc, sample, scale = build(cfg, seed)
    is_conn = spec.agent.startswith("connectome")
    if is_conn:
        agent = ConnectomeCompareAgent(enc)
    elif spec.agent == "linear":
        agent = LinearCompareAgent(lr=cfg["lr"], seed=seed, scale=scale)
    else:
        agent = MLPCompareAgent(lr=cfg["lr"], seed=seed, scale=scale)

    r_eval, r_cal, r_tr = (np.random.default_rng([seed, k]) for k in (1, 2, 3))
    eval_pairs = [sample(r_eval) for _ in range(N_EVAL)]
    if is_conn:
        agent.calibrate_baseline([sample(r_cal) for _ in range(N_CALIB)])   # BEFORE any training

    def evaluate(step):
        a = acc(agent, eval_pairs)
        log.log(step, "eval_acc", a, split="eval")
        if is_conn:
            log.log(step, "drift_pct", agent.drift_pct(), split="eval")
        return a

    pre = evaluate(0)
    ev, online = {0: pre}, []
    for t in range(cfg["rounds"]):
        a, b = sample(r_tr)
        ok = float(agent.predict(a, b)[0] == truth(a, b))
        online.append(ok)
        log.log(t, "online_correct", ok)
        log.log(t, "abs_diff", abs(a - b))
        if cfg["train"]:
            agent.train_on(a, b)
        if cfg["sleep_every"] and (t + 1) % cfg["sleep_every"] == 0:
            agent.sleep()
        if (t + 1) % EVAL_EVERY == 0:
            ev[t + 1] = evaluate(t + 1)
    first90 = next((s for s, v in sorted(ev.items()) if v >= 0.9), None)
    out = {"eval_acc_pre": pre, "eval_acc_final": ev[cfg["rounds"]], "online_acc": float(np.mean(online)),
           "online_acc_last40": float(np.mean(online[-40:])), "rounds_to_eval90": first90}
    if is_conn:
        out["final_drift_pct"] = agent.drift_pct()
    return out


def main():
    args = common_parser(__doc__).parse_args()
    if args.smoke:
        tasks = {"knob_n9": {"kind": "knob", "n_max": 9}, "digits_2": {"kind": "digits", "n_digits": 2}}
        rounds = 60
    else:
        tasks = {f"knob_n{n}": {"kind": "knob", "n_max": n} for n in KNOB_N}
        tasks.update({f"digits_{d}": {"kind": "digits", "n_digits": d} for d in DIGITS})
        rounds = ROUNDS
    if args.tuning:     # baselines only: the connectome has no tunable hyperparameter here
        variants = ([("linear", f"lr{lr}", True, None, lr) for lr in TUNE_LINEAR] +
                    [("mlp", f"lr{lr}", True, None, lr) for lr in TUNE_MLP])
    else:
        variants = FIXED_VARIANTS + [("linear", "fair", True, None, LR_LINEAR), ("mlp", "fair", True, None, LR_MLP)]
    specs = []
    for task, tcfg in tasks.items():
        for agent, tier, trains, sleep, lr in variants:
            for seed in resolve_seeds(args):
                cfg = {**tcfg, "rounds": rounds, "train": trains, "sleep_every": sleep, "lr": lr,
                       "n_eval": N_EVAL, "eval_every": EVAL_EVERY}
                specs.append(RunSpec(EXPERIMENT, task, agent, tier, seed, cfg))
    run_all(specs, run_one, args)

if __name__ == "__main__":
    main()
