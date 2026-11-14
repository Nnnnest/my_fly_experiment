"""4-armed bandit: reward-only versus reward-and-punishment feedback for the connectome.
Everything (probabilities, arm order, ALPN allocation, forced exploration, seeds, metrics) is imported from
run_bandit_perm.py. Tier '<encoding>_reward'  = existing rule (train(reward=1) on rewarded pulls only; must reproduce
runs/bandit_perm exactly), tier '<encoding>_punish' = additionally train(punish=1) on every unrewarded pull.
    python -m scripts.bandit.run_bandit_punish --smoke
    python -m scripts.bandit.run_bandit_punish --workers 14 [--encodings balanced block]"""
import numpy as np

from core import paths
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from agents.shared.mb_value_cache import train_no_readout
import scripts.bandit.run_bandit_perm as rp

EXPERIMENT = "bandit_punish"
PROBLEMS = {k: v for k, v in rp.PROBLEMS.items() if k.startswith("k4")}


class PunishAgent(rp.ConnectomeBanditAgentK):
    def __init__(self, k, encoding, min_pulls, seed, punish):
        super().__init__(k, encoding, min_pulls, seed)
        self.punish = bool(punish)

    def update(self, arm, reward):
        self.counts[arm] += 1
        if reward > 0:
            train_no_readout(self.brain, self.pats[arm], hops=1, reward=float(reward))
            self.cache.refresh()
        elif self.punish:
            train_no_readout(self.brain, self.pats[arm], hops=1, punish=1.0)
            self.cache.refresh()


_orig_make_agent = rp.make_agent


def _make_agent(spec, k):
    cfg = spec.config
    if "punish" in cfg:
        return PunishAgent(k, cfg["encoding"], cfg["min_pulls"], spec.seed, cfg["punish"])
    return _orig_make_agent(spec, k)


rp.make_agent = _make_agent       # rp.run_one resolves make_agent by name at call time


def run_one(spec, log):           # wrapper so that spawned workers import THIS module (and therefore the patch above)
    out = rp.run_one(spec, log)
    probs = rp.arm_probs(spec.config, spec.seed)
    arms = np.array([v for (_, _, m, v) in log.rows if m == "arm"], int)
    rank = np.argsort(np.argsort(-np.array(probs)))          # 0 = best arm
    ranks = rank[arms]
    for r in range(len(probs)):
        out[f"share_rank{r}_last200"] = float((ranks[-200:] == r).mean())
    out["frac_best_all"] = float((ranks == 0).mean())
    return out


def main():
    ap = common_parser(__doc__)
    ap.add_argument("--encodings", nargs="+", default=["balanced"], choices=["balanced", "block"])
    args = ap.parse_args()
    if args.tuning:
        raise SystemExit("no tuning mode: the connectome has no tunable hyperparameter here")
    sweep = [10] if args.smoke else rp.MIN_PULLS_SWEEP
    specs = []
    for pname, (probs, n) in PROBLEMS.items():
        n = 50 if args.smoke else n
        for order in rp.ORDERS:
            for mp in sweep:
                for enc in args.encodings:
                    for punish in (False, True):
                        tier = f"{enc}_{'punish' if punish else 'reward'}"
                        for seed in resolve_seeds(args):
                            specs.append(RunSpec(EXPERIMENT, f"{pname}_{order}_mp{mp}", "connectome", tier, seed,
                                                 {"probs": probs, "order": order, "n_trials": n, "min_pulls": mp,
                                                  "lr": None, "encoding": enc, "punish": punish}))
    run_all(specs, run_one, args, cost=lambda s: s.config["n_trials"])


if __name__ == "__main__":
    main()
