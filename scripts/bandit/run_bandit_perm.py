"""4-armed bandit with arm order decoupled from arm index (positional-bias control).
Orders: rev  = probabilities in descending order (best arm = index 0)
        perm = a fresh random permutation per seed
Connectome tiers: block    = arm i drives the i-th contiguous quarter of the 685 ALPN slots (as in bandit_multi)
                  balanced = live ALPN positions shuffled per seed and split into equal groups
    python -m scripts.bandit.run_bandit_perm --smoke
    python -m scripts.bandit.run_bandit_perm --tuning --workers 14
    python -m scripts.bandit.run_bandit_perm --workers 14"""
import numpy as np

from core import paths
from core.brain_pool import fresh_brain, cache_for
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from envs.bandit_env import BernoulliBandit
from agents.bandit.qlearning_agent import QLearningBanditAgent
from agents.bandit.mlp_agent import MLPBanditAgent
from agents.shared.mb_value_cache import train_no_readout

EXPERIMENT = "bandit_perm"
PROBLEMS = {"k2": ([0.3, 0.7], 300), "k4_spread": ([0.2, 0.4, 0.6, 0.8], 300), "k4_close": ([0.50, 0.55, 0.60, 0.65], 1000)}
ORDERS = ["rev", "perm"]
MIN_PULLS_SWEEP = [5, 10, 20]
LR_MLP = 0.1
TUNE_LR_MLP = [0.03, 0.1, 0.3, 1.0]


class ConnectomeBanditAgentK:
    def __init__(self, k, encoding, min_pulls, seed, epsilon=0.1):
        self.brain = fresh_brain()
        n = len(self.brain.ALPN)
        pats = np.zeros((k, n), dtype=np.float32)
        if encoding == "block":
            for i in range(k):
                pats[i, i * n // k:(i + 1) * n // k] = 1.0
        else:
            live = np.random.default_rng([seed, 5]).permutation(paths.live_alpn())
            for i, chunk in enumerate(np.array_split(live, k)):
                pats[i, chunk] = 1.0
        self.pats = pats
        name = f"bandit_perm_{encoding}_k{k}" + (f"_s{seed}" if encoding == "balanced" else "")
        self.cache = cache_for(name, pats)
        self.epsilon, self.min_pulls = epsilon, min_pulls
        self.counts = np.zeros(k)
        self.rng = np.random.default_rng(seed)
        self.baseline = self.cache.values_all().copy()

    def select_action(self):
        under = np.where(self.counts < self.min_pulls)[0]
        if len(under) > 0:
            return int(under[0])
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.counts)))
        return int(np.argmax(self.cache.values_all() - self.baseline))

    def update(self, arm, reward):
        self.counts[arm] += 1
        if reward > 0:
            train_no_readout(self.brain, self.pats[arm], hops=1, reward=float(reward))
            self.cache.refresh()

    def drift_pct(self):
        num = float(np.abs(self.brain.wM - self.brain.wM0).mean())
        return 100.0 * num / (float(np.abs(self.brain.wM0).mean()) + 1e-9)


def arm_probs(cfg, seed):
    base = list(cfg["probs"])
    if cfg["order"] == "rev":
        return base[::-1]
    perm = np.random.default_rng([seed, 11]).permutation(len(base))
    return [base[i] for i in perm]


def make_agent(spec, k):
    cfg = spec.config
    if spec.agent == "qlearning":
        return QLearningBanditAgent(k, min_pulls=cfg["min_pulls"], seed=spec.seed)
    if spec.agent == "mlp":
        return MLPBanditAgent(k, lr=cfg["lr"], min_pulls=cfg["min_pulls"], seed=spec.seed)
    return ConnectomeBanditAgentK(k, spec.tier, cfg["min_pulls"], spec.seed)


def run_one(spec, log):
    cfg = spec.config
    probs = arm_probs(cfg, spec.seed)
    n, k = cfg["n_trials"], len(probs)
    agent = make_agent(spec, k)
    env = BernoulliBandit(probs, seed=spec.seed)
    best, best_arm = max(probs), int(np.argmax(probs))
    tot_reward = cum_regret = cum_pseudo = 0.0
    arms, last_opt = [], []
    for t in range(n):
        arm = int(agent.select_action())
        reward = env.pull(arm)
        agent.update(arm, reward)
        pseudo = best - probs[arm]
        tot_reward += reward
        cum_regret += best - reward
        cum_pseudo += pseudo
        arms.append(arm)
        log.log(t, "arm", arm)
        log.log(t, "reward", reward)
        log.log(t, "pseudo_regret", pseudo)
        last_opt.append(probs[arm] == best)
    out = {"total_reward": tot_reward, "cum_regret": cum_regret, "cum_pseudo_regret": cum_pseudo,
           "frac_optimal_last50": sum(last_opt[-50:]) / 50, "best_arm": best_arm,
           "top_arm": int(np.bincount(arms[-200:], minlength=k).argmax())}
    if hasattr(agent, "drift_pct"):
        out["final_drift_pct"] = agent.drift_pct()
    return out


def main():
    args = common_parser(__doc__).parse_args()
    sweep = [10] if args.smoke else MIN_PULLS_SWEEP
    if args.tuning:
        agents = [("mlp", f"lr{lr}", lr) for lr in TUNE_LR_MLP]
    else:
        agents = [("qlearning", "-", None), ("mlp", "-", LR_MLP),
                  ("connectome", "block", None), ("connectome", "balanced", None)]
    specs = []
    for pname, (probs, n) in PROBLEMS.items():
        n = 50 if args.smoke else n
        for order in ORDERS:
            for mp in sweep:
                for agent, tier, lr in agents:
                    for seed in resolve_seeds(args):
                        specs.append(RunSpec(EXPERIMENT, f"{pname}_{order}_mp{mp}", agent, tier, seed,
                                             {"probs": probs, "order": order, "n_trials": n,
                                              "min_pulls": mp, "lr": lr}))
    run_all(specs, run_one, args,
            cost=lambda s: s.config["n_trials"] * (5 if s.agent == "connectome" else 1))


if __name__ == "__main__":
    main()
