"""Bandit experiment: probs [0.3, 0.7], min_pulls sweep, 3 agents, FINAL_SEEDS (10).
    python -m scripts.bandit.run_bandit --smoke
    python -m scripts.bandit.run_bandit"""
from core import paths
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from envs.bandit_env import BernoulliBandit
from agents.bandit.qlearning_agent import QLearningBanditAgent
from agents.bandit.mlp_agent import MLPBanditAgent
from agents.bandit.connectome_agent import ConnectomeBanditAgent

EXPERIMENT = "bandit"
PROBS = [0.3, 0.7]
N_TRIALS = 300
MIN_PULLS_SWEEP = [2, 5, 8, 10, 15, 20, 30]     # main results = min_pulls 10
AGENTS = ["qlearning", "mlp", "connectome"]

LR_MLP = 0.1 
TUNE_LR_MLP = [0.03, 0.1, 0.3, 1.0]


def make_agent(name, min_pulls, seed, lr):
    k = len(PROBS)
    if name == "qlearning":
        return QLearningBanditAgent(k, min_pulls=min_pulls, seed=seed)
    if name == "mlp":
        return MLPBanditAgent(k, lr=lr, min_pulls=min_pulls, seed=seed)
    return ConnectomeBanditAgent(k, ["A", "B"], min_pulls=min_pulls, seed=seed)


def run_one(spec, log):
    cfg = spec.config
    probs, n = cfg["probs"], cfg["n_trials"]
    agent = make_agent(spec.agent, cfg["min_pulls"], spec.seed, cfg.get("lr"))
    env = BernoulliBandit(probs, seed=spec.seed)
    best = max(probs)
    tot_reward = cum_regret = cum_pseudo = 0.0
    last_opt = []
    for t in range(n):
        arm = agent.select_action()
        reward = env.pull(arm)
        agent.update(arm, reward)
        pseudo = best - probs[arm]
        tot_reward += reward
        cum_regret += best - reward
        cum_pseudo += pseudo
        log.log(t, "arm", arm)
        log.log(t, "reward", reward)
        log.log(t, "regret", best - reward)          # realized (as in v1)
        log.log(t, "pseudo_regret", pseudo)          # expected, less noisy
        last_opt.append(probs[arm] == best)
    out = {"total_reward": tot_reward, "cum_regret": cum_regret, "cum_pseudo_regret": cum_pseudo,
           "frac_optimal_last50": sum(last_opt[-50:]) / 50}
    if hasattr(agent, "drift_pct"):
        out["final_drift_pct"] = agent.drift_pct()
    return out


def main():
    args = common_parser(__doc__).parse_args()
    sweep = [10] if args.smoke else MIN_PULLS_SWEEP
    n = 50 if args.smoke else N_TRIALS
    agents = ([("mlp", f"lr{lr}", lr) for lr in TUNE_LR_MLP] if args.tuning else
              [("qlearning", "-", None), ("mlp", "-", LR_MLP), ("connectome", "-", None)])
    specs = [RunSpec(EXPERIMENT, f"mp{mp}", agent, tier, seed,
                     {"probs": PROBS, "n_trials": n, "min_pulls": mp, "lr": lr})
             for mp in sweep for agent, tier, lr in agents for seed in resolve_seeds(args)]
    run_all(specs, run_one, args)


if __name__ == "__main__":
    main()
