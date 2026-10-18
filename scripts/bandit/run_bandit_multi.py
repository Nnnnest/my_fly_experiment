"""4-armed bandit. Tasks: k4_spread (20/40/60/80 %), k4_close (50/55/60/65 %).
    python -m scripts.bandit.run_bandit_multi --smoke
    python -m scripts.bandit.run_bandit_multi --tuning --workers 14
    python -m scripts.bandit.run_bandit_multi --workers 14"""
from core import paths  # noqa: F401
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from envs.bandit_env import BernoulliBandit
from agents.bandit.qlearning_agent import QLearningBanditAgent
from agents.bandit.mlp_agent import MLPBanditAgent
from agents.bandit.connectome_agent_k import ConnectomeBanditAgentK

EXPERIMENT = "bandit_multi"
PROBLEMS = {"k4_spread": ([0.2, 0.4, 0.6, 0.8], 300), "k4_close": ([0.50, 0.55, 0.60, 0.65], 1000)}
MIN_PULLS_SWEEP = [5, 10, 20]
LR_MLP = 0.1
TUNE_LR_MLP = [0.03, 0.1, 0.3, 1.0]


def make_agent(name, k, min_pulls, seed, lr):
    if name == "qlearning":
        return QLearningBanditAgent(k, min_pulls=min_pulls, seed=seed)
    if name == "mlp":
        return MLPBanditAgent(k, lr=lr, min_pulls=min_pulls, seed=seed)
    return ConnectomeBanditAgentK(k, min_pulls=min_pulls, seed=seed)


def run_one(spec, log):
    cfg = spec.config
    probs, n = cfg["probs"], cfg["n_trials"]
    agent = make_agent(spec.agent, len(probs), cfg["min_pulls"], spec.seed, cfg.get("lr"))
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
        log.log(t, "pseudo_regret", pseudo)
        last_opt.append(probs[arm] == best)
    out = {"total_reward": tot_reward, "cum_regret": cum_regret, "cum_pseudo_regret": cum_pseudo,
           "frac_optimal_last50": sum(last_opt[-50:]) / 50}
    if hasattr(agent, "drift_pct"):
        out["final_drift_pct"] = agent.drift_pct()
    return out


def main():
    args = common_parser(__doc__).parse_args()
    sweep = [10] if args.smoke else MIN_PULLS_SWEEP
    agents = ([("mlp", f"lr{lr}", lr) for lr in TUNE_LR_MLP] if args.tuning else
              [("qlearning", "-", None), ("mlp", "-", LR_MLP), ("connectome", "-", None)])
    specs = []
    for name, (probs, n) in PROBLEMS.items():
        n = 50 if args.smoke else n
        for mp in sweep:
            for agent, tier, lr in agents:
                for seed in resolve_seeds(args):
                    specs.append(RunSpec(EXPERIMENT, f"{name}_mp{mp}", agent, tier, seed,
                                         {"probs": probs, "n_trials": n, "min_pulls": mp, "lr": lr}))
    run_all(specs, run_one, args)


if __name__ == "__main__":
    main()
