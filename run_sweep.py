import sys, os, csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from envs.bandit_env import BernoulliBandit
from agents.qlearning_agent import QLearningBanditAgent
from agents.mlp_agent import MLPBanditAgent
from agents.connectome_agent import ConnectomeBanditAgent

PROBS = [0.3, 0.7]
OPTIMAL_REWARD = max(PROBS)
N_TRIALS = 300
SEEDS = [0, 1, 2, 3, 4]
MIN_PULLS_VALUES = [2, 5, 8, 10, 15, 20, 30]

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "min_pulls_sweep.csv")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "agent", "min_pulls", "seed", "trial", "arm", "reward",
        "cum_reward", "regret", "cum_regret"
    ])

    for min_pulls in MIN_PULLS_VALUES:
        for seed in SEEDS:
            for agent_name, agent in [
                ("qlearning", QLearningBanditAgent(len(PROBS), min_pulls=min_pulls, seed=seed)),
                ("mlp", MLPBanditAgent(len(PROBS), lr=0.3, min_pulls=min_pulls, seed=seed)),
                ("connectome", ConnectomeBanditAgent(len(PROBS), ["A", "B"], min_pulls=min_pulls, seed=seed)),
            ]:
                env = BernoulliBandit(PROBS, seed=seed)
                cum_reward = 0.0
                cum_regret = 0.0
                for t in range(N_TRIALS):
                    arm = agent.select_action()
                    reward = env.pull(arm)
                    agent.update(arm, reward)

                    regret = OPTIMAL_REWARD - reward
                    cum_reward += reward
                    cum_regret += regret

                    writer.writerow([
                        agent_name, min_pulls, seed, t, arm, reward,
                        cum_reward, regret, cum_regret
                    ])
                f.flush()
        print(f"min_pulls={min_pulls} done")

print(f"Wrote {CSV_PATH}")
