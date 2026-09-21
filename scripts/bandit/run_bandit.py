import sys, os, csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from envs.bandit_env import BernoulliBandit
from agents.bandit.qlearning_agent import QLearningBanditAgent
from agents.bandit.mlp_agent import MLPBanditAgent
from agents.bandit.connectome_agent import ConnectomeBanditAgent

PROBS = [0.3, 0.7]
OPTIMAL_REWARD = max(PROBS)
N_TRIALS = 300
SEEDS = [0, 1, 2]
MIN_PULLS = 10

RESULTS_DIR = os.path.join(ROOT, "results", "bandit")
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "bandit_results.csv")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "agent", "seed", "trial", "arm", "reward",
        "cum_reward", "regret", "cum_regret"
    ])

    for seed in SEEDS:
        for agent_name, agent in [
            ("qlearning", QLearningBanditAgent(len(PROBS), min_pulls=MIN_PULLS, seed=seed)),
            ("mlp", MLPBanditAgent(len(PROBS), lr=0.3, min_pulls=MIN_PULLS, seed=seed)),
            ("connectome", ConnectomeBanditAgent(len(PROBS), ["A", "B"], min_pulls=MIN_PULLS, seed=seed)),
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
                    agent_name, seed, t, arm, reward,
                    cum_reward, regret, cum_regret
                ])
                f.flush()
            print(f"{agent_name} seed={seed} final cum_reward={cum_reward} final cum_regret={cum_regret}")

print(f"Wrote results to {CSV_PATH}")
