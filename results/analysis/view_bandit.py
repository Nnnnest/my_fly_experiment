import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "..", "results", "bandit", "bandit_results.csv")

df = pd.read_csv(RESULT_PATH)
summary = df.groupby(["agent", "trial"])["reward"].mean().unstack("agent")
print(summary.tail())          # average reward near the end of training
regret = df.groupby("agent").apply(lambda g: (0.7 - g["reward"]).cumsum().iloc[-1])
print(regret)                  # total regret per agent, lower is better
