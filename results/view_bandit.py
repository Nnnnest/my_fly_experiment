import pandas as pd
df = pd.read_csv("my_experiment/results/bandit_results.csv")
summary = df.groupby(["agent", "trial"])["reward"].mean().unstack("agent")
print(summary.tail())          # average reward near the end of training
regret = df.groupby("agent").apply(lambda g: (0.7 - g["reward"]).cumsum().iloc[-1])
print(regret)                  # total regret per agent, lower is better
