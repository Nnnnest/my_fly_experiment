import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "min_pulls_sweep.csv")

df = pd.read_csv(CSV_PATH)

# take the last trial per (agent, min_pulls, seed) run
final = df.loc[df.groupby(["agent", "min_pulls", "seed"])["trial"].idxmax()]

# flag incomplete sweeps before trusting the averages
seed_counts = final.groupby(["agent", "min_pulls"])["seed"].nunique().unstack("agent")
expected_seeds = df["seed"].nunique()
if (seed_counts != expected_seeds).any().any():
    print("WARNING: not every (agent, min_pulls) combo has all seeds — sweep may be incomplete.")
    print(seed_counts)
    print()

print("=== Mean final cumulative reward, by min_pulls ===")
reward_table = final.groupby(["min_pulls", "agent"])["cum_reward"].mean().unstack("agent")
print(reward_table.round(1))

print("\n=== Mean final cumulative regret, by min_pulls ===")
regret_table = final.groupby(["min_pulls", "agent"])["cum_regret"].mean().unstack("agent")
print(regret_table.round(1))

print("\n=== Std dev of final cumulative reward across seeds, by min_pulls ===")
std_table = final.groupby(["min_pulls", "agent"])["cum_reward"].std().unstack("agent")
print(std_table.round(1))
