import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "gridworld_baselines.csv")

df = pd.read_csv(CSV_PATH)

# completeness check
episode_counts = df.groupby(["agent", "seed"])["episode"].count().unstack("agent")
print("=== Episodes logged per (agent, seed) — should all match N_EPISODES ===")
print(episode_counts)
print()

print("=== Success rate (fraction of episodes that reached the goal), by agent ===")
success_rate = df.groupby("agent")["reached_goal"].mean()
print(success_rate.round(3))

print("\n=== Mean steps-to-goal, first 20 vs last 20 episodes (per seed, averaged) ===")
def early_late_steps(g):
    g = g.sort_values("episode")
    early = g.head(20)["steps"].mean()
    late = g.tail(20)["steps"].mean()
    return pd.Series({"first_20_avg_steps": early, "last_20_avg_steps": late})

early_late = df.groupby(["agent", "seed"]).apply(early_late_steps).groupby("agent").mean()
print(early_late.round(1))

print("\n=== Steps-to-goal over training, every 20th episode (mean across seeds) ===")
steps_curve = df.groupby(["agent", "episode"])["steps"].mean().unstack("agent")
print(steps_curve.iloc[::20].round(1))

print("\n=== Total reward over training, every 20th episode (mean across seeds) ===")
reward_curve = df.groupby(["agent", "episode"])["total_reward"].mean().unstack("agent")
print(reward_curve.iloc[::20].round(1))
