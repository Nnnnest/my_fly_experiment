import os
import pandas as pd

ROOT = os.path.join(SCRIPT_DIR, "..", "..")
CSV_PATH = os.path.join(ROOT, "results", "gridworld", "gridworld_results.csv")

df = pd.read_csv(CSV_PATH)

# completeness check
episode_counts = df.groupby(["agent", "seed"])["episode"].count().unstack("agent")
print("=== Episodes logged per (agent, seed) ===")
print(episode_counts)
print()

print("=== Overall success rate (fraction of episodes reaching goal) ===")
print(df.groupby("agent")["reached_goal"].mean().round(3))

print("\n=== Mean steps-to-goal, SUCCESSFUL episodes only (timeouts excluded) ===")
success_only = df[df.reached_goal]
print(success_only.groupby("agent")["steps"].agg(["mean", "std", "count"]).round(1))

print("\n=== Rolling success rate over training (window=10 episodes, mean across seeds) ===")
df_sorted = df.sort_values(["agent", "seed", "episode"]).copy()
df_sorted["rolling_success"] = (
    df_sorted.groupby(["agent", "seed"])["reached_goal"]
    .transform(lambda s: s.rolling(10, min_periods=1).mean())
)
rolling = df_sorted.groupby(["agent", "episode"])["rolling_success"].mean().unstack("agent")
print(rolling.iloc[::10].round(3))

print("\n=== Episode of first sustained success (rolling_success >= 0.9, first time it holds) ===")
for agent_name, g in df_sorted.groupby("agent"):
    per_seed = {}
    for seed, gs in g.groupby("seed"):
        hit = gs[gs["rolling_success"] >= 0.9]
        per_seed[seed] = int(hit["episode"].iloc[0]) if len(hit) else None
    print(f"  {agent_name}: {per_seed}")
