import os
import pandas as pd

ROOT = os.path.join(SCRIPT_DIR, "..", "..")
CSV_PATH = os.path.join(ROOT, "results", "gridworld", "gridworld_results_multimaze.csv")

df = pd.read_csv(CSV_PATH)

print("=== Episodes logged per (maze, agent, seed) ===")
counts = df.groupby(["maze", "agent", "seed"])["episode"].count().unstack("agent")
print(counts)
print()

print("=== Overall success rate, by maze and agent ===")
print(df.groupby(["maze", "agent"])["reached_goal"].mean().unstack("agent").round(3))

print("\n=== Mean steps-to-goal, successful episodes only, by maze and agent ===")
success_only = df[df.reached_goal]
steps_summary = success_only.groupby(["maze", "agent"])["steps"].agg(["mean", "std", "count"])
print(steps_summary.round(1))

print("\n=== Episode of first sustained success (rolling_success >= 0.9), by maze ===")
df_sorted = df.sort_values(["maze", "agent", "seed", "episode"]).copy()
df_sorted["rolling_success"] = (
    df_sorted.groupby(["maze", "agent", "seed"])["reached_goal"]
    .transform(lambda s: s.rolling(10, min_periods=1).mean())
)
for maze_name, g in df_sorted.groupby("maze"):
    print(f"\n  -- {maze_name} --")
    for agent_name, ga in g.groupby("agent"):
        per_seed = {}
        for seed, gs in ga.groupby("seed"):
            hit = gs[gs["rolling_success"] >= 0.9]
            per_seed[seed] = int(hit["episode"].iloc[0]) if len(hit) else None
        print(f"    {agent_name}: {per_seed}")

print("\n=== Rolling success rate over training, by maze (every 20th episode, mean across seeds) ===")
for maze_name, g in df_sorted.groupby("maze"):
    print(f"\n  -- {maze_name} --")
    rolling = g.groupby(["agent", "episode"])["rolling_success"].mean().unstack("agent")
    print(rolling.iloc[::20].round(3))
