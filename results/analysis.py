import os
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--source", choices=["headline", "sweep"], default="headline",
    help="headline = bandit_results.csv (single run). sweep = min_pulls_sweep.csv, filtered by --min-pulls."
)
parser.add_argument(
    "--min-pulls", type=int, default=10,
    help="Only used with --source sweep: which min_pulls setting to isolate."
)
parser.add_argument(
    "--every", type=int, default=20,
    help="Print every Nth trial in the curve tables."
)
parser.add_argument(
    "--window", type=int, default=20,
    help="Window size for the rolling reward rate."
)
args = parser.parse_args()

if args.source == "headline":
    df = pd.read_csv(os.path.join(SCRIPT_DIR, "bandit_results.csv"))
    label = "headline run (bandit_results.csv)"
else:
    df = pd.read_csv(os.path.join(SCRIPT_DIR, "min_pulls_sweep.csv"))
    df = df[df.min_pulls == args.min_pulls]
    label = f"sweep run, min_pulls={args.min_pulls}"
    if df.empty:
        raise SystemExit(f"No rows found for min_pulls={args.min_pulls} — check it was actually run.")

print(f"=== Source: {label} ===\n")

print("=== Mean cumulative regret over trials (lower / flatter = faster convergence) ===")
regret_curve = df.groupby(["agent", "trial"])["cum_regret"].mean().unstack("agent")
print(regret_curve.iloc[::args.every].round(2))

print(f"\n=== Rolling reward rate (mean reward over trailing {args.window} trials, per agent) ===")
df_sorted = df.sort_values(["agent", "seed", "trial"]).copy()
df_sorted["rolling_reward"] = (
    df_sorted.groupby(["agent", "seed"])["reward"]
    .transform(lambda s: s.rolling(args.window).mean())
)
rolling_curve = (
    df_sorted.groupby(["agent", "trial"])["rolling_reward"]
    .mean()
    .unstack("agent")
)
print(rolling_curve.iloc[::args.every].round(3))

print("\n=== Final summary ===")
final = df.loc[df.groupby(["agent", "seed"])["trial"].idxmax()]
summary = final.groupby("agent")[["cum_reward", "cum_regret"]].agg(["mean", "std"])
print(summary.round(2))
