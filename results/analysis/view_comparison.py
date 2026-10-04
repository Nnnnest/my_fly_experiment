"""View results written by scripts/comparison/run_comparison.py.

    python results/analysis/view_comparison.py
    python results/analysis/view_comparison.py --window 20

Reads results/comparison/comparison_results.csv. Beyond the usual
accuracy-by-block/drift views, breaks accuracy down by |a-b| (difficulty)
-- the key diagnostic for whether this is a genuine graded-magnitude
comparison (accuracy should rise with |a-b|) or something else (flat
accuracy regardless of gap would mean magnitude isn't really driving the
decision).
"""
import os
import argparse

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def block_table(df, value, window, agg="mean"):
    d = df.copy()
    d["block"] = (d["round"] // window) * window
    t = d.groupby(["agent", "block"])[value].agg(agg).unstack("agent")
    last_r = int(df["round"].max())
    t.index = [f"{b}-{min(b + window - 1, last_r)}" for b in t.index]
    t.index.name = "rounds"
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results", "comparison"))
    ap.add_argument("--plots-dir", default=os.path.join(ROOT, "results", "plots"))
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--filename", default="comparison_results.csv",
                    help="e.g. comparison_results_notrain.csv for the no-train diagnostic run")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(a.results_dir, a.filename))
    df["correct"] = df["correct"].astype(bool)
    W = a.window
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    agents = list(df["agent"].unique())
    n_rounds = int(df[df["round"] >= 0].groupby("agent").size().max())
    print(f"=== comparison: agents={agents} rounds={n_rounds} ===\n")

    baseline_df = df[df["round"] == -1]
    if len(baseline_df):
        print(f"=== Pre-training baseline (round=-1, n={len(baseline_df) // len(agents)} pairs, "
              f"ZERO train_on calls -- the genuine before-any-learning number) ===")
        print(baseline_df.groupby("agent")["correct"].mean().rename("baseline_accuracy").to_string(), "\n")
    else:
        print("(no round=-1 baseline rows in this file -- rerun with the updated "
              "run_comparison.py to get one)\n")

    df = df[df["round"] >= 0]  # exclude baseline rows from everything below

    overall = df.groupby("agent")["correct"].mean().rename("overall_accuracy")
    last = df[df["round"] >= df["round"].max() - W + 1]
    recent = last.groupby("agent")["correct"].mean().rename(f"last_{W}_accuracy")
    print("=== Overall vs. most-recent accuracy ===")
    print(pd.concat([overall, recent], axis=1).to_string(), "\n")

    print(f"=== Accuracy by {W}-round block ===")
    print(block_table(df.assign(c=df["correct"].astype(float)), "c", W).to_string(), "\n")

    drift_df = df[df["agent"] == "connectome"].dropna(subset=["drift_pct"])
    if len(drift_df):
        print(f"=== Connectome weight drift (%) by {W}-round block ===")
        print(block_table(drift_df, "drift_pct", W).rename(
            columns={"connectome": "drift_pct"}).to_string(), "\n")

    print("=== Accuracy by |a-b| (difficulty), per agent -- should RISE with the gap "
          "if magnitude genuinely drives the decision ===")
    diff_acc = df.assign(c=df["correct"].astype(float)).pivot_table(
        index="abs_diff", columns="agent", values="c", aggfunc="mean")
    diff_n = df.pivot_table(index="abs_diff", columns="agent", values="correct", aggfunc="count")
    print(diff_acc.to_string(), "\n")
    print("(n per cell:)")
    print(diff_n.to_string(), "\n")

    if not a.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("(matplotlib not installed: skipping plot)")
            return
        os.makedirs(a.plots_dir, exist_ok=True)
        colors = {ag: c for ag, c in zip(agents, plt.rcParams["axes.prop_cycle"].by_key()["color"] * 3)}
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

        for ag in agents:
            g = df[df["agent"] == ag].sort_values("round")
            roll = g["correct"].astype(float).rolling(W, min_periods=W).mean()
            ax[0].plot(g["round"], roll, color=colors[ag], lw=2, label=ag)
        ax[0].axhline(0.5, color="gray", ls="--", lw=1, label="chance")
        ax[0].set(title=f"Rolling accuracy ({W} rounds)", xlabel="round", ylim=(-0.02, 1.02))
        ax[0].legend(fontsize=8)

        for ag in agents:
            g = diff_acc[ag].dropna()
            ax[1].plot(g.index, g.values, marker="o", color=colors[ag], lw=2, label=ag)
        ax[1].axhline(0.5, color="gray", ls="--", lw=1)
        ax[1].set(title="Accuracy vs. |a-b|", xlabel="|a-b|", ylabel="accuracy", ylim=(-0.02, 1.02))
        ax[1].legend(fontsize=8)

        fig.tight_layout()
        out = os.path.join(a.plots_dir, f"view_comparison_{os.path.splitext(a.filename)[0]}.png")
        fig.savefig(out, dpi=130)
        print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
