"""View results written by scripts/shapes/run_shapes.py.

    python results/analysis/view_shapes.py                     # defaults, results/shapes/ next to this file's tree
    python results/analysis/view_shapes.py --window 20 --per-agent-confusion
    python results/analysis/view_shapes.py --no-plot

Reads results/shapes/shapes_results.csv (columns: round, agent, label,
predicted, correct, drift_pct -- see run_shapes.py). Prints: overall
accuracy per agent, accuracy by round-block (same rolling-block
convention as view_gridworld_large_v2.py / view_tictactoe.py), a
per-digit confusion matrix per agent, and the connectome's drift_pct
curve. Saves results/plots/view_shapes.png (needs matplotlib; skipped if
missing).
"""
import os
import argparse

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # my_experiments/


def block_table(df, value, window, agg="mean"):
    """value averaged per (agent, block of `window` rounds)."""
    d = df.copy()
    d["block"] = (d["round"] // window) * window
    t = d.groupby(["agent", "block"])[value].agg(agg).unstack("agent")
    last_r = int(df["round"].max())
    t.index = [f"{b}-{min(b + window - 1, last_r)}" for b in t.index]
    t.index.name = "rounds"
    return t


def confusion_matrix(df, agent, n_digits):
    g = df[df["agent"] == agent]
    cm = np.zeros((n_digits, n_digits), dtype=int)
    for label, pred in zip(g["label"], g["predicted"]):
        cm[int(label), int(pred)] += 1
    cols = pd.MultiIndex.from_product([["predicted"], range(n_digits)])
    idx = pd.Index(range(n_digits), name="actual")
    return pd.DataFrame(cm, index=idx, columns=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results", "shapes"))
    ap.add_argument("--plots-dir", default=os.path.join(ROOT, "results", "plots"))
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--n-digits", type=int, default=5)
    ap.add_argument("--per-agent-confusion", action="store_true",
                    help="print a full confusion matrix per agent (default: connectome only)")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    csv_path = os.path.join(a.results_dir, "shapes_results.csv")
    df = pd.read_csv(csv_path)
    df["correct"] = df["correct"].astype(bool)
    W = a.window
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    agents = list(df["agent"].unique())
    n_rounds = df.groupby("agent").size().max()
    print(f"=== shapes: agents={agents} rounds={n_rounds} n_digits={a.n_digits} ===\n")

    # 1. overall accuracy
    overall = df.groupby("agent")["correct"].mean().rename("overall_accuracy")
    last_block = df[df["round"] >= df["round"].max() - W + 1]
    recent = last_block.groupby("agent")["correct"].mean().rename(f"last_{W}_accuracy")
    print("=== Overall vs. most-recent accuracy ===")
    print(pd.concat([overall, recent], axis=1).to_string(), "\n")

    # 2. accuracy by block
    print(f"=== Accuracy by {W}-round block ===")
    print(block_table(df.assign(c=df["correct"].astype(float)), "c", W).to_string(), "\n")

    # 3. drift (connectome only -- other agents have no drift_pct)
    drift_df = df[df["agent"] == "connectome"].dropna(subset=["drift_pct"])
    if len(drift_df):
        print(f"=== Connectome weight drift (%) by {W}-round block ===")
        print(block_table(drift_df, "drift_pct", W).rename(
            columns={"connectome": "drift_pct"}).to_string(), "\n")

    # 4. confusion matrices
    conf_agents = agents if a.per_agent_confusion else [ag for ag in agents if ag == "connectome"]
    for ag in conf_agents:
        print(f"=== Confusion matrix: {ag} (rows=actual digit, cols=predicted digit) ===")
        print(confusion_matrix(df, ag, a.n_digits).to_string(), "\n")

    # 5. plot
    if not a.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("(matplotlib not installed: skipping plot)")
            return
        os.makedirs(a.plots_dir, exist_ok=True)
        has_drift = len(drift_df) > 0
        panels = 2 if has_drift else 1
        fig, ax = plt.subplots(1, panels, figsize=(5.5 * panels, 4), squeeze=False)
        ax = ax[0]
        colors = {ag: c for ag, c in zip(agents, plt.rcParams["axes.prop_cycle"].by_key()["color"] * 3)}
        for ag in agents:
            g = df[df["agent"] == ag].sort_values("round")
            roll = g["correct"].astype(float).rolling(W, min_periods=W).mean()
            ax[0].plot(g["round"], roll, color=colors[ag], lw=2, label=ag)
        ax[0].axhline(1.0 / a.n_digits, color="gray", ls="--", lw=1, label="chance")
        ax[0].set(title=f"Rolling accuracy ({W} rounds)", xlabel="round", ylim=(-0.02, 1.02))
        ax[0].legend(fontsize=8)
        if has_drift:
            g = drift_df.sort_values("round")
            ax[1].plot(g["round"], g["drift_pct"], color=colors.get("connectome", "C0"), lw=2)
            ax[1].set(title="Connectome weight drift", xlabel="round", ylabel="drift %")
        fig.tight_layout()
        out = os.path.join(a.plots_dir, "view_shapes.png")
        fig.savefig(out, dpi=130)
        print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
