"""View results written by run_addition.py / run_addition2.py.

    python view_addition.py --tag nmax9
    python view_addition.py --tag two_digit
    python view_addition.py --tag two_digit --no-plot

Reads results/addition/addition_results_<tag>.csv. Prints block accuracy
per agent, overall/last-block accuracy, final drift, and saves
results/addition/view_<tag>.png (needs matplotlib; skipped if missing).
"""
import argparse
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "..", "results", "addition")


def block_table(df, window):
    d = df.copy()
    d["block"] = (d["round"] // window) * window
    t = d.groupby(["agent", "block"])["correct"].mean().unstack("agent")
    last_r = int(df["round"].max())
    t.index = [f"{b}-{min(b + window - 1, last_r)}" for b in t.index]
    t.index.name = "rounds"
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="nmax9")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    csv_path = os.path.join(RESULTS_DIR, f"addition_results_{args.tag}.csv")
    df = pd.read_csv(csv_path)
    df["correct"] = df["correct"].astype(bool)
    agents = list(df["agent"].unique())
    n_rounds = df["round"].max() + 1
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    print(f"=== {args.tag}: agents={agents} rounds={n_rounds} ===\n")

    print(f"=== Accuracy by {args.window}-round block ===")
    print(block_table(df, args.window).to_string(), "\n")

    print("=== Overall / last-block summary ===")
    last_block_start = n_rounds - args.window
    summ = {}
    for ag in agents:
        g = df[df["agent"] == ag]
        overall = g["correct"].mean()
        last = g[g["round"] >= last_block_start]["correct"].mean()
        row = {"overall": overall, f"last{args.window}": last}
        if "drift_pct" in g and ag == "connectome":
            drift = pd.to_numeric(g["drift_pct"], errors="coerce").dropna()
            row["final_drift_%"] = drift.iloc[-1] if len(drift) else float("nan")
        summ[ag] = row
    print(pd.DataFrame(summ).T.to_string(), "\n")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("(matplotlib not installed: skipping plot)")
            return
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        bt = block_table(df, args.window)
        for ag in agents:
            ax[0].plot(range(len(bt)), bt[ag], marker="o", label=ag)
        ax[0].set_xticks(range(len(bt)))
        ax[0].set_xticklabels(bt.index, rotation=60, fontsize=6)
        ax[0].set(title=f"Accuracy by {args.window}-round block", ylim=(0, 1.02))
        ax[0].legend(fontsize=8)

        conn = df[df["agent"] == "connectome"].copy()
        conn["drift_pct"] = pd.to_numeric(conn["drift_pct"], errors="coerce")
        ax[1].plot(conn["round"], conn["drift_pct"])
        ax[1].set(title="Connectome weight drift", xlabel="round", ylabel="drift %")

        fig.tight_layout()
        out = os.path.join(RESULTS_DIR, f"view_{args.tag}.png")
        fig.savefig(out, dpi=130)
        print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
