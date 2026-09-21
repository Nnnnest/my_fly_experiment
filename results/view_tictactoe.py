"""Summarize and plot tic-tac-toe experiment results.

Usage: python view_tictactoe.py
Reads every results/tictactoe_*.csv written by run_tictactoe.py, prints a
win/loss/draw summary table (also saved as tictactoe_summary.csv), and
saves a rolling-win-rate plot per file (<name>_rolling.png).
"""
import glob
import os

import pandas as pd
import matplotlib.pyplot as plt

ROLLING_WINDOW = 50
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def summarize(path):
    df = pd.read_csv(path)
    outcome_col = "outcome" if "outcome" in df.columns else "outcome_a1"
    n = len(df)
    wins = (df[outcome_col] == 1).sum()
    losses = (df[outcome_col] == -1).sum()
    draws = (df[outcome_col] == 0).sum()
    last_100 = df[outcome_col].tail(100)
    return {
        "file": os.path.basename(path),
        "episodes": n,
        "win_pct": round(100 * wins / n, 1),
        "loss_pct": round(100 * losses / n, 1),
        "draw_pct": round(100 * draws / n, 1),
        "win_pct_last100": round(100 * (last_100 == 1).mean(), 1),
    }


def plot_rolling(path):
    df = pd.read_csv(path)
    outcome_col = "outcome" if "outcome" in df.columns else "outcome_a1"
    rolling = (df[outcome_col] == 1).rolling(ROLLING_WINDOW).mean()
    plt.figure()
    plt.plot(df["episode"], rolling)
    plt.xlabel("episode")
    plt.ylabel(f"win rate (rolling {ROLLING_WINDOW})")
    plt.title(os.path.basename(path))
    plt.ylim(0, 1)
    out_path = path.replace(".csv", "_rolling.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_drift(path):
    """Only produced for connectome runs (drift_pct / drift_pct_a1
    column present) -- weight drift over the run, to check whether a
    declining rolling win rate tracks cumulative weight drift."""
    df = pd.read_csv(path)
    drift_col = "drift_pct" if "drift_pct" in df.columns else (
        "drift_pct_a1" if "drift_pct_a1" in df.columns else None)
    if drift_col is None:
        return None
    plt.figure()
    plt.plot(df["episode"], df[drift_col])
    plt.xlabel("episode")
    plt.ylabel("weight drift (%)")
    plt.title(os.path.basename(path))
    out_path = path.replace(".csv", "_drift.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def main():
    csv_paths = sorted(
        p for p in glob.glob(os.path.join(RESULTS_DIR, "tictactoe_*.csv"))
        if os.path.basename(p) != "tictactoe_summary.csv"
    )
    if not csv_paths:
        print(f"no tictactoe_*.csv files found in {RESULTS_DIR} -- run run_tictactoe.py first")
        return

    rows = [summarize(p) for p in csv_paths]
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(RESULTS_DIR, "tictactoe_summary.csv"), index=False)

    for p in csv_paths:
        out = plot_rolling(p)
        print(f"saved {out}")
        drift_out = plot_drift(p)
        if drift_out:
            print(f"saved {drift_out}")


if __name__ == "__main__":
    main()
