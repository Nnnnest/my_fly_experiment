"""Three-tier view for the counting experiment (Tier 1: run_counting.py,
Tier 2: run_counting_tier2.py, Tier 3: train_imitation_counting.py) --
same combined-report idea as view_tictactoe_three_tier.py.

    python results/analysis/view_counting.py
    python results/analysis/view_counting.py --window 20 --no-plot

Each tier's file has a different shape (Tier 1: per-round x per-agent;
Tier 2: per-round, connectome only, with passes_used; Tier 3: periodic
held-out eval, not per-round), so each gets its own section rather than
being forced into one shared table. Missing tier files are reported and
skipped, not treated as an error -- run whichever tiers you have so far.
"""
import os
import glob
import re
import argparse

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # my_experiments/


def block_table(df, value, window, agg="mean", group_col="agent"):
    d = df.copy()
    d["block"] = (d["round"] // window) * window
    t = d.groupby([group_col, "block"])[value].agg(agg).unstack(group_col)
    last_r = int(df["round"].max())
    t.index = [f"{b}-{min(b + window - 1, last_r)}" for b in t.index]
    t.index.name = "rounds"
    return t


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def view_tier1(path, window, n_max):
    if not os.path.exists(path):
        print(f"[Tier 1] not found: {path} (skipping)")
        return None
    df = pd.read_csv(path)
    df["correct"] = df["correct"].astype(bool)
    agents = list(df["agent"].unique())

    section("TIER 1: online, one random example per round")
    overall_acc = df.groupby("agent")["correct"].mean().rename("overall_accuracy")
    overall_mae = df.groupby("agent")["abs_err"].mean().rename("overall_MAE")
    last = df[df["round"] >= df["round"].max() - window + 1]
    recent_acc = last.groupby("agent")["correct"].mean().rename(f"last_{window}_accuracy")
    recent_mae = last.groupby("agent")["abs_err"].mean().rename(f"last_{window}_MAE")
    print(pd.concat([overall_acc, recent_acc, overall_mae, recent_mae], axis=1).to_string())

    if "output_drift_pct" in df.columns:
        od = df[df["agent"] == "connectome"].dropna(subset=["output_drift_pct"])
        if len(od) and od["output_drift_pct"].notna().any():
            print(f"\nconnectome output_drift_pct (its own ~512 edges, NOT the "
                  f"whole-brain-diluted drift_pct): final = "
                  f"{od['output_drift_pct'].iloc[-1]:.4f}%")

    print(f"\nAccuracy by {window}-round block:")
    print(block_table(df.assign(c=df["correct"].astype(float)), "c", window).to_string())
    print(f"\nMean absolute error by {window}-round block:")
    print(block_table(df, "abs_err", window).to_string())

    return {"df": df, "agents": agents,
            "connectome_last_acc": float(recent_acc.get("connectome", float("nan"))),
            "connectome_last_mae": float(recent_mae.get("connectome", float("nan")))}


def view_tier2(path, window):
    if not os.path.exists(path):
        print(f"\n[Tier 2] not found: {path} (skipping)")
        return None
    df = pd.read_csv(path)
    df["correct"] = df["correct"].astype(bool)
    df["agent"] = "connectome_v2"  # so block_table's groupby works unchanged

    section("TIER 2: error-adaptive ('enforced') teaching, connectome only")
    overall_acc = df["correct"].mean()
    overall_mae = df["abs_err"].mean()
    last = df[df["round"] >= df["round"].max() - window + 1]
    recent_acc = last["correct"].mean()
    recent_mae = last["abs_err"].mean()
    mean_passes = df["passes_used"].mean()
    recent_passes = last["passes_used"].mean()
    print(f"overall_accuracy={overall_acc:.3f}  last_{window}_accuracy={recent_acc:.3f}")
    print(f"overall_MAE={overall_mae:.3f}  last_{window}_MAE={recent_mae:.3f}")
    print(f"mean extra-training passes/round: overall={mean_passes:.2f}  "
          f"last_{window}={recent_passes:.2f}  "
          f"(0 passes = prediction was already within tol before any extra training)")

    print(f"\nAccuracy by {window}-round block:")
    print(block_table(df.assign(c=df["correct"].astype(float)), "c", window).to_string())
    print(f"\nMean absolute error by {window}-round block:")
    print(block_table(df, "abs_err", window).to_string())
    print(f"\nMean passes_used by {window}-round block:")
    print(block_table(df, "passes_used", window).to_string())

    return {"df": df, "last_acc": recent_acc, "last_mae": recent_mae}


def view_tier3(results_dir):
    pattern = os.path.join(results_dir, "imitation_counting_results_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"\n[Tier 3] no files matching {pattern} (skipping)")
        return None

    section("TIER 3: imitation-style dense supervision (held-out eval)")
    runs = []
    for fp in files:
        m = re.search(r"shuffle(\w+?)_sleep(\w+)\.csv$", os.path.basename(fp))
        tag = m.group(0)[:-4] if m else os.path.basename(fp)
        df = pd.read_csv(fp)
        pre = df[df["step"] == -1].iloc[0]
        final = df[df["step"] != -1].iloc[-1] if (df["step"] != -1).any() else pre
        print(f"\n{tag}")
        print(f"  untrained baseline: accuracy={pre['held_out_accuracy']:.3f} "
              f"MAE={pre['held_out_mae']:.3f}")
        print(f"  final:              accuracy={final['held_out_accuracy']:.3f} "
              f"MAE={final['held_out_mae']:.3f}  "
              f"output_drift={final.get('output_drift_pct', float('nan')):.2f}%")
        runs.append({"tag": tag, "final_acc": final["held_out_accuracy"],
                     "final_mae": final["held_out_mae"]})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results", "counting"))
    ap.add_argument("--plots-dir", default=os.path.join(ROOT, "results", "plots"))
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--n-max", type=int, default=4)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    t1 = view_tier1(os.path.join(a.results_dir, "counting_results.csv"), a.window, a.n_max)
    t2 = view_tier2(os.path.join(a.results_dir, "counting_results_tier2.csv"), a.window)
    t3 = view_tier3(a.results_dir)

    section("SUMMARY: connectome across tiers")
    rows = []
    if t1 is not None:
        rows.append(("Tier 1 (online)", t1["connectome_last_acc"], t1["connectome_last_mae"]))
    if t2 is not None:
        rows.append(("Tier 2 (enforced)", t2["last_acc"], t2["last_mae"]))
    if t3 is not None:
        for r in t3:
            rows.append((f"Tier 3 imitation ({r['tag']})", r["final_acc"], r["final_mae"]))
    if rows:
        summary = pd.DataFrame(rows, columns=["tier", "accuracy", "MAE"]).set_index("tier")
        print(summary.to_string())
        if t1 is not None:
            for ag in t1["agents"]:
                if ag == "connectome":
                    continue
                last = t1["df"][(t1["df"]["agent"] == ag) &
                                (t1["df"]["round"] >= t1["df"]["round"].max() - a.window + 1)]
                print(f"(for context, Tier 1 {ag}: accuracy={last['correct'].mean():.3f} "
                      f"MAE={last['abs_err'].mean():.3f})")
    else:
        print("no tier results found")

    if not a.no_plot and t1 is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n(matplotlib not installed: skipping plot)")
            return
        os.makedirs(a.plots_dir, exist_ok=True)
        df1 = t1["df"]
        agents = t1["agents"]
        colors = {ag: c for ag, c in zip(agents, plt.rcParams["axes.prop_cycle"].by_key()["color"] * 3)}
        colors.setdefault("connectome_v2", "black")

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

        for ag in agents:
            g = df1[df1["agent"] == ag].sort_values("round")
            roll = g["correct"].astype(float).rolling(a.window, min_periods=a.window).mean()
            ax[0].plot(g["round"], roll, color=colors[ag], lw=2, label=f"T1 {ag}")
        if t2 is not None:
            g = t2["df"].sort_values("round")
            roll = g["correct"].astype(float).rolling(a.window, min_periods=a.window).mean()
            ax[0].plot(g["round"], roll, color=colors["connectome_v2"], lw=2, ls="--", label="T2 connectome")
        ax[0].axhline(1.0 / (a.n_max + 1), color="gray", ls=":", lw=1, label="chance")
        ax[0].set(title=f"Rolling accuracy ({a.window} rounds)", xlabel="round", ylim=(-0.02, 1.02))
        ax[0].legend(fontsize=7)

        for ag in agents:
            g = df1[df1["agent"] == ag].sort_values("round")
            roll = g["abs_err"].astype(float).rolling(a.window, min_periods=a.window).mean()
            ax[1].plot(g["round"], roll, color=colors[ag], lw=2, label=f"T1 {ag}")
        if t2 is not None:
            g = t2["df"].sort_values("round")
            roll = g["abs_err"].astype(float).rolling(a.window, min_periods=a.window).mean()
            ax[1].plot(g["round"], roll, color=colors["connectome_v2"], lw=2, ls="--", label="T2 connectome")
        ax[1].set(title=f"Rolling MAE ({a.window} rounds)", xlabel="round", ylabel="MAE")
        ax[1].legend(fontsize=7)

        jitter = np.random.default_rng(0)
        for ag in agents:
            g = df1[df1["agent"] == ag]
            xj = g["count"] + jitter.uniform(-0.15, 0.15, len(g))
            ax[2].scatter(xj, g["predicted"], s=8, alpha=0.3, color=colors[ag], label=f"T1 {ag}")
        lims = (-0.5, a.n_max + 0.5)
        ax[2].plot(lims, lims, color="gray", ls="--", lw=1)
        ax[2].set(title="Tier 1: predicted vs. actual count", xlabel="true count",
                  ylabel="predicted count", xlim=lims, ylim=lims)
        ax[2].legend(fontsize=7)

        fig.tight_layout()
        out = os.path.join(a.plots_dir, "view_counting.png")
        fig.savefig(out, dpi=130)
        print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
