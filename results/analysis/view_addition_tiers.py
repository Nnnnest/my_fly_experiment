"""View results written by run_addition_experiment.py -- aggregates across
seeds (mean +/- std per checkpoint), so agent differences can be judged
against seed-to-seed spread instead of read off one noisy trajectory.

    python view_addition_tiers.py --tag all_tiers
    python view_addition_tiers.py --tag all_tiers --tier tier3 --digit-width two
"""
import argparse
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "addition")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="all_tiers")
    ap.add_argument("--tier", default=None, choices=[None, "tier1", "tier2", "tier3"])
    ap.add_argument("--digit-width", default=None, choices=[None, "single", "two"])
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(RESULTS_DIR, f"addition_results_{args.tag}.csv"))
    if args.tier:
        df = df[df["tier"] == args.tier]
    if args.digit_width:
        df = df[df["digit_width"] == args.digit_width]

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    n_seeds = df["seed"].nunique()
    print(f"seeds={sorted(df['seed'].unique())}\n")

    for (tier, dw), g in df.groupby(["tier", "digit_width"]):
        print(f"=== {tier} / {dw}-digit (n_seeds={g['seed'].nunique()}) ===")
        agg = g.groupby(["round", "agent"])["eval_acc"].agg(["mean", "std"])
        table = agg["mean"].unstack("agent")
        se_table = (agg["std"] / (n_seeds ** 0.5)).unstack("agent")  # standard error of the mean
        print("mean eval_acc by checkpoint:")
        print(table.to_string(), "\n")

        last_round = g["round"].max()
        final = g[g["round"] == last_round].groupby("agent")["eval_acc"].agg(["mean", "std"])
        final["se"] = final["std"] / (n_seeds ** 0.5)
        final["ci95"] = 1.96 * final["se"]
        print(f"final checkpoint (round={last_round}), mean +/- 95% CI across seeds:")
        print(final.to_string(), "\n")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("(matplotlib not installed: skipping plot)")
            return
        combos = list(df.groupby(["tier", "digit_width"]).groups.keys())
        fig, axes = plt.subplots(1, len(combos), figsize=(5 * len(combos), 4), squeeze=False)
        axes = axes[0]
        for ax, (tier, dw) in zip(axes, combos):
            g = df[(df["tier"] == tier) & (df["digit_width"] == dw)]
            for agent, gg in g.groupby("agent"):
                stats = gg.groupby("round")["eval_acc"].agg(["mean", "std"])
                se = stats["std"] / (n_seeds ** 0.5)
                ax.plot(stats.index, stats["mean"], label=agent, lw=2)
                ax.fill_between(stats.index, stats["mean"] - se, stats["mean"] + se, alpha=0.2)
            ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance")
            ax.set(title=f"{tier} / {dw}-digit", xlabel="round / examples trained",
                  ylim=(0, 1.02))
            ax.legend(fontsize=7)
        fig.tight_layout()
        out = os.path.join(RESULTS_DIR, f"view_{args.tag}.png")
        fig.savefig(out, dpi=130)
        print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
