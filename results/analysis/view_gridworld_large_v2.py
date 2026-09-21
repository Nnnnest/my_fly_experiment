"""View results written by run_gridworld_large_v2.py.

    python view_gridworld_v2.py                      # tag large_v2, results/ next to this file
    python view_gridworld_v2.py --tag large_v2 --window 20 --per-seed
    python view_gridworld_v2.py --no-plot

Reads results/gridworld_results_<tag>.csv (+ _meta.csv, _policy.csv when present).
Prints: run summary, success by episode block, policy accuracy by block, training volume,
and saves results/view_<tag>.png (needs matplotlib; skipped if missing).
"""
import os, argparse
import numpy as np
import pandas as pd

ROOT = os.path.join(SCRIPT_DIR, "..", "..")

def block_table(df, value, window, agg="mean"):
    """value averaged per (agent, block of `window` episodes), then across seeds."""
    d = df.copy()
    d["block"] = (d["episode"] // window) * window
    t = d.groupby(["agent", "seed", "block"])[value].agg(agg).groupby(["agent", "block"]).mean()
    t = t.unstack("agent")
    last_ep = int(df["episode"].max())
    t.index = [f"{b}-{min(b + window - 1, last_ep)}" for b in t.index]
    t.index.name = "episodes"
    return t


def rolling_by_episode(df, window):
    out = {}
    for (agent, seed), g in df.sort_values("episode").groupby(["agent", "seed"]):
        r = g["reached_goal"].astype(float).rolling(window, min_periods=window).mean()
        out[(agent, seed)] = pd.Series(r.values, index=g["episode"].values)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="large_v2")
    ap.add_argument("--results-dir", default=os.path.join(SCRIPT_DIR, ""))
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--per-seed", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    base = os.path.join(a.results_dir, f"gridworld_results_{a.tag}")
    df = pd.read_csv(base + ".csv")
    df["reached_goal"] = df["reached_goal"].astype(bool)
    meta = pd.read_csv(base + "_meta.csv") if os.path.exists(base + "_meta.csv") else None
    pol = pd.read_csv(base + "_policy.csv") if os.path.exists(base + "_policy.csv") else None
    W = a.window
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    agents = list(df["agent"].unique())
    seeds = [int(x) for x in sorted(df["seed"].unique())]
    print(f"=== {a.tag}: agents={agents} seeds={seeds} episodes/run={df.groupby(['agent','seed']).size().max()} ===\n")

    # 1. run summary (mean across seeds)
    if meta is not None:
        m = meta.copy()
        m["collapsed"] = m["collapsed"].astype(str).str.lower() == "true"
        g = m.groupby("agent")
        summ = pd.DataFrame({
            "peak20": g["peak20"].mean(), "final20": g["final20"].mean(),
            "collapsed": g["collapsed"].sum().astype(int).astype(str) + "/" + g["collapsed"].count().astype(str),
            "failed(final<0.5)": (m["final20"] < 0.5).groupby(m["agent"]).sum().astype(int).astype(str) + "/" + g["collapsed"].count().astype(str),
            "first>=90%": g["first_ep_ge90"].apply(lambda x: "-" if x.isna().all() else f"{x.dropna().median():.0f} ({x.notna().sum()}/{len(x)} seeds)"),
            "seconds": g["seconds"].mean(),
        })
        if "n_trained" in m:
            summ["trains"] = g["n_trained"].apply(lambda x: pd.to_numeric(x, errors="coerce").mean())
            summ["skipped"] = g["n_skipped"].apply(lambda x: pd.to_numeric(x, errors="coerce").mean())
        if "top10_share" in m:
            summ["top10_share"] = g["top10_share"].apply(lambda x: pd.to_numeric(x, errors="coerce").mean())
            summ["max_pair_trains"] = g["max_pair_trains"].apply(lambda x: pd.to_numeric(x, errors="coerce").mean())
        succ = df.groupby("agent")["reached_goal"].mean().rename("overall_success")
        steps = df[df["reached_goal"]].groupby("agent")["steps"].mean().rename("steps_if_success")
        print("=== Run summary (mean over seeds; collapsed = peak20>=0.5 and peak-final>0.3; failed = final20<0.5, includes runs that never learned) ===")
        print(summ.join(succ).join(steps).to_string(), "\n")

    # 2. success by block
    print(f"=== Success rate by {W}-episode block (mean over seeds) ===")
    print(block_table(df.assign(s=df["reached_goal"].astype(float)), "s", W).to_string(), "\n")

    # 3. policy accuracy + training volume
    if pol is not None and len(pol):
        print(f"=== Greedy-policy accuracy by block (fraction of states whose greedy action moves closer to goal) ===")
        print(block_table(pol, "policy_acc", W).to_string(), "\n")
        last = pol.sort_values("episode").groupby(["agent", "seed"]).tail(1)
        vol = last.groupby("agent")[["cum_trained", "cum_skipped"]].mean()
        vol["train_share"] = vol["cum_trained"] / (vol["cum_trained"] + vol["cum_skipped"])
        print("=== Training volume at the end (mean over seeds) ===")
        print(vol.to_string(), "\n")
        print(f"=== Trainings per {W}-episode block (mean over seeds) ===")
        p = pol.sort_values(["agent", "seed", "episode"]).copy()
        p["trained_now"] = p.groupby(["agent", "seed"])["cum_trained"].diff().fillna(p["cum_trained"])
        print(block_table(p, "trained_now", W, agg="sum").to_string(), "\n")

    if "optimal_steps" in df.columns:
        ok = df[df["reached_goal"] & (df["steps"] > 0)].copy()
        ok["eff"] = (ok["optimal_steps"] / ok["steps"]).clip(upper=1.0)
        print(f"=== Path efficiency = optimal steps / actual steps, successful episodes only (1.00 = shortest path) ===")
        print(block_table(ok, "eff", W).to_string(), "\n")

    if a.per_seed and meta is not None:
        print("=== Per seed ===")
        print(meta.to_string(index=False), "\n")

    # 4. plot
    if not a.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("(matplotlib not installed: skipping plot)")
            return
        panels = 3 if pol is not None and len(pol) else 1
        fig, ax = plt.subplots(1, panels, figsize=(5.5 * panels, 4), squeeze=False)
        ax = ax[0]
        roll = rolling_by_episode(df, W)
        colors = {ag: c for ag, c in zip(agents, plt.rcParams["axes.prop_cycle"].by_key()["color"] * 3)}
        for ag in agents:
            cols = [c for c in roll.columns if c[0] == ag]
            for c in cols:
                ax[0].plot(roll.index, roll[c], color=colors[ag], alpha=0.2, lw=0.8)
            ax[0].plot(roll.index, roll[cols].mean(axis=1), color=colors[ag], lw=2, label=ag)
        ax[0].set(title=f"Rolling success ({W} ep)", xlabel="episode", ylim=(-0.02, 1.02))
        ax[0].legend(fontsize=8)
        if panels == 3:
            for ag in pol["agent"].unique():
                g = pol[pol["agent"] == ag].groupby("episode")["policy_acc"].mean()
                ax[1].plot(g.index, g.values, color=colors[ag], lw=2, label=ag)
                c = pol[pol["agent"] == ag].groupby("episode")["cum_trained"].mean()
                ax[2].plot(c.index, c.values, color=colors[ag], lw=2, label=ag)
            ax[1].set(title="Greedy-policy accuracy", xlabel="episode", ylim=(0, 1.02))
            ax[2].set(title="Cumulative train() calls", xlabel="episode", yscale="symlog")
            ax[1].legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(a.results_dir, f"view_{a.tag}.png")
        fig.savefig(out, dpi=130)
        print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
