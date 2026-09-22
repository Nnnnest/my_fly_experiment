"""View + aggregate results from run_gridworld_large_v2.py across one or
more --tag runs, plus the Tier-3 imitation summary.

    python view_gridworld_large_v2.py --tags small26_fair medium72_fair large113_fair
    python view_gridworld_large_v2.py --tags large113_opt large113_assisted --window 20
"""
import os, argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))            # results/analysis
ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))       # my_experiments
RESULTS_DIR = os.path.join(ROOT, "results", "gridworld")


def block_table(df, value, window, agg="mean"):
    d = df.copy()
    d["block"] = (d["episode"] // window) * window
    t = d.groupby(["agent", "seed", "block"])[value].agg(agg).groupby(["agent", "block"]).mean()
    t = t.unstack("agent")
    last_ep = int(df["episode"].max())
    t.index = [f"{b}-{min(b + window - 1, last_ep)}" for b in t.index]
    t.index.name = "episodes"
    return t


def load_tag(tag, results_dir):
    base = os.path.join(results_dir, f"gridworld_results_{tag}")
    if not os.path.exists(base + ".csv"):
        print(f"(skipping tag '{tag}': {base}.csv not found)")
        return None
    df = pd.read_csv(base + ".csv")
    df["reached_goal"] = df["reached_goal"].astype(bool)
    meta = pd.read_csv(base + "_meta.csv") if os.path.exists(base + "_meta.csv") else None
    pol = pd.read_csv(base + "_policy.csv") if os.path.exists(base + "_policy.csv") else None
    return df, meta, pol


def summarize_tag(tag, df, meta):
    succ = df.groupby("agent")["reached_goal"].mean().rename("overall_success")
    out = pd.DataFrame({"overall_success": succ})
    out.insert(0, "tag", tag)
    if meta is not None:
        m = meta.copy()
        m["collapsed"] = m["collapsed"].astype(str).str.lower() == "true"
        g = m.groupby("agent")
        out["peak20"] = g["peak20"].mean()
        out["final20"] = g["final20"].mean()
        out["collapsed"] = g["collapsed"].sum().astype(int).astype(str) + "/" + g["collapsed"].count().astype(str)
        out["collapsed"] = g["collapsed"].sum().astype(int).astype(str) + "/" + g["collapsed"].count().astype(str)
        out["failed(final<0.5)"] = ((m["final20"] < 0.5).groupby(m["agent"]).sum().astype(int).astype(str)
                                     + "/" + g["collapsed"].count().astype(str))
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True, help="one or more --tag values from run_gridworld_large_v2.py")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--detail-tag", default=None,
                    help="if set, also print the full per-block breakdown for this one tag")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    rows = []
    loaded = {}
    for tag in a.tags:
        res = load_tag(tag, a.results_dir)
        if res is None:
            continue
        df, meta, pol = res
        loaded[tag] = (df, meta, pol)
        rows.append(summarize_tag(tag, df, meta))

    if rows:
        print("=== Cross-run summary (one row per tag x agent) ===")
        print(pd.concat(rows, ignore_index=True).to_string(index=False), "\n")
    else:
        print(f"No tags found under {a.results_dir}")

    if a.detail_tag and a.detail_tag in loaded:
        df, meta, pol = loaded[a.detail_tag]
        W = a.window
        print(f"=== Detail for '{a.detail_tag}': success rate by {W}-episode block ===")
        print(block_table(df.assign(s=df["reached_goal"].astype(float)), "s", W).to_string(), "\n")
        if pol is not None and len(pol):
            print(f"=== Detail for '{a.detail_tag}': greedy-policy accuracy by block ===")
            print(block_table(pol, "policy_acc", W).to_string(), "\n")

    tier3_path = os.path.join(a.results_dir, "tier3_imitation_summary.csv")
    print("=== Tier 3 (imitation) -- full-coverage-style policy accuracy, not win rate; do not read row-for-row against Tiers 1/1-opt/2 ===")
    if os.path.exists(tier3_path):
        print(pd.read_csv(tier3_path).to_string(index=False))
    else:
        print(f"none found at {tier3_path} -- run train_imitation_gridworld.py first")


if __name__ == "__main__":
    main()
