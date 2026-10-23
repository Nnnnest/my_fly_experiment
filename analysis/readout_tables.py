"""python -m analysis.readout_tables --tuning     # mean accuracy of the delta rule per alpha (tuning seeds)
python -m analysis.readout_tables                 # final tables (seeds 0-9)"""
import os
import sys

import pandas as pd

from core import paths
from core.load import load_summaries
from analysis.common import load, cells, save_table, boot_ci
from analysis.validate import read_manifests

COLS = ["acc_untrained", "acc_native", "acc_delta", "acc_ridge_kc_1x", "acc_ridge_kc_5x",
        "acc_ridge_raw_1x", "acc_ridge_raw_5x"]
COMPS = [("native - delta (KC code)", "acc_native", "acc_delta"),
         ("native - ridge_kc_1x", "acc_native", "acc_ridge_kc_1x"),
         ("ridge_kc_5x - native", "acc_ridge_kc_5x", "acc_native"),
         ("ridge_kc_5x - ridge_kc_1x", "acc_ridge_kc_5x", "acc_ridge_kc_1x"),
         ("ridge_kc_1x - ridge_raw_1x (what the KC expansion adds)", "acc_ridge_kc_1x", "acc_ridge_raw_1x"),
         ("ridge_kc_5x - ridge_raw_5x (what the KC expansion adds)", "acc_ridge_kc_5x", "acc_ridge_raw_5x")]


def tuning():
    d = load_summaries("readouts", root=os.path.join(paths.RUNS_DIR, "_tuning"))
    d = d[d.status == "ok"]
    cols = sorted(c for c in d.columns if c.startswith("acc_delta_a"))
    print(d.groupby("task")[cols].mean().round(3).to_string())
    print("\nmean over tasks (pick the best, set ALPHA_DELTA in scripts/readouts/run_readouts.py, commit, run the final batch):")
    print(d[cols].mean().round(4).to_string())


def main():
    m = read_manifests("readouts")
    print("status:", m.status.value_counts().to_dict(), "| commits:", m.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m.repo_dirty.fillna(False).sum()))
    d = load("readouts")
    print("alpha_delta used:", sorted(d.alpha_delta.unique()))
    save_table(cells(d, ["task"], COLS, 3).drop(columns="n"), "readouts_acc",
               "Held-out accuracy on identical KC codes: native plasticity, delta rule and ridge readouts (mean [95% CI], 10 seeds)")
    rows = []
    for task, g in d.groupby("task", sort=False):
        for name, a, b in COMPS:
            diff = (g[a] - g[b]).dropna().values
            mean, lo, hi = boot_ci(diff)
            rows.append({"comparison": name, "task": task, "n": len(diff), "mean_diff": mean,
                         "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)})
    save_table(pd.DataFrame(rows), "readouts_paired", "Paired differences in held-out accuracy (matched on task, seed)", 2)


if __name__ == "__main__":
    tuning() if "--tuning" in sys.argv else main()
