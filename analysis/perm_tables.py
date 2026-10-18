"""python -m analysis.perm_tables [--tuning]"""
import os
import sys

from core import paths
from core.load import load_summaries
from analysis.common import *
from analysis.validate import read_manifests


def main():
    if "--tuning" in sys.argv:
        d = load_summaries("bandit_perm", root=os.path.join(paths.RUNS_DIR, "_tuning"))
        d = d[d.status == "ok"]
        print(d.groupby("tier").cum_pseudo_regret.mean().round(2))
        return
    m = read_manifests("bandit_perm")
    print("status:", m.status.value_counts().to_dict(), "| commits:", m.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m.repo_dirty.fillna(False).sum()))
    d = load("bandit_perm")
    d["k"] = d.task.str.startswith("k2").map({True: 2, False: 4})
    d["pick_best"] = (d.top_arm == d.best_arm).astype(float)
    d["pick_last_index"] = (d.top_arm == d.k - 1).astype(float)
    d["best_is_last_index"] = (d.best_arm == d.k - 1).astype(float)
    c = cells(d, ["task", "label"], ["cum_pseudo_regret", "frac_optimal_last50", "pick_best", "pick_last_index"], 2)
    for met in ("cum_pseudo_regret", "frac_optimal_last50", "pick_best", "pick_last_index"):
        save_table(wide(c, "task", met), f"bandit_perm_{met}", f"Bandit, arm order controlled: {met}")
    comps = [("connectome [block] - qlearning", ("connectome", "block"), ("qlearning", "-")),
             ("connectome [balanced] - qlearning", ("connectome", "balanced"), ("qlearning", "-")),
             ("connectome [block] - mlp", ("connectome", "block"), ("mlp", "-")),
             ("connectome [balanced] - mlp", ("connectome", "balanced"), ("mlp", "-")),
             ("connectome [block] - connectome [balanced]", ("connectome", "block"), ("connectome", "balanced"))]
    save_table(paired_table(d, "cum_pseudo_regret", comps), "bandit_perm_paired",
               "Bandit, arm order controlled: paired differences in cumulative pseudo-regret", 3)
    print("\nseeds (of 10) where the best arm is the last index:")
    print(d[d.agent == "qlearning"].groupby("task").best_is_last_index.sum().astype(int).to_string())


if __name__ == "__main__":
    main()
