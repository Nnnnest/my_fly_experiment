"""python -m analysis.extra_tables"""
import os

from core import paths
from core.load import load_summaries
from analysis.common import *
from analysis.validate import read_manifests


def bandit_multi():
    m = read_manifests("bandit_multi")
    print("status:", m.status.value_counts().to_dict(), "| commits:", m.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m.repo_dirty.fillna(False).sum()))
    d = load("bandit_multi")
    c = cells(d, ["task", "label"], ["total_reward", "cum_pseudo_regret", "frac_optimal_last50"], 2)
    for met in ("cum_pseudo_regret", "frac_optimal_last50", "total_reward"):
        save_table(wide(c, "task", met), f"bandit_multi_{met}", f"4-armed bandit: {met} by min_pulls (mp)")
    comps = [("connectome - qlearning", ("connectome", "-"), ("qlearning", "-")),
             ("connectome - mlp", ("connectome", "-"), ("mlp", "-"))]
    save_table(paired_table(d, "cum_pseudo_regret", comps), "bandit_multi_paired",
               "4-armed bandit: paired differences in cumulative pseudo-regret (negative = connectome better)", 3)


def tuning():
    d = load_summaries("bandit_multi", root=os.path.join(paths.RUNS_DIR, "_tuning"))
    d = d[d.status == "ok"].copy()
    d["problem"] = d.task.str.split("_mp").str[0]
    print(d.groupby(["tier", "problem"]).cum_pseudo_regret.mean().unstack().round(2))


if __name__ == "__main__":
    import sys
    tuning() if "--tuning" in sys.argv else bandit_multi()
