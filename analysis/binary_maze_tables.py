"""python -m analysis.binary_maze_tables --tuning     (choose ALPHA_Q_BIN / LR_MLP_BIN)
   python -m analysis.binary_maze_tables              (final tables)"""
import os
import sys

import pandas as pd

from core import paths
from core.load import load_summaries
from analysis.common import load, cells, wide, paired_table, save_table
from analysis.validate import read_manifests


def tuning():
    d = load_summaries("gridworld_binary", root=os.path.join(paths.RUNS_DIR, "_tuning"))
    d = d[d.status == "ok"]
    print(d.groupby(["agent", "tier"]).overall_success.agg(["mean", "count"]).round(4))
    print(d.groupby(["agent", "tier", "task"]).overall_success.mean().unstack().round(4))


def final():
    m = read_manifests("gridworld_binary")
    print("status:", m.status.value_counts().to_dict(), "| commits:", m.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m.repo_dirty.fillna(False).sum()))
    b = load("gridworld_binary")
    s = load("gridworld")
    s = s[s.label.isin(["qlearning", "mlp", "connectome [gate_error]", "connectome [gate_rank]"])]
    allv = pd.concat([b, s], ignore_index=True)
    c = cells(allv, ["task", "label"], ["overall_success", "final20", "final_policy_acc"], 3)
    for met, cap in (("overall_success", "success over the whole run"), ("final20", "success in the last 20 episodes"),
                     ("final_policy_acc", "greedy-policy accuracy over all states")):
        save_table(wide(c, "task", met), f"gridworld_binary_{met}",
                   f"Maze, baselines with shaped vs binary feedback: {cap} (qlearning/mlp = shaped as in the main experiment)")
    comps = [("qlearning [binary] - qlearning [shaped]", ("qlearning", "binary"), ("qlearning", "fair")),
             ("mlp [binary] - mlp [shaped]", ("mlp", "binary"), ("mlp", "fair")),
             ("connectome [gate_error] - qlearning [binary]", ("connectome", "gate_error"), ("qlearning", "binary")),
             ("connectome [gate_rank] - qlearning [binary]", ("connectome", "gate_rank"), ("qlearning", "binary")),
             ("connectome [gate_error] - mlp [binary]", ("connectome", "gate_error"), ("mlp", "binary")),
             ("connectome [gate_rank] - mlp [binary]", ("connectome", "gate_rank"), ("mlp", "binary"))]
    for met in ("overall_success", "final20"):
        save_table(paired_table(allv, met, comps), f"gridworld_binary_paired_{met}",
                   f"Paired differences ({met}) with binary-feedback baselines, matched on task and seed", 3)


if __name__ == "__main__":
    tuning() if "--tuning" in sys.argv else final()
