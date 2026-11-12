"""python -m analysis.constrained_tables --tuning     (choose N0_DECAY)
   python -m analysis.constrained_tables              (final tables)"""
import os
import sys

import pandas as pd

from core import paths
from core.load import load_summaries
from analysis.common import load, cells, save_table, boot_ci
from analysis.validate import read_manifests

COLS = ["acc_untrained", "acc_native", "acc_ridge_1x", "acc_ridge_5x", "acc_nonneg_1x", "acc_nonneg_5x",
        "acc_anat_1x", "acc_anat_5x", "acc_rule_native", "acc_rule_no_renorm", "acc_rule_push_pull",
        "acc_rule_decay", "acc_rule_mask_active", "acc_rule_relax_all"]
COMPS = [("ridge_1x - native", "acc_ridge_1x", "acc_native"),
         ("nonneg_1x - ridge_1x (sign constraint)", "acc_nonneg_1x", "acc_ridge_1x"),
         ("anat_1x - nonneg_1x (anatomical channels)", "acc_anat_1x", "acc_nonneg_1x"),
         ("anat_1x - native", "acc_anat_1x", "acc_native"),
         ("anat_5x - native", "acc_anat_5x", "acc_native"),
         ("anat_5x - anat_1x", "acc_anat_5x", "acc_anat_1x"),
         ("rule_native - native (replica check)", "acc_rule_native", "acc_native"),
         ("rule_no_renorm - rule_native", "acc_rule_no_renorm", "acc_rule_native"),
         ("rule_push_pull - rule_native", "acc_rule_push_pull", "acc_rule_native"),
         ("rule_decay - rule_native", "acc_rule_decay", "acc_rule_native"),
         ("rule_mask_active - rule_native", "acc_rule_mask_active", "acc_rule_native"),
         ("rule_relax_all - rule_native", "acc_rule_relax_all", "acc_rule_native"),
         ("rule_relax_all - ridge_1x", "acc_rule_relax_all", "acc_ridge_1x")]


def tuning():
    d = load_summaries("readouts_constrained", root=os.path.join(paths.RUNS_DIR, "_tuning"))
    d = d[d.status == "ok"]
    cols = sorted(c for c in d.columns if c.startswith("acc_rule_decay_n"))
    print(d.groupby("task")[cols].mean().round(3).to_string())
    print("\nmean over tasks (set N0_DECAY in scripts/readouts/run_readouts_constrained.py to the best, commit):")
    print(d[cols].mean().round(4).to_string())


def reproduction(d):
    print("\nreproduction checks (max |difference|, must be 0 unless noted)")
    print("  rule_native vs native, weights:", float(d.max_dw_replica.max()),
          "| accuracy:", float((d.acc_rule_native - d.acc_native).abs().max()))
    for split, exp, tasks, agent in (("main", "readouts", {"single": "add_single", "two": "add_two"}, None),
                                     ("disjoint", "addition_disjoint", {"single": "single_online", "two": "two_online"}, "readouts")):
        try:
            old = load_summaries(exp)
        except FileNotFoundError:
            continue
        old = old[old.status == "ok"]
        if agent:
            old = old[old.agent == agent]
        for w, t_old in tasks.items():
            n = d[d.task == f"{w}_{split}"].set_index("seed")
            o = old[old.task == t_old].set_index("seed")
            if len(n) and len(o):
                print(f"  {w}_{split} vs runs/{exp}: ridge_1x {float((n.acc_ridge_1x - o.acc_ridge_kc_1x).abs().max()):.3g}, "
                      f"ridge_5x {float((n.acc_ridge_5x - o.acc_ridge_kc_5x).abs().max()):.3g}, "
                      f"native {float((n.acc_native - o.acc_native).abs().max()):.3g}")


def final():
    m = read_manifests("readouts_constrained")
    print("status:", m.status.value_counts().to_dict(), "| commits:", m.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m.repo_dirty.fillna(False).sum()))
    d = load("readouts_constrained")
    reproduction(d)
    print("decay N0 used: see scripts/readouts/run_readouts_constrained.py (N0_DECAY)")
    save_table(cells(d, ["task"], COLS, 3).drop(columns="n"), "constrained_acc",
               "Held-out accuracy on identical KC codes: native rule, constrained offline readouts, relaxed native rules (mean [95% CI], 10 seeds)", 1)
    rows = []
    for task, g in d.groupby("task", sort=False):
        for name, a, b in COMPS:
            diff = (g[a] - g[b]).dropna().values
            mean, lo, hi = boot_ci(diff)
            rows.append({"comparison": name, "task": task, "n": len(diff), "mean_diff": mean, "ci_lo": lo,
                         "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)})
    save_table(pd.DataFrame(rows), "constrained_paired", "Paired differences in held-out accuracy (matched on task, seed)", 2)
    p = ["n_kc", "n_km_edges", "n_kc_app", "n_kc_avo", "n_params_ridge", "n_params_nonneg", "n_params_anat",
         "nnz_nonneg_1x", "nnz_nonneg_5x", "nnz_anat_1x", "nnz_anat_5x"]
    save_table(d.groupby("task")[p].mean().round(1).reset_index(), "constrained_params",
               "Trainable parameters and number of non-zero constrained weights (mean over seeds)", 1)
    d[["task", "seed"] + COLS].sort_values(["task", "seed"]).to_csv(
        os.path.join(paths.REPO_ROOT, "paper", "tables", "constrained_per_seed.csv"), index=False)
    print("\nper-seed accuracies written to paper/tables/constrained_per_seed.csv")


if __name__ == "__main__":
    tuning() if "--tuning" in sys.argv else final()
