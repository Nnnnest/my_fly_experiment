"""python -m analysis.capacity_tables"""
import numpy as np
import pandas as pd

from analysis.common import load, add_label, cells, wide, paired_table, save_table, boot_ci, fmt, sort_tasks
from analysis.validate import read_manifests

K = lambda tier: ("connectome", tier)


def drops(a, tier, d_hi):
    """Per-seed accuracy drop from 2 digits to d_hi digits."""
    p = a[a.tier == tier].pivot(index="seed", columns="task", values="eval_acc_final")
    c = f"digits_{d_hi}"
    return (p["digits_2"] - p[c]).dropna() if c in p and "digits_2" in p else pd.Series(dtype=float)


def main():
    m = read_manifests("comparison_capacity")
    print("status:", m.status.value_counts().to_dict(), "| commits:", m.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m.repo_dirty.fillna(False).sum()))
    d = load("comparison_capacity")
    main_runs = load("comparison")
    main_runs = main_runs[(main_runs.agent == "connectome") & (main_runs.tier == "fair")
                          & main_runs.task.isin(d.task.unique())].copy()
    main_runs["tier"] = "main"
    allv = add_label(pd.concat([d, main_runs], ignore_index=True))

    a = d[d.agent == "connectome"]
    j = a[a.tier == "fixed"][["task", "seed", "eval_acc_final"]].merge(
        main_runs[["task", "seed", "eval_acc_final"]], on=["task", "seed"], suffixes=("_cap", "_main"))
    print("\nreproduction check, tier fixed vs runs/comparison (must be 0):",
          float((j.eval_acc_final_cap - j.eval_acc_final_main).abs().max()))

    c = cells(d[d.agent != "readouts_cap"], ["task", "label"], ["eval_acc_final"], 3)
    save_table(wide(c, "task", "eval_acc_final"), "capacity_final_acc",
               "Number comparison, digit positions: held-out accuracy by ALPNs per position (fixed = main allocation)")
    comps = [("connectome [g16] - connectome [fixed]", K("g16"), K("fixed")),
             ("connectome [g30] - connectome [fixed]", K("g30"), K("fixed")),
             ("connectome [g16] - connectome_notrain [g16]", K("g16"), ("connectome_notrain", "g16")),
             ("connectome [fixed] - connectome [main] (reproduction)", K("fixed"), K("main"))]
    save_table(paired_table(allv, "eval_acc_final", comps), "capacity_paired",
               "Paired differences in held-out accuracy (matched on task, seed)", 3)

    rows = []
    for tier in ("fixed", "g16", "g30"):
        for hi in (3, 4, 5, 9):
            x = drops(a, tier, hi)
            if len(x):
                rows.append({"regime": tier, "from_2_to": hi, "n": len(x), "drop": fmt(*boot_ci(x.values), d=3)})
    save_table(pd.DataFrame(rows), "capacity_drop", "Accuracy drop from 2 digits to n digits, by capacity regime", 2)
    rows = []
    for hi in (5, 9):
        x, y = drops(a, "fixed", hi), drops(a, "g16", hi)
        diff = (x - y).dropna()
        if len(diff):
            mean, lo, hi_ = boot_ci(diff.values)
            rows.append({"comparison": f"drop(2->{hi}) fixed - drop(2->{hi}) g16", "n": len(diff), "mean_diff": mean,
                         "ci_lo": lo, "ci_hi": hi_, "excludes_zero": bool(lo > 0 or hi_ < 0)})
    save_table(pd.DataFrame(rows), "capacity_drop_paired",
               "Does the digit-count effect shrink when per-position capacity is held constant? (positive = fixed degrades more)", 1)

    r = d[d.agent == "readouts_cap"]
    cols = ["n_alpn_per_position", "n_alpn_total", "kc_pos_mean", "kc_read_mean", "kc_read_std", "kc_read_p05",
            "kc_empty_frac", "acc_ridge_kc_1x", "acc_ridge_kc_5x", "acc_ridge_raw_1x", "acc_ridge_raw_5x", "acc_delta"]
    t = r.groupby(["task", "tier"])[cols].mean().round(3).reset_index()
    t["_t"] = t.task.map({x: i for i, x in enumerate(sort_tasks(t.task))})
    save_table(t.sort_values(["_t", "tier"]).drop(columns="_t"), "capacity_kc_stats",
               "ALPN capacity, KC activity on evaluation pairs, and readouts on the same inputs (mean over seeds)", 2)


if __name__ == "__main__":
    main()
