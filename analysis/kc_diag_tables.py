"""python -m analysis.kc_diag_tables"""
import numpy as np
import pandas as pd

from analysis.common import load, cells, wide, paired_table, save_table
from analysis.validate import read_manifests

FREQ = ["frac_zero", "frac_lt_k", "frac_mismatch", "frac_all_kc", "frac_none", "frac_same", "frac_tie_extended",
        "mean_pos", "mean_plast", "mean_read"]


def main():
    m = read_manifests("kc_diag")
    print("status:", m.status.value_counts().to_dict(), "| commits:", m.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m.repo_dirty.fillna(False).sum()))
    d = load("kc_diag")
    r = d[d.tier == "reproduce"]
    print("n_kc, k:", sorted(set(zip(d.n_kc.astype(int), d.k.astype(int)))),
          "| cache vs simulator KC activity disagreed in", int(d.n_cache_disagree.sum()), "updates (0 = US drive does not reach KCs)")

    main_runs = load("addition")
    main_runs = main_runs[(main_runs.agent == "connectome") & (main_runs.tier == "fair")
                          & main_runs.task.isin(r.task.unique())]
    j = r[["task", "seed", "eval_acc_pre", "eval_acc_final"]].merge(
        main_runs[["task", "seed", "eval_acc_pre", "eval_acc_final"]], on=["task", "seed"], suffixes=("_diag", "_main"))
    print(f"\nreproduction check vs runs/addition ({len(j)} matched runs; must be 0):",
          float((j.eval_acc_final_diag - j.eval_acc_final_main).abs().max()),
          float((j.eval_acc_pre_diag - j.eval_acc_pre_main).abs().max()))

    for prefix, cap in (("train_", "training updates (US drive included in the KC activity)"),
                        ("eval_", "evaluation inputs (input-only KC code)")):
        t = cells(r, ["task"], [prefix + f for f in FREQ], 3).drop(columns="n")
        save_table(t, f"kc_diag_freq_{prefix[:-1]}", f"Plasticity-mask diagnostics on {cap}: mean [95% CI] over 10 seeds", 1)
    t = cells(r, ["task"], ["train_frac_zero_pos", "train_frac_zero_neg", "mean_n_sel_edges", "mean_n_changed_edges",
                            "mean_rel_change"], 3).drop(columns="n")
    save_table(t, "kc_diag_updates", "Zero-input training updates by label; edges selected / changed per update", 1)

    c = cells(d, ["task", "label"], ["eval_acc_final", "n_trained", "n_skipped", "final_drift_pct"], 3)
    for met in ("eval_acc_final", "n_skipped", "n_trained"):
        save_table(wide(c, "task", met), f"kc_diag_{met}", f"Addition, training with degenerate updates skipped: {met}")
    K = ("connectome", "reproduce")
    comps = [("exclude_zero - reproduce", ("connectome", "exclude_zero"), K),
             ("exclude_mismatch - reproduce", ("connectome", "exclude_mismatch"), K)]
    save_table(paired_table(d, "eval_acc_final", comps), "kc_diag_paired",
               "Paired differences in held-out accuracy (same seeds, same evaluation set)", 3)


if __name__ == "__main__":
    main()
