"""python -m analysis.dan_kc_tables"""
import pandas as pd

from analysis.common import load, cells, wide, paired_table, save_table
from analysis.validate import read_manifests

CONDS = ["reward", "punish", "as_trained"]
KEYS = ["frac_kc_changed", "cont_rel_l2", "cont_max_abs", "jaccard", "jaccard_min", "mask_differs", "n_sel",
        "n_sel_delta", "mbon_rel_l2", "mbpref_abs_diff", "n_changed_native", "n_changed_odor", "n_edges_differ",
        "rel_update_diff"]


def main():
    m0 = read_manifests("addition_dan_kc")
    print("status:", m0.status.value_counts().to_dict(), "| commits:", m0.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m0.repo_dirty.fillna(False).sum()))
    d = load("addition_dan_kc")
    m = d[d.tier == "measure"].reset_index(drop=True)
    parts = []
    for cond in CONDS:
        p = m[["task", "seed"]].copy()
        for k in KEYS:
            p[k] = m[f"{cond}_{k}"].values
        p["condition"] = cond
        parts.append(p)
    long = pd.concat(parts, ignore_index=True)
    c = cells(long, ["task", "condition"], KEYS, 3)
    save_table(c.drop(columns="n"), "dan_kc_measure",
               "Effect of US (DAN) drive on KC activity and plasticity, relative to the odour-only forward pass "
               "(200 triples per seed, mean of per-seed means [95% bootstrap CI over 10 seeds]; jaccard_min = worst item per seed)", 2)
    save_table(cells(m, ["task"], ["none_n_sel", "mbpref_scale", "cache_mask_jaccard", "cache_max_abs"], 3).drop(columns="n"),
               "dan_kc_baseline",
               "Odour-only reference: selected KCs, mean |MB_pref|, and agreement of the exact cache with the simulator forward pass", 1)
    a = d[d.tier != "measure"]
    ca = cells(a, ["task", "label"], ["eval_acc_final", "eval_acc_pre", "final_drift_pct"], 3)
    save_table(wide(ca, "task", "eval_acc_final"), "dan_kc_ablation_acc",
               "Addition, native update with DAN-modulated vs odour-only plasticity mask: held-out accuracy")
    comps = [("odor_mask - native", ("connectome", "odor_mask"), ("connectome", "native"))]
    save_table(paired_table(a, "eval_acc_final", comps), "dan_kc_ablation_paired",
               "Addition: paired difference in held-out accuracy, odour-only mask minus native (matched on task, seed)", 3)
    main_runs = load("addition")
    main_runs = main_runs[(main_runs.agent == "connectome") & (main_runs.tier == "fair")
                          & main_runs.task.isin(["single_online", "two_online"])]
    j = main_runs[["task", "seed", "eval_acc_final"]].merge(
        a[a.tier == "native"][["task", "seed", "eval_acc_final"]], on=["task", "seed"], suffixes=("_main", "_native"))
    print(f"\nreproduction check ({len(j)} matched runs): max |main - native replica| = "
          f"{float((j.eval_acc_final_main - j.eval_acc_final_native).abs().max()):.3g} (must be 0)")


if __name__ == "__main__":
    main()
