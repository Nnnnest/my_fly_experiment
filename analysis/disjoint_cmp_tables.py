"""python -m analysis.disjoint_cmp_tables
Tables for the disjoint-evaluation number-comparison experiment (runs/comparison_disjoint), including pairing
with the main comparison runs (same seeds, same evaluation pairs)."""
import pandas as pd

from analysis.common import load, add_label, cells, wide, paired_table, save_table
from analysis.validate import read_manifests


def main():
    m0 = read_manifests("comparison_disjoint")
    print("status:", m0.status.value_counts().to_dict(), "| commits:", m0.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m0.repo_dirty.fillna(False).sum()))
    d = load("comparison_disjoint")
    print("max overlap with evaluation pairs:", int(d.n_overlap_with_eval.max()))
    main_runs = load("comparison")
    main_runs = main_runs[main_runs.task.isin(d.task.unique()) & (main_runs.tier == "fair")
                          & main_runs.agent.isin(["connectome", "linear", "mlp"])].copy()
    main_runs["tier"] = "fair_main"
    main_runs = add_label(main_runs)
    allv = pd.concat([d, main_runs], ignore_index=True)

    c = cells(allv, ["task", "label"], ["eval_acc_final"], 3)
    save_table(wide(c, "task", "eval_acc_final"), "disjoint_cmp_final_acc",
               "Number comparison, evaluation pairs disjoint from training (both orientations). "
               "'[fair_main]' = main experiment, same seeds and evaluation pairs, overlap allowed")

    K = ("connectome", "fair")
    comps = [("connectome - mlp", K, ("mlp", "fair")),
             ("connectome - linear", K, ("linear", "fair")),
             ("connectome - untrained connectome", K, ("connectome_notrain", "control")),
             ("native - delta (KC code)", K, ("delta", "kc")),
             ("native - ridge_kc_1x", K, ("ridge", "kc_1x")),
             ("ridge_kc_5x - native", ("ridge", "kc_5x"), K),
             ("ridge_kc_5x - ridge_kc_1x", ("ridge", "kc_5x"), ("ridge", "kc_1x")),
             ("ridge_kc_1x - ridge_raw_1x (what the KC expansion adds)", ("ridge", "kc_1x"), ("ridge", "raw_1x")),
             ("ridge_kc_5x - ridge_raw_5x (what the KC expansion adds)", ("ridge", "kc_5x"), ("ridge", "raw_5x")),
             ("connectome: disjoint - main", K, ("connectome", "fair_main")),
             ("mlp: disjoint - main", ("mlp", "fair"), ("mlp", "fair_main")),
             ("linear: disjoint - main", ("linear", "fair"), ("linear", "fair_main"))]
    save_table(paired_table(allv, "eval_acc_final", comps), "disjoint_cmp_paired",
               "Paired differences in evaluation accuracy, disjoint evaluation (matched on task and seed)", 3)

    order = ["connectome [fair]", "delta [kc]", "ridge [kc_1x]", "ridge [raw_1x]"]
    print("\nmean accuracy by task (ranking check):")
    mean = d.assign(lab=d.agent + " [" + d.tier + "]").groupby(["task", "lab"]).eval_acc_final.mean().unstack()
    mean = mean.rename(columns={"connectome [fair]": "connectome [fair]"})
    cols = [x for x in order if x in mean.columns]
    print(mean[cols].round(3).to_string())
    for task, row in mean[cols].iterrows():
        print(f"  {task}: " + " < ".join(row.sort_values().index))


if __name__ == "__main__":
    main()
