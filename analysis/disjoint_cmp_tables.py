"""python -m analysis.disjoint_cmp_tables
Tables for the disjoint-evaluation number-comparison experiment (runs/comparison_disjoint), paired with the main
comparison runs (same seeds, same evaluation pairs), plus a reproduction check against runs/readouts."""
import pandas as pd

from core.load import load_summaries
from analysis.common import load, add_label, cells, wide, paired_table, save_table
from analysis.validate import read_manifests

# (agent, tier) in comparison_disjoint  ->  key in runs/readouts summaries
READOUT_KEYS = {("delta", "kc"): "acc_delta",
                ("ridge", "kc_1x"): "acc_ridge_kc_1x", ("ridge", "kc_5x"): "acc_ridge_kc_5x",
                ("ridge", "raw_1x"): "acc_ridge_raw_1x", ("ridge", "raw_5x"): "acc_ridge_raw_5x"}


def reproduction_check(d):
    """Where no evaluation pair is rejected from the training stream (digits_5, digits_9), the readouts of the disjoint
    experiment must equal those of the 'readouts' experiment exactly."""
    old = load_summaries("readouts")
    old = old[old.status == "ok"].copy()
    print("\nreproduction check against runs/readouts (digits_5, digits_9: max |difference| must be 0)")
    for task in ("digits_5", "digits_9"):
        o = old[old.task == "cmp_" + task].set_index("seed")
        for (agent, tier), key in READOUT_KEYS.items():
            n = d[(d.task == task) & (d.agent == agent) & (d.tier == tier)].set_index("seed")
            diff = (n.eval_acc_final - o[key]).abs().max()
            print(f"  {task:9s} {agent:6s} {tier:7s} max diff {diff:.3g}")
    print("digits_2 and knob_n49 may differ slightly (rejected pairs change the training stream).")


def main():
    m0 = read_manifests("comparison_disjoint")
    print("status:", m0.status.value_counts().to_dict(), "| commits:", m0.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m0.repo_dirty.fillna(False).sum()))
    d = load("comparison_disjoint")
    print("max overlap with evaluation pairs:", int(d.n_overlap_with_eval.max()))
    reproduction_check(d)
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
    cols = [x for x in order if x in mean.columns]
    print(mean[cols].round(3).to_string())
    for task, row in mean[cols].iterrows():
        print(f"  {task}: " + " < ".join(row.sort_values().index))


if __name__ == "__main__":
    main()
