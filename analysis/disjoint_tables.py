"""python -m analysis.disjoint_tables"""
import pandas as pd

from analysis.common import load, cells, wide, paired_table, save_table, boot_ci
from analysis.validate import read_manifests

COLS = ["acc_untrained", "acc_native", "acc_delta", "acc_ridge_kc_1x", "acc_ridge_kc_5x",
        "acc_ridge_raw_1x", "acc_ridge_raw_5x"]
COMPS = [("native - delta (KC code)", "acc_native", "acc_delta"),
         ("native - ridge_kc_1x", "acc_native", "acc_ridge_kc_1x"),
         ("ridge_kc_5x - native", "acc_ridge_kc_5x", "acc_native"),
         ("ridge_kc_1x - ridge_raw_1x (what the KC expansion adds)", "acc_ridge_kc_1x", "acc_ridge_raw_1x")]


def main():
    m = read_manifests("addition_disjoint")
    print("status:", m.status.value_counts().to_dict(), "| commits:", m.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m.repo_dirty.fillna(False).sum()))
    d = load("addition_disjoint")
    print("max train/eval overlap over runs:", int(d.train_eval_overlap.max()))
    a = d[d.agent != "readouts"]
    c = cells(a, ["task", "label"], ["eval_acc_final", "eval_acc_pre"], 3)
    save_table(wide(c, "task", "eval_acc_final"), "disjoint_addition_final_acc",
               "Addition with held-out operand pairs: accuracy after training (chance 0.5)")
    save_table(wide(c, "task", "eval_acc_pre"), "disjoint_addition_pre_acc", "Addition with held-out operand pairs: accuracy before training")
    k = ("connectome", "fair")
    comps = [("connectome - mlp [raw]", k, ("mlp", "raw")), ("connectome - linear [raw]", k, ("linear", "raw")),
             ("connectome - mlp [feat] (oracle features)", k, ("mlp", "feat")),
             ("connectome - linear [feat] (oracle features)", k, ("linear", "feat")),
             ("connectome [gate_error] - connectome", ("connectome", "gate_error"), k),
             ("connectome - untrained connectome", k, ("connectome_notrain", "control"))]
    save_table(paired_table(a, "eval_acc_final", comps), "disjoint_addition_paired",
               "Addition with held-out operand pairs: paired differences in accuracy", 3)
    r = d[d.agent == "readouts"]
    save_table(cells(r, ["task"], COLS, 3).drop(columns="n"), "disjoint_readouts_acc",
               "Readouts on identical KC codes, held-out operand pairs (mean [95% CI], 10 seeds)")
    rows = []
    for task, g in r.groupby("task", sort=False):
        for name, x, y in COMPS:
            diff = (g[x] - g[y]).dropna().values
            mean, lo, hi = boot_ci(diff)
            rows.append({"comparison": name, "task": task, "n": len(diff), "mean_diff": mean, "ci_lo": lo,
                         "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)})
    save_table(pd.DataFrame(rows), "disjoint_readouts_paired", "Readouts, held-out operand pairs: paired differences", 2)


if __name__ == "__main__":
    main()
