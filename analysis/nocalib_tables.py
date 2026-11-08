"""python -m analysis.nocalib_tables"""
import pandas as pd

from analysis.common import load, cells, wide, paired_table, save_table
from analysis.validate import read_manifests


def main():
    m = read_manifests("nocalib")
    print("status:", m.status.value_counts().to_dict(), "| commits:", m.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m.repo_dirty.fillna(False).sum()))
    d = load("nocalib")

    # reproduction check: tier 'full' must equal the existing overlap-free connectome runs exactly
    print("\nreproduction check (max |nocalib[full] - existing run|, must be 0)")
    for exp in ("comparison_disjoint", "addition_disjoint"):
        old = load(exp)
        old = old[(old.agent == "connectome") & (old.tier == "fair")][["task", "seed", "eval_acc_final", "eval_acc_pre"]]
        new = d[d.tier == "full"][["task", "seed", "eval_acc_final", "eval_acc_pre"]]
        j = new.merge(old, on=["task", "seed"], suffixes=("_new", "_old"))
        if len(j):
            print(f"  {exp}: {len(j)} matched runs, final {float((j.eval_acc_final_new - j.eval_acc_final_old).abs().max()):.3g}, "
                  f"pre {float((j.eval_acc_pre_new - j.eval_acc_pre_old).abs().max()):.3g}")

    c = cells(d, ["task", "label"], ["eval_acc_pre", "eval_acc_final"], 3)
    for met in ("eval_acc_final", "eval_acc_pre"):
        save_table(wide(c, "task", met), f"nocalib_{met}",
                   f"Native rule with full, constant-offset-only and no calibration: {met} (mean [95% CI], 10 seeds)")
    F = ("connectome", "full")
    comps = [("const - full", ("connectome", "const"), F), ("raw - full", ("connectome", "raw"), F),
             ("raw - const", ("connectome", "raw"), ("connectome", "const"))]
    save_table(paired_table(d, "eval_acc_final", comps), "nocalib_paired",
               "Accuracy after training without calibration minus with it (paired on task and seed)", 3)


if __name__ == "__main__":
    main()
