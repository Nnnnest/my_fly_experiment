"""python -m analysis.layout_tables
Unit of analysis = maze layout: results are first averaged over agent seeds within a layout, then bootstrapped over layouts."""
import pandas as pd

from analysis.common import load, boot_ci, fmt, save_table, label
from analysis.validate import read_manifests

METRICS = ["overall_success", "final20", "final_policy_acc", "collapsed_f"]
COMPS = [("connectome [gate_error] - qlearning", ("connectome", "gate_error"), ("qlearning", "fair")),
         ("connectome [gate_rank] - qlearning", ("connectome", "gate_rank"), ("qlearning", "fair")),
         ("connectome [opt] - qlearning", ("connectome", "opt"), ("qlearning", "fair")),
         ("mlp - qlearning", ("mlp", "fair"), ("qlearning", "fair")),
         ("connectome [gate_rank] - connectome [gate_error]", ("connectome", "gate_rank"), ("connectome", "gate_error")),
         ("kc_delta - rand_delta (does the wiring help?)", ("kc_delta", "diagnostic"), ("rand_delta", "diagnostic"))]


def main():
    m = read_manifests("gridworld_layouts")
    print("status:", m.status.value_counts().to_dict(), "| commits:", m.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m.repo_dirty.fillna(False).sum()))
    d = load("gridworld_layouts")
    d["size"] = d.task.str.extract(r"(maze\d+)@")[0]
    d["layout"] = d.task.str.extract(r"@(\d+)_")[0].astype(int)
    d["collapsed_f"] = d.collapsed.astype(float)
    g = d.groupby(["size", "agent", "tier", "label", "layout"])[METRICS].mean().reset_index()   # mean over agent seeds
    rows = []
    for (size, lab), x in g.groupby(["size", "label"], sort=False):
        row = {"size": size, "label": lab, "n_layouts": len(x)}
        for met in METRICS:
            row[met] = fmt(*boot_ci(x[met].values), d=3)
        row["worst_layout_final20"] = round(float(x.final20.min()), 3)
        rows.append(row)
    save_table(pd.DataFrame(rows), "layouts_summary",
               "Independent maze layouts, random start: mean over layouts [95% bootstrap CI over layouts]", 2)
    ix = g.set_index(["agent", "tier", "size", "layout"]).sort_index()
    rows = []
    for name, a, b in COMPS:
        try:
            A, B = ix.loc[a], ix.loc[b]
        except KeyError:
            continue
        for met in ("overall_success", "final20", "final_policy_acc"):
            diff = (A[met] - B[met]).dropna()
            for size, s in diff.groupby(level="size"):
                mean, lo, hi = boot_ci(s.values)
                rows.append({"comparison": name, "size": size, "metric": met, "n_layouts": len(s), "mean_diff": mean,
                             "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)})
    save_table(pd.DataFrame(rows), "layouts_paired", "Paired differences across layouts (matched on layout)", 2)


if __name__ == "__main__":
    main()
