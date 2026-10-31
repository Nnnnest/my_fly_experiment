"""python -m analysis.layouts_binary_tables
Independent layouts: success is averaged over agent seeds within a layout and intervals are bootstrapped over layouts
(as in the layout experiment). Binary-feedback baselines are paired with the shaped baselines and the gates on the
same layouts."""
import pandas as pd

from analysis.common import load, boot_ci, fmt, save_table
from analysis.validate import read_manifests

METRICS = ["overall_success", "final20", "final_policy_acc", "collapsed_f"]
KEEP = ["qlearning", "qlearning [binary]", "mlp", "mlp [binary]", "connectome [gate_error]", "connectome [gate_rank]"]


def paired_layouts(agg, metric, comps):
    rows = []
    for name, a, b in comps:
        for size, g in agg.groupby("size"):
            pa = g[g.label == a].set_index("layout")[metric]
            pb = g[g.label == b].set_index("layout")[metric]
            diff = (pa - pb).dropna()
            if len(diff):
                m, lo, hi = boot_ci(diff.values)
                rows.append({"comparison": name, "size": size, "metric": metric, "n_layouts": len(diff),
                             "mean_diff": m, "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def main():
    m = read_manifests("gridworld_layouts_binary")
    print("status:", m.status.value_counts().to_dict(), "| commits:", m.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m.repo_dirty.fillna(False).sum()))
    b = load("gridworld_layouts_binary")
    s = load("gridworld_layouts")
    d = pd.concat([b, s[s.label.isin(KEEP)]], ignore_index=True)
    d["collapsed_f"] = d.collapsed.astype(float)
    d["size"] = d.task.str.split("@").str[0]
    d["layout"] = d.task.str.split("@").str[1].str.split("_").str[0].astype(int)
    agg = d.groupby(["size", "label", "layout"])[METRICS].mean().reset_index()      # mean over agent seeds
    rows = []
    for (size, label), g in agg.groupby(["size", "label"]):
        row = {"size": size, "label": label, "n_layouts": len(g)}
        for met in METRICS:
            row[met] = fmt(*boot_ci(g[met].values), d=3)
        row["worst_layout_final20"] = round(float(g.final20.min()), 3)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["_o"] = summary.label.map({k: i for i, k in enumerate(KEEP)})
    summary = summary.sort_values(["size", "_o"]).drop(columns="_o")
    save_table(summary, "layouts_binary_summary",
               "Independent layouts; shaped vs binary-feedback baselines (mean over agent seeds within a layout; CIs over layouts)", 2)
    comps = [("qlearning [binary] - qlearning [shaped]", "qlearning [binary]", "qlearning"),
             ("mlp [binary] - mlp [shaped]", "mlp [binary]", "mlp"),
             ("connectome [gate_error] - qlearning [binary]", "connectome [gate_error]", "qlearning [binary]"),
             ("connectome [gate_rank] - qlearning [binary]", "connectome [gate_rank]", "qlearning [binary]"),
             ("connectome [gate_error] - mlp [binary]", "connectome [gate_error]", "mlp [binary]"),
             ("connectome [gate_rank] - mlp [binary]", "connectome [gate_rank]", "mlp [binary]")]
    out = pd.concat([paired_layouts(agg, met, comps) for met in ("overall_success", "final20", "final_policy_acc")],
                    ignore_index=True)
    save_table(out, "layouts_binary_paired", "Paired differences over layouts (shaped vs binary baselines, gates)", 2)


if __name__ == "__main__":
    main()
