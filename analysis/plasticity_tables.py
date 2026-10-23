"""python -m analysis.plasticity_tables"""
import pandas as pd

from analysis.common import load, cells, boot_ci, save_table
from analysis.validate import read_manifests

METRIC = {"maze72_rs": ["overall_success", "final20"], "maze26_rs": ["overall_success", "final20"]}
DEFAULT = ["eval_acc_final"]


def grid(dt, met):
    c = cells(dt, ["g", "f"], [met], 3)
    p = c.pivot(index="g", columns="f", values=met).sort_index().sort_index(axis=1)
    p.columns = [f"f={x}" for x in p.columns]
    p.index.name = "g"
    return p.fillna("-").reset_index()


def paired(dt, met):
    base = dt[(dt.g == 1.0) & (dt.f == 1.0)].set_index("seed")[met]
    rows = []
    for (g, f), x in dt.groupby(["g", "f"]):
        if (g, f) == (1.0, 1.0):
            continue
        diff = (x.set_index("seed")[met] - base).dropna()
        mean, lo, hi = boot_ci(diff.values)
        rows.append({"g": g, "f": f, "n": len(diff), "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
                     "excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def main():
    m = read_manifests("plasticity")
    print("status:", m.status.value_counts().to_dict(), "| commits:", m.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m.repo_dirty.fillna(False).sum()))
    d = load("plasticity")
    d["g"] = d.tier.str.extract(r"g([\d.]+)_f")[0].astype(float)
    d["f"] = d.tier.str.extract(r"_f([\d.]+)$")[0].astype(float)
    if "collapsed" in d:
        d["collapsed_f"] = d.collapsed.astype(float)
    for task, dt in d.groupby("task"):
        for met in METRIC.get(task, DEFAULT):
            save_table(grid(dt, met), f"plasticity_{task}_{met}", f"{task}: {met} by strength g (rows) and frequency f (columns)")
            save_table(paired(dt, met), f"plasticity_{task}_{met}_paired",
                       f"{task}: {met}, difference to the unmodified rule (g=1, f=1), matched on seed", 2)
        if "collapsed_f" in dt and dt.collapsed_f.notna().any():
            c = dt.groupby(["g", "f"]).collapsed_f.agg(["sum", "count"]).reset_index()
            print(f"\n{task}: collapsed runs (sum / n)\n", c.to_string(index=False))


if __name__ == "__main__":
    main()
