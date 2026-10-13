import os
import re

import numpy as np
import pandas as pd

from core import paths
from core.load import load_summaries

OUT_TABLES = os.path.join(paths.REPO_ROOT, "paper", "tables")
OUT_FIGS = os.path.join(paths.REPO_ROOT, "paper", "figures")
N_BOOT = 10000
BASE = ["qlearning", "mlp", "linear", "connectome", "connectome_notrain", "kc_delta", "rand_delta"]


def label(agent, tier):
    return agent if tier in ("-", "fair") else f"{agent} [{tier}]"


def add_label(df):
    df["label"] = [label(a, t) for a, t in zip(df.agent, df.tier)]
    return df


def load(exp, root=None):
    d = load_summaries(exp, root=root)
    return add_label(d[d.status == "ok"].copy())


def sort_labels(labels):
    def k(l):
        b = l.split(" ")[0]
        return (BASE.index(b) if b in BASE else 99, l)
    return sorted(set(labels), key=k)


def sort_tasks(tasks):
    def k(t):
        return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(t))]
    return sorted(set(tasks), key=k)


def boot_ci(x, n_boot=N_BOOT, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return x[0], x[0], x[0]
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def fmt(m, lo, hi, d=3):
    return "-" if np.isnan(m) else f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def cells(df, by, metrics, digits=3):
    """One row per group; each metric as 'mean [lo, hi]'."""
    out = []
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        for m in metrics:
            row[m] = fmt(*boot_ci(g[m].values), d=digits)
        row["n"] = len(g)
        out.append(row)
    return pd.DataFrame(out)


def wide(c, index, metric, col="label"):
    p = c.pivot(index=index, columns=col, values=metric)
    p = p[sort_labels(p.columns)] if col == "label" else p[sorted(p.columns)]
    p = p.loc[sort_tasks(p.index)] if index == "task" else p.sort_index()
    return p.fillna("-").reset_index()


def paired_table(df, metric, comps):
    """comps: [(name, (agent, tier), (agent, tier))]; difference a-b matched on (task, seed)."""
    ix = df.set_index(["agent", "tier", "task", "seed"])[metric].sort_index()
    rows = []
    for name, a, b in comps:
        try:
            A, B = ix.loc[a], ix.loc[b]
        except KeyError:
            continue
        diff = (A - B).dropna()
        for task, g in diff.groupby(level="task"):
            m, lo, hi = boot_ci(g.values)
            rows.append({"comparison": name, "task": task, "n": len(g), "mean_diff": m,
                         "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)})
    out = pd.DataFrame(rows)
    if len(out):
        out["_c"] = out.comparison.map({n: i for i, (n, _, _) in enumerate(comps)})
        out["_t"] = out.task.map({t: i for i, t in enumerate(sort_tasks(out.task))})
        out = out.sort_values(["_c", "_t"]).drop(columns=["_c", "_t"]).reset_index(drop=True)
    return out


def _esc(s):
    return (str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_")
            .replace("%", "\\%").replace("&", "\\&"))


def to_latex(df, caption, name, n_left):
    cols = "l" * n_left + "r" * (df.shape[1] - n_left)
    L = ["\\begin{table}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
         " & ".join(_esc(c) for c in df.columns) + " \\\\", "\\midrule"]
    for _, r in df.round(4).iterrows():
        L.append(" & ".join(_esc(v) for v in r.values) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{_esc(caption)}}}", f"\\label{{tab:{name}}}", "\\end{table}"]
    return "\n".join(L)


def save_table(df, name, caption=None, n_left=1):
    os.makedirs(OUT_TABLES, exist_ok=True)
    df.to_csv(os.path.join(OUT_TABLES, name + ".csv"), index=False)
    with open(os.path.join(OUT_TABLES, name + ".tex"), "w") as f:
        f.write(to_latex(df, caption or name, name, n_left))
    print(f"\n--- {name} ({len(df)} rows) ---")
    print(df.round(4).to_string(index=False))
