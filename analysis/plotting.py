import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.common import OUT_FIGS, boot_ci

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
_COL = {}


def color(lab):
    if lab not in _COL:
        _COL[lab] = plt.cm.tab10(len(_COL) % 10)
    return _COL[lab]


def band(ax, d, lab, xcol="step", ycol="value"):
    g = d.groupby(xcol)[ycol]
    mean, sem = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
    c = color(lab)
    ax.plot(mean.index, mean.values, color=c, label=lab, lw=1.2)
    ax.fill_between(mean.index, (mean - 1.96 * sem).values, (mean + 1.96 * sem).values, color=c, alpha=0.2, lw=0)


def line_ci(ax, d, xcol, ycol, lab):
    xs, ms, los, his = [], [], [], []
    for x, g in d.groupby(xcol):
        m, lo, hi = boot_ci(g[ycol].values)
        xs.append(x); ms.append(m); los.append(lo); his.append(hi)
    c = color(lab)
    ax.plot(xs, ms, marker="o", ms=3, color=c, label=lab, lw=1.2)
    ax.fill_between(xs, los, his, color=c, alpha=0.2, lw=0)


def dotplot_grouped(ax, df, metric, tasks, labels, ylab=None):
    w, rng, seen = 0.8 / len(labels), np.random.default_rng(0), set()
    for j, lab in enumerate(labels):
        for i, t in enumerate(tasks):
            v = df[(df.label == lab) & (df.task == t)][metric].dropna().values
            if len(v) == 0:
                continue
            x = i - 0.4 + w * (j + 0.5)
            ax.scatter(x + rng.uniform(-w / 4, w / 4, len(v)), v, s=5, alpha=0.35, color=color(lab), lw=0)
            m, lo, hi = boot_ci(v)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=3.5, capsize=2, color=color(lab),
                        label=None if lab in seen else lab)
            seen.add(lab)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    if ylab:
        ax.set_ylabel(ylab)


def dots_by_label(ax, df, metric, labels, ylab=None):
    rng = np.random.default_rng(0)
    for i, lab in enumerate(labels):
        v = df[df.label == lab][metric].dropna().values
        if len(v) == 0:
            continue
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=6, alpha=0.4, color=color(lab), lw=0)
        m, lo, hi = boot_ci(v)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=4, capsize=2, color=color(lab))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    if ylab:
        ax.set_ylabel(ylab)


def savefig(fig, name):
    os.makedirs(OUT_FIGS, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_FIGS, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", name)
