"""python -m analysis.figures"""
import os
import re
import traceback

import numpy as np
import pandas as pd

import analysis.plotting as P
from core.load import load_metrics
from analysis.common import load, add_label, sort_labels, sort_tasks, OUT_TABLES
from analysis.plotting import *

# fixed palette (Okabe-Ito): one colour per agent, identical in every figure
P._COL.update({
    "qlearning": "#000000", "qlearning [opt]": "#000000", "qlearning [sym]": "#000000",
    "mlp": "#E69F00", "mlp [opt]": "#E69F00", "mlp [sym]": "#E69F00",
    "linear": "#CC79A7",
    "connectome": "#0072B2", "connectome [sleep50]": "#56B4E9", "connectome_notrain [control]": "#999999",
    "connectome [opt]": "#56B4E9", "connectome [gate_visit]": "#CC79A7",
    "connectome [gate_error]": "#009E73", "connectome [gate_rank]": "#D55E00",
    "connectome [sym]": "#56B4E9", "connectome [sym_sleep500]": "#009E73", "connectome [opt_sleep100]": "#009E73",
    "kc_delta [diagnostic]": "#7B3294", "rand_delta [diagnostic]": "#999999",
    "connectome [block]": "#0072B2", "connectome [balanced]": "#56B4E9",
    "single-digit": "#0072B2", "two-digit": "#D55E00",
})
color = P.color

SIZES = {"maze26_rs": 26, "maze72_rs": 72, "maze113_rs": 113, "maze316_rs": 316}
NUM = re.compile(r"(-?\d+\.?\d*)\s*\[\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\]")


def parse(cell):
    """'0.919 [0.888, 0.944]' -> (0.919, 0.888, 0.944)"""
    m = NUM.search(str(cell))
    return tuple(float(x) for x in m.groups()) if m else (np.nan, np.nan, np.nan)


def read_table(name):
    p = os.path.join(OUT_TABLES, name + ".csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing: regenerate the table first")
    return pd.read_csv(p)


def rolling(m, col, w):
    m = m.sort_values(["label", "seed", "step"]).copy()
    m["roll"] = m.groupby(["label", "seed"])[col].transform(lambda s: s.rolling(w, min_periods=w).mean())
    return m.dropna(subset=["roll"])


def fig_bandit():
    m = add_label(load_metrics("bandit", metric="pseudo_regret", task="mp10"))
    m = m.sort_values(["label", "seed", "step"])
    m["value"] = m.groupby(["label", "seed"]).value.cumsum()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for lab in sort_labels(m.label):
        band(ax[0], m[m.label == lab], lab)
    ax[0].set(xlabel="trial", ylabel="cumulative pseudo-regret", title="min_pulls = 10")
    ax[0].legend()
    s = load("bandit")
    s["mp"] = s.task.str[2:].astype(int)
    for lab in sort_labels(s.label):
        line_ci(ax[1], s[s.label == lab], "mp", "cum_pseudo_regret", lab)
    ax[1].set(xlabel="forced-exploration floor (min_pulls)", ylabel="final cumulative pseudo-regret")
    savefig(fig, "fig_bandit")


def fig_bandit_perm():
    reg, best = read_table("bandit_perm_cum_pseudo_regret"), read_table("bandit_perm_pick_best")
    tasks = [t for t in sort_tasks(reg.task) if t.endswith("_mp10")]
    cols = [c for c in ("qlearning", "mlp", "connectome [block]", "connectome [balanced]") if c in reg.columns]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, tab, yl in ((ax[0], reg, "cumulative pseudo-regret (log)"),
                       (ax[1], best, "fraction of seeds whose most-played arm is the best")):
        w = 0.8 / len(cols)
        for j, c in enumerate(cols):
            for i, t in enumerate(tasks):
                m, lo, hi = parse(tab.loc[tab.task == t, c].iloc[0])
                a.errorbar(i - 0.4 + w * (j + 0.5), m, yerr=[[m - lo], [hi - m]], fmt="o", ms=4, capsize=2,
                           color=color(c), label=c if i == 0 else None)
        a.set_xticks(range(len(tasks)))
        a.set_xticklabels([t.replace("_mp10", "") for t in tasks], rotation=30, ha="right")
        a.set_ylabel(yl)
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=6)
    ax[0].set_title("arm order controlled (rev = best arm first, perm = random), min_pulls = 10", fontsize=7)
    savefig(fig, "fig_bandit_perm")


def fig_comparison():
    d = load("comparison")
    labels = sort_labels(d.label)
    tasks = sort_tasks(d.task)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    dotplot_grouped(ax[0], d, "eval_acc_final", [t for t in tasks if t.startswith("knob")], labels, "held-out accuracy")
    dotplot_grouped(ax[1], d, "eval_acc_final", [t for t in tasks if t.startswith("digits")], labels)
    for a in ax:
        a.axhline(0.5, ls=":", c="gray", lw=0.8)
    ax[0].legend(fontsize=6)
    savefig(fig, "fig_comparison")


def _levels_panel(a, name, title, ylab=None):
    t = read_table(name)
    tasks = sort_tasks(t.task)
    cols = ["real", "shuf0", "shuf1", "shuf2"]
    style = {"real": ("o", color("connectome")), "shuf0": ("s", "#7f7f7f"),
             "shuf1": ("^", "#7f7f7f"), "shuf2": ("D", "#7f7f7f")}
    w = 0.8 / len(cols)
    for j, c in enumerate(cols):
        mk, col = style[c]
        for i, tk in enumerate(tasks):
            m, lo, hi = parse(t.loc[t.task == tk, c].iloc[0])
            lab = ("real wiring" if c == "real" else f"shuffled ({c[-1]})") if i == 0 else None
            a.errorbar(i - 0.4 + w * (j + 0.5), m, yerr=[[m - lo], [hi - m]], fmt=mk, ms=4, capsize=2,
                       color=col, label=lab)
    a.axhline(0.5, ls=":", c="gray", lw=0.8)
    a.set_xticks(range(len(tasks)))
    a.set_xticklabels(tasks, rotation=30, ha="right")
    a.set_title(title, fontsize=8)
    if ylab:
        a.set_ylabel(ylab)


def fig_wiring():
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    _levels_panel(ax[0], "wiring_comparison", "number comparison", "held-out accuracy (connectome)")
    _levels_panel(ax[1], "wiring_addition", "addition")
    ax[0].legend(fontsize=6)
    cmp_, add_ = read_table("wiring_comparison_paired"), read_table("wiring_addition_paired")
    p = pd.concat([cmp_, add_], ignore_index=True)
    tasks = sort_tasks(cmp_.task) + sort_tasks(add_.task)
    w = 0.8 / 3
    for j, (cn, mk) in enumerate((("real - shuf0", "s"), ("real - shuf1", "^"), ("real - shuf2", "D"))):
        for i, tk in enumerate(tasks):
            r = p[(p.comparison == cn) & (p.task == tk)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax[2].errorbar(i - 0.4 + w * (j + 0.5), r.mean_diff, yerr=[[r.mean_diff - r.ci_lo], [r.ci_hi - r.mean_diff]],
                           fmt=mk, ms=3.5, capsize=2, color=color("connectome"),
                           label=f"vs shuffled ({cn[-1]})" if i == 0 else None)
    ax[2].axhline(0, c="k", lw=0.8)
    ax[2].set_xticks(range(len(tasks)))
    ax[2].set_xticklabels(tasks, rotation=40, ha="right", fontsize=6)
    ax[2].set_ylabel("accuracy: real minus shuffled wiring")
    ax[2].set_title("paired difference, 95% bootstrap CI", fontsize=8)
    ax[2].legend(fontsize=6)
    savefig(fig, "fig_wiring")


GRID_PANELS = [
    ("no gate (one-hot / k-hot encoding)", ["connectome", "connectome [opt]"]),
    ("gate: visit count", ["connectome [gate_visit]"]),
    ("gate: error", ["connectome [gate_error]"]),
    ("gate: rank", ["connectome [gate_rank]"]),
    ("external delta rule on the KC code", ["kc_delta [diagnostic]", "rand_delta [diagnostic]"]),
]


def fig_gridworld_scaling():
    d = load("gridworld")
    d = d[d.task.isin(SIZES)].copy()
    d["states"] = d.task.map(SIZES)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (title, labs) in zip(axes, GRID_PANELS):
        for ref in ("qlearning", "mlp"):
            r = d[d.label == ref].groupby("states").final20.mean()
            ax.plot(r.index, r.values, ls="--", lw=1, color=color(ref), label=ref)
        for lab in labs:
            if lab in set(d.label):
                line_ci(ax, d[d.label == lab], "states", "final20", lab)
        ax.set_title(title, fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(list(SIZES.values()))
        ax.set_xticklabels(list(SIZES.values()))
        ax.minorticks_off()
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=6, loc="lower left")
    for ax in axes[3:5]:
        ax.set_xlabel("traversable states")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("success, last 20 episodes")
    axes[5].axis("off")
    axes[5].text(0, 0.6, "dashed = reference agents\nbands = 95% bootstrap CI over 10 seeds\n"
                         "random start; 'connectome' (one-hot) exists\nonly for 26 and 72 states", fontsize=7)
    savefig(fig, "fig_gridworld_scaling")


def fig_gridworld_curves():
    keep = ["qlearning", "mlp", "connectome", "connectome [opt]", "connectome [gate_error]", "connectome [gate_rank]"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for a, task in zip(ax, ("maze72_rs", "maze113_rs")):
        m = add_label(load_metrics("gridworld", metric="reached", task=task))
        m = rolling(m[m.label.isin(keep)], "value", 20)
        for lab in keep:
            if lab in set(m.label):
                band(a, m[m.label == lab], lab, ycol="roll")
        a.set(xlabel="episode", title=task.replace("_rs", ", random start"))
        a.legend(fontsize=6, loc="lower right")
    ax[0].set_ylabel("success (rolling 20)")
    savefig(fig, "fig_gridworld_curves")


def fig_tictactoe():
    d = load("tictactoe")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    keep = ["qlearning", "mlp", "connectome", "connectome [opt]", "connectome [gate_error]"]
    m = add_label(load_metrics("tictactoe", metric="coverage_acc", task="rl_selfplay"))
    m = m[m.label.isin(keep)]
    for lab in keep:
        if lab in set(m.label):
            line_ci(ax[0], m[m.label == lab], "step", "value", lab)
    ax[0].set(xlabel="self-play episodes", ylabel="minimax coverage", title="self-play: policy quality")
    ax[0].legend(fontsize=6)
    im = d[d.task == "imitation"]
    dots_by_label(ax[1], im, "coverage_acc", sort_labels(im.label), "minimax coverage")
    ax[1].set_title("imitation")
    h = d[d.task == "rl_heuristic"]
    dots_by_label(ax[2], h, "eval_heuristic_draw", sort_labels(h.label), "draw rate vs heuristic")
    ax[2].set_title("RL vs heuristic, greedy eval")
    savefig(fig, "fig_tictactoe")


def fig_addition():
    d = load("addition")
    labels = sort_labels(d.label)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    dotplot_grouped(ax[0], d, "eval_acc_final", ["single_online", "two_online"], labels, "held-out accuracy")
    dotplot_grouped(ax[1], d, "eval_acc_final", ["single_imitation", "two_imitation"], labels)
    for a in ax[:2]:
        a.axhline(0.5, ls=":", c="gray", lw=0.8)
    ax[0].legend(fontsize=6)
    g = load("addition_diag")
    cols = [c for c in g.columns if c.startswith("jaccard_diff")]
    long = g.melt(id_vars=["task", "seed"], value_vars=cols, var_name="name", value_name="jaccard")
    long["d"] = long.name.str.replace("jaccard_diff", "").astype(int)
    for t in ("single", "two"):
        line_ci(ax[2], long[long.task == t], "d", "jaccard", f"{t}-digit")
        ax[2].axhline(g[g.task == t].jaccard_random_mean.mean(), ls="--", lw=0.8, color=color(f"{t}-digit"))
    ax[2].set(xlabel="c - (a+b) offset d", ylabel="KC-code Jaccard vs correct triple", title="dashed = random-triple baseline")
    ax[2].legend(fontsize=6)
    savefig(fig, "fig_addition")


def main():
    for fn in (fig_bandit, fig_bandit_perm, fig_comparison, fig_wiring, fig_gridworld_scaling,
               fig_gridworld_curves, fig_tictactoe, fig_addition):
        try:
            fn()
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
