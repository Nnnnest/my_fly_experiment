"""python -m analysis.figures"""
import traceback

import numpy as np

from core.load import load_metrics
from analysis.common import load, add_label, sort_labels, sort_tasks
from analysis.plotting import *

SIZES = {"maze26_rs": 26, "maze72_rs": 72, "maze113_rs": 113, "maze316_rs": 316}


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


def fig_gridworld():
    d = load("gridworld")
    d = d[d.task.isin(SIZES)].copy()
    d["states"] = d.task.map(SIZES)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for lab in sort_labels(d.label):
        line_ci(ax[0], d[d.label == lab], "states", "final20", lab)
    ax[0].set_xscale("log")
    ax[0].set_xticks(list(SIZES.values()))
    ax[0].set_xticklabels(list(SIZES.values()))
    ax[0].set(xlabel="traversable states", ylabel="success, last 20 episodes")
    ax[0].legend(fontsize=6)
    keep = ["qlearning", "mlp", "connectome [opt]", "connectome [gate_error]", "connectome [gate_rank]", "kc_delta [diagnostic]"]
    m = add_label(load_metrics("gridworld", metric="reached", task="maze113_rs"))
    m = rolling(m[m.label.isin(keep)], "value", 20)
    for lab in keep:
        if lab in set(m.label):
            band(ax[1], m[m.label == lab], lab, ycol="roll")
    ax[1].set(xlabel="episode", ylabel="success (rolling 20)", title="maze113, random start")
    ax[1].legend(fontsize=6)
    savefig(fig, "fig_gridworld")


def fig_tictactoe():
    d = load("tictactoe")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    keep = ["qlearning", "mlp", "connectome", "connectome [opt]", "connectome [gate_error]"]
    m = add_label(load_metrics("tictactoe", metric="outcome", task="rl_selfplay"))
    m = m[m.label.isin(keep)].copy()
    m["win"] = (m.value == 1).astype(float)
    m = rolling(m, "win", 100)
    for lab in keep:
        if lab in set(m.label):
            band(ax[0], m[m.label == lab], lab, ycol="roll")
    ax[0].set(xlabel="episode", ylabel="win rate (rolling 100)", title="self-play training")
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
    for fn in (fig_bandit, fig_comparison, fig_gridworld, fig_tictactoe, fig_addition):
        try:
            fn()
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
