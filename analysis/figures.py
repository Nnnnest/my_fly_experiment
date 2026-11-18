"""python -m analysis.figures"""
import os
import re
import traceback

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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
# follow-up experiments (colours keyed by display label)
P._COL.update({
    "qlearning [binary]": "#666666", "mlp [binary]": "#F0E442",
    "native rule": "#0072B2", "untrained": "#999999", "delta rule on KC": "#7B3294",
    "ridge on KC (1x)": "#009E73", "ridge on KC (5x)": "#005C45",
    "ridge on raw (1x)": "#BBBBBB", "ridge on raw (5x)": "#555555",
    "non-negative ridge on KC (1x)": "#56B4E9", "anatomical channels on KC (1x)": "#D55E00",
    "anatomical channels on KC (5x)": "#8C3A00",
    "native, no renormalisation": "#56B4E9", "native, push-pull": "#009E73", "native, decaying step": "#E69F00",
    "native, active-mask only": "#CC79A7", "native, relax all constraints": "#D55E00",
    "fixed (300 ALPN total)": "#0072B2", "16 ALPN per position": "#D55E00", "30 ALPN per position": "#009E73",
    "connectome (main, overlap allowed)": "#56B4E9", "connectome (disjoint)": "#0072B2",
    "untrained connectome": "#999999", "mlp [oracle features]": "#E69F00", "linear [oracle features]": "#CC79A7",
    "native update": "#0072B2", "odour-only plasticity mask": "#D55E00",
    "reward-only connectome": "#0072B2", "reward+punish connectome": "#009E73",
    "win": "#009E73", "draw": "#0072B2", "loss": "#D55E00",
    "reward drive": "#0072B2", "punishment drive": "#D55E00", "as trained": "#009E73",
})
color = P.color

SIZES = {"maze26_rs": 26, "maze72_rs": 72, "maze113_rs": 113, "maze316_rs": 316}
NUM = re.compile(r"(-?\d+\.?\d*)\s*\[\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\]")
NAN3 = (np.nan, np.nan, np.nan)


def parse(cell):
    """'0.919 [0.888, 0.944]' -> (0.919, 0.888, 0.944); a plain number -> (x, x, x); anything else -> nan"""
    s = str(cell)
    m = NUM.search(s)
    if m:
        return tuple(float(x) for x in m.groups())
    try:
        v = float(s)
        return (v, v, v)
    except ValueError:
        return NAN3


def read_table(name):
    p = os.path.join(OUT_TABLES, name + ".csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing: regenerate the table first")
    return pd.read_csv(p)


def _cell(tab, row_col, row_val, col):
    if col not in tab.columns:
        return NAN3
    r = tab.loc[tab[row_col] == row_val, col]
    return parse(r.iloc[0]) if len(r) else NAN3


def _bar(ax, x, m, lo, hi, lab, seen, fmt="o", ms=3.5):
    ax.errorbar(x, m, yerr=[[max(m - lo, 0)], [max(hi - m, 0)]], fmt=fmt, ms=ms, capsize=2, color=color(lab),
                label=None if lab in seen else lab)
    seen.add(lab)


def groups(ax, tab, tasks, cols, row_col="task", ylab=None, ref=None, rotate=30):
    """cols: [(table column, display label)]; one point with CI per (task, column)."""
    w, seen = 0.8 / len(cols), set()
    for j, (c, lab) in enumerate(cols):
        for i, t in enumerate(tasks):
            m, lo, hi = _cell(tab, row_col, t, c)
            if not np.isnan(m):
                _bar(ax, i - 0.4 + w * (j + 0.5), m, lo, hi, lab, seen)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=rotate, ha="right")
    if ref is not None:
        ax.axhline(ref, ls=":", c="gray", lw=0.8)
    if ylab:
        ax.set_ylabel(ylab)


def rolling(m, col, w):
    m = m.sort_values(["label", "seed", "step"]).copy()
    m["roll"] = m.groupby(["label", "seed"])[col].transform(lambda s: s.rolling(w, min_periods=w).mean())
    return m.dropna(subset=["roll"])


# ======================================================================= main experiments

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


# ======================================================================= follow-up experiments

def fig_readouts():
    """Native rule vs delta rule vs ridge on identical KC codes (and on raw inputs), evaluation items disjoint from training."""
    cmp_t, add_t = read_table("disjoint_cmp_final_acc"), read_table("disjoint_readouts_acc")
    cmp_cols = [("connectome", "native rule"), ("delta [kc]", "delta rule on KC"),
                ("ridge [kc_1x]", "ridge on KC (1x)"), ("ridge [kc_5x]", "ridge on KC (5x)"),
                ("ridge [raw_1x]", "ridge on raw (1x)"), ("ridge [raw_5x]", "ridge on raw (5x)"),
                ("connectome_notrain [control]", "untrained")]
    add_cols = [("acc_native", "native rule"), ("acc_delta", "delta rule on KC"),
                ("acc_ridge_kc_1x", "ridge on KC (1x)"), ("acc_ridge_kc_5x", "ridge on KC (5x)"),
                ("acc_ridge_raw_1x", "ridge on raw (1x)"), ("acc_ridge_raw_5x", "ridge on raw (5x)"),
                ("acc_untrained", "untrained")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    groups(ax[0], cmp_t, sort_tasks(cmp_t.task), cmp_cols, ylab="held-out accuracy", ref=0.5)
    ax[0].set_title("number comparison (evaluation pairs excluded from training)", fontsize=8)
    groups(ax[1], add_t, sort_tasks(add_t.task), add_cols, ref=0.5)
    ax[1].set_title("addition (held-out operand pairs)", fontsize=8)
    ax[0].legend(fontsize=6)
    savefig(fig, "fig_readouts")


def fig_constrained():
    """Which constraint of the native mechanism costs accuracy? Readout ladder (left), native-rule ablations (right)."""
    t = read_table("constrained_acc")
    tasks = sort_tasks(t.task)
    ladder = [("acc_untrained", "untrained"), ("acc_native", "native rule"),
              ("acc_nonneg_1x", "non-negative ridge on KC (1x)"), ("acc_anat_1x", "anatomical channels on KC (1x)"),
              ("acc_ridge_1x", "ridge on KC (1x)"), ("acc_ridge_5x", "ridge on KC (5x)"),
              ("acc_anat_5x", "anatomical channels on KC (5x)")]
    rules = [("acc_untrained", "untrained"), ("acc_rule_native", "native rule"),
             ("acc_rule_no_renorm", "native, no renormalisation"), ("acc_rule_push_pull", "native, push-pull"),
             ("acc_rule_decay", "native, decaying step"), ("acc_rule_mask_active", "native, active-mask only"),
             ("acc_rule_relax_all", "native, relax all constraints")]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.8), sharey=True)
    groups(ax[0], t, tasks, ladder, ylab="held-out accuracy", ref=0.5)
    ax[0].set_title("readouts on the same KC codes, constraints added step by step", fontsize=8)
    groups(ax[1], t, tasks, rules, ref=0.5)
    ax[1].set_title("variants of the native update rule", fontsize=8)
    for a in ax:
        a.legend(fontsize=6, loc="center left")
    savefig(fig, "fig_constrained")


def fig_capacity():
    """Digit comparison with a fixed total ALPN pool versus a fixed number of ALPNs per digit position."""
    t = read_table("capacity_final_acc")
    tasks = sort_tasks(t.task)
    nd = [int(x.split("_")[1]) for x in tasks]
    cols = [("connectome [fixed]", "fixed (300 ALPN total)"), ("connectome [g16]", "16 ALPN per position"),
            ("connectome [g30]", "30 ALPN per position"), ("connectome_notrain [fixed]", "untrained")]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.7))
    for c, lab in cols:
        xs, ms, los, his = [], [], [], []
        for n, tk in zip(nd, tasks):
            m, lo, hi = _cell(t, "task", tk, c)
            if not np.isnan(m):
                xs.append(n); ms.append(m); los.append(lo); his.append(hi)
        if xs:
            ms_, lo_, hi_ = np.array(ms), np.array(los), np.array(his)
            ax[0].errorbar(xs, ms_, yerr=[ms_ - lo_, hi_ - ms_], marker="o", ms=3, capsize=2, lw=1,
                           color=color(lab), label=lab)
    ax[0].set(xlabel="digits per number", ylabel="held-out accuracy", title="native rule", xticks=nd)
    ax[0].axhline(0.5, ls=":", c="gray", lw=0.8)
    ax[0].legend(fontsize=6)
    dr = read_table("capacity_drop")
    names = {"fixed": "fixed (300 ALPN total)", "g16": "16 ALPN per position", "g30": "30 ALPN per position"}
    for reg, lab in names.items():
        s = dr[dr.regime == reg].sort_values("from_2_to")
        if s.empty:
            continue
        p = np.array([parse(x) for x in s["drop"]])
        ax[1].errorbar(s.from_2_to, p[:, 0], yerr=[p[:, 0] - p[:, 1], p[:, 2] - p[:, 0]], marker="o", ms=3,
                       capsize=2, lw=1, color=color(lab), label=lab)
    ax[1].axhline(0, c="k", lw=0.8)
    ax[1].set(xlabel="digits per number", ylabel="accuracy drop relative to 2 digits",
              title="drop with digit count", xticks=sorted(dr.from_2_to.unique()))
    ax[1].legend(fontsize=6)
    ks = read_table("capacity_kc_stats")
    for tier, lab in (("fixed", "fixed (300 ALPN total)"), ("g16", "16 ALPN per position"), ("g30", "30 ALPN per position")):
        s = ks[ks.tier == tier].copy()
        s["n"] = s.task.str.split("_").str[1].astype(int)
        s = s.sort_values("n")
        ax[2].plot(s.n, s.acc_ridge_kc_1x, marker="o", ms=3, lw=1, color=color(lab), label=lab)
    ax[2].set(xlabel="digits per number", ylabel="accuracy of ridge readout on the KC code (1x)",
              title="information in the KC code", xticks=sorted(ks.task.str.split("_").str[1].astype(int).unique()))
    ax[2].legend(fontsize=6)
    savefig(fig, "fig_capacity")


def fig_plasticity():
    """Update strength g (x) and update frequency f (lines) of the native rule."""
    panels = [("plasticity_cmp_knob_n49_eval_acc_final", "comparison, knob n_max = 49"),
              ("plasticity_cmp_digits_5_eval_acc_final", "comparison, 5 digits"),
              ("plasticity_add_single_eval_acc_final", "addition, single-digit"),
              ("plasticity_add_two_eval_acc_final", "addition, two-digit")]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8))
    axes = axes.ravel()
    for a, (name, title) in zip(axes[:4], panels):
        t = read_table(name)
        gs = sorted(t.g.astype(float))
        for c in [c for c in t.columns if c.startswith("f=")]:
            xs, ms, los, his = [], [], [], []
            for _, r in t.iterrows():
                m, lo, hi = parse(r[c])
                if not np.isnan(m):
                    xs.append(float(r["g"])); ms.append(m); los.append(lo); his.append(hi)
            if xs:
                ms_, lo_, hi_ = np.array(ms), np.array(los), np.array(his)
                a.errorbar(xs, ms_, yerr=[ms_ - lo_, hi_ - ms_], marker="o", ms=3, capsize=2, lw=1,
                           color=color("plasticity " + c), label=c.replace("f=", "frequency f = "))
        a.set_xscale("log")
        a.set_xticks(gs)
        a.set_xticklabels([str(g) for g in gs])
        a.minorticks_off()
        a.set(xlabel="update strength g", ylabel="held-out accuracy")
        a.set_title(title, fontsize=8)
        a.legend(fontsize=6)
    for a, (name, title) in zip(axes[4:], (("plasticity_maze72_rs_overall_success", "72-state maze, ungated, overall success"),
                                          ("plasticity_maze72_rs_final20", "72-state maze, ungated, last 20 episodes"))):
        t = read_table(name)
        pts = []
        for _, r in t.iterrows():
            for c in [c for c in t.columns if c.startswith("f=")]:
                m, lo, hi = parse(r[c])
                if not np.isnan(m):
                    pts.append((float(r["g"]), float(c[2:]), m, lo, hi))
        pts.sort(key=lambda p: (p[0], p[1]))
        x = np.arange(len(pts))
        m_, lo_, hi_ = (np.array([p[i] for p in pts]) for i in (2, 3, 4))
        a.errorbar(x, m_, yerr=[m_ - lo_, hi_ - m_], fmt="o", ms=4, capsize=2, color=color("connectome"))
        a.set_xticks(x)
        a.set_xticklabels([f"g={p[0]}\nf={p[1]}" for p in pts], fontsize=6)
        a.set_ylim(-0.02, 1.02)
        a.set_ylabel("success")
        a.set_title(title, fontsize=8)
    savefig(fig, "fig_plasticity")


def fig_layouts():
    """Independent maze layouts (unit = layout), shaped and binary-feedback baselines."""
    s = pd.concat([read_table("layouts_summary"), read_table("layouts_binary_summary")], ignore_index=True)
    s = s.drop_duplicates(["size", "label"])
    order = ["qlearning", "qlearning [binary]", "mlp", "mlp [binary]", "connectome [gate_error]",
             "connectome [gate_rank]", "kc_delta [diagnostic]", "rand_delta [diagnostic]"]
    labs = [l for l in order if l in set(s["label"])]
    sizes, mk = ["maze72", "maze113"], {"maze72": "o", "maze113": "s"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    for a, met, yl in zip(ax, ("final20", "final_policy_acc", "collapsed_f"),
                          ("success, last 20 episodes", "greedy-policy accuracy (all states)",
                           "fraction of runs that collapsed")):
        w = 0.8 / len(sizes)
        for j, sz in enumerate(sizes):
            for i, lab in enumerate(labs):
                sub = s[(s["size"] == sz) & (s["label"] == lab)]
                if sub.empty:
                    continue
                m, lo, hi = parse(sub[met].iloc[0])
                if np.isnan(m):
                    continue
                a.errorbar(i - 0.4 + w * (j + 0.5), m, yerr=[[max(m - lo, 0)], [max(hi - m, 0)]], fmt=mk[sz], ms=4,
                           capsize=2, color=color(lab))
        a.set_xticks(range(len(labs)))
        a.set_xticklabels(labs, rotation=40, ha="right", fontsize=6)
        a.set_ylabel(yl)
    ax[0].legend(handles=[Line2D([], [], marker=mk[sz], color="k", ls="", label=sz.replace("maze", "") + " states")
                          for sz in sizes], fontsize=6)
    ax[0].set_title("mean over 10 layouts, 95% CI over layouts", fontsize=8)
    savefig(fig, "fig_layouts")


def fig_gridworld_binary():
    """Baselines given the connectome's binary feedback, by maze size (main experiment mazes)."""
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    cols = [("qlearning", "qlearning"), ("qlearning [binary]", "qlearning [binary]"), ("mlp", "mlp"),
            ("mlp [binary]", "mlp [binary]"), ("connectome [gate_error]", "connectome [gate_error]"),
            ("connectome [gate_rank]", "connectome [gate_rank]")]
    for a, met, yl in zip(ax, ("overall_success", "final20", "final_policy_acc"),
                          ("success over the whole run", "success, last 20 episodes", "greedy-policy accuracy (all states)")):
        t = read_table(f"gridworld_binary_{met}")
        groups(a, t, sort_tasks(t.task), cols, ylab=yl)
        a.set_ylim(-0.02, 1.02)
    ax[0].legend(fontsize=6, loc="lower left")
    ax[0].set_title("shaped = r + d(s) - 0.95 d(s'); binary = +1/-1 (connectome's signal)", fontsize=7)
    savefig(fig, "fig_gridworld_binary")


def fig_disjoint():
    """Main comparison and addition results when evaluation items are excluded from training."""
    cmp_t, add_t = read_table("disjoint_cmp_final_acc"), read_table("disjoint_addition_final_acc")
    cmp_cols = [("connectome [fair_main]", "connectome (main, overlap allowed)"), ("connectome", "connectome (disjoint)"),
                ("mlp", "mlp"), ("linear", "linear"), ("connectome_notrain [control]", "untrained connectome")]
    add_cols = [("connectome", "connectome"), ("connectome [gate_error]", "connectome [gate_error]"),
                ("mlp [raw]", "mlp"), ("linear [raw]", "linear"), ("mlp [feat]", "mlp [oracle features]"),
                ("linear [feat]", "linear [oracle features]"), ("connectome_notrain [control]", "untrained connectome")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    groups(ax[0], cmp_t, sort_tasks(cmp_t.task), cmp_cols, ylab="held-out accuracy", ref=0.5)
    ax[0].set_title("number comparison, evaluation pairs excluded from training", fontsize=8)
    groups(ax[1], add_t, sort_tasks(add_t.task), add_cols, ref=0.5)
    ax[1].set_title("addition, held-out operand pairs", fontsize=8)
    for a in ax:
        a.legend(fontsize=6)
    savefig(fig, "fig_disjoint")


def fig_eval_overlap():
    """How many evaluation items also occurred in the online training stream of the main experiments."""
    c, a_ = read_table("eval_overlap_comparison"), read_table("eval_overlap_addition")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4))
    for a, tab, met, title in ((ax[0], c, "frac_exact", "number comparison (same orientation)"),
                               (ax[1], a_, "frac_all", "addition (all triples)")):
        tasks, ns = sort_tasks(tab.task), sorted(tab.n_train.unique())
        w, seen = 0.8 / len(ns), set()
        for j, n in enumerate(ns):
            sub = tab[tab.n_train == n]
            lab = f"{n} training items"
            for i, t in enumerate(tasks):
                m, lo, hi = _cell(sub, "task", t, met)
                if not np.isnan(m):
                    _bar(a, i - 0.4 + w * (j + 0.5), m, lo, hi, lab, seen)
        a.set_xticks(range(len(tasks)))
        a.set_xticklabels(tasks, rotation=30, ha="right")
        a.set_title(title, fontsize=8)
        a.legend(fontsize=6)
    ax[0].set_ylabel("fraction of evaluation items seen in training")
    savefig(fig, "fig_eval_overlap")


def fig_tictactoe_roles():
    """Greedy play against the heuristic opponent, by role."""
    t = read_table("tictactoe_roles")
    labs = list(t["label"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for a, role in zip(ax, ("first", "second")):
        w, seen = 0.8 / 3, set()
        for j, out in enumerate(("win", "draw", "loss")):
            for i, lab in enumerate(labs):
                m, lo, hi = _cell(t, "label", lab, f"{role}_{out}")
                if not np.isnan(m):
                    _bar(a, i - 0.4 + w * (j + 0.5), m, lo, hi, out, seen)
        a.set_xticks(range(len(labs)))
        a.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
        a.set_title(f"agent moves {role}", fontsize=8)
    ax[0].set_ylabel("rate against heuristic opponent (200 games)")
    ax[0].legend(fontsize=6)
    savefig(fig, "fig_tictactoe_roles")


def fig_dan_kc():
    """Effect of the dopaminergic drive on KC activity and plasticity; ablation with an odour-only plasticity mask."""
    t = read_table("dan_kc_measure")
    tasks = sort_tasks(t.task)
    conds = [("reward", "reward drive"), ("punish", "punishment drive"), ("as_trained", "as trained")]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.7))
    for a, met, yl in ((ax[0], "jaccard", "Jaccard overlap of the top-k KC sets"),
                       (ax[1], "frac_kc_changed", "fraction of KCs whose activity changes")):
        w, seen = 0.8 / len(conds), set()
        for j, (cn, lab) in enumerate(conds):
            sub = t[t.condition == cn]
            for i, tk in enumerate(tasks):
                m, lo, hi = _cell(sub, "task", tk, met)
                if not np.isnan(m):
                    _bar(a, i - 0.4 + w * (j + 0.5), m, lo, hi, lab, seen)
        a.set_xticks(range(len(tasks)))
        a.set_xticklabels(tasks, rotation=30, ha="right")
        a.set_ylabel(yl)
        a.set_title("DAN-driven vs odour-only forward pass", fontsize=8)
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].legend(fontsize=6)
    ab = read_table("dan_kc_ablation_acc")
    groups(ax[2], ab, sort_tasks(ab.task), [("connectome [native]", "native update"),
                                           ("connectome [odor_mask]", "odour-only plasticity mask")],
           ylab="held-out accuracy", ref=0.5)
    ax[2].set_title("addition: native update vs odour-only mask", fontsize=8)
    ax[2].legend(fontsize=6)
    savefig(fig, "fig_dan_kc")


def fig_bandit_punish():
    """Four-arm bandit: reward-only versus reward-and-punishment feedback (balanced ALPN groups, min_pulls = 10)."""
    reg, opt = read_table("bandit_punish_cum_pseudo_regret"), read_table("bandit_punish_frac_optimal_last50")
    sh = read_table("bandit_punish_choice_share")
    tasks = [t for t in sort_tasks(reg.task) if t.endswith("_mp10")]
    cols = [("qlearning", "qlearning"), ("mlp", "mlp"), ("connectome [balanced_reward]", "reward-only connectome"),
            ("connectome [balanced_punish]", "reward+punish connectome")]
    names = [t.replace("_mp10", "") for t in tasks]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a, tab, yl in ((ax[0], reg, "cumulative pseudo-regret (log)"), (ax[1], opt, "fraction of last 50 pulls on the best arm")):
        groups(a, tab, tasks, cols, ylab=yl)
        a.set_xticklabels(names, rotation=30, ha="right")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=6)
    w, seen = 0.8 / 2, set()
    for j, (cn, lab) in enumerate(cols[2:]):
        sub = sh[sh["label"] == cn]
        for i, tk in enumerate(tasks):
            m, lo, hi = _cell(sub, "task", tk, "share_rank0_last200")
            if not np.isnan(m):
                _bar(ax[2], i - 0.4 + w * (j + 0.5), m, lo, hi, lab, seen)
    ax[2].set_xticks(range(len(tasks)))
    ax[2].set_xticklabels(names, rotation=30, ha="right")
    ax[2].set_ylabel("share of last 200 pulls on the best arm")
    ax[2].set_title("min_pulls = 10; rev = best arm first, perm = random order", fontsize=7)
    savefig(fig, "fig_bandit_punish")


def main():
    for fn in (fig_bandit, fig_bandit_perm, fig_comparison, fig_wiring, fig_gridworld_scaling,
               fig_gridworld_curves, fig_tictactoe, fig_addition,
               fig_readouts, fig_constrained, fig_capacity, fig_plasticity, fig_layouts, fig_gridworld_binary,
               fig_disjoint, fig_eval_overlap, fig_tictactoe_roles, fig_dan_kc, fig_bandit_punish):
        try:
            fn()
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
