"""python -m analysis.tables"""
import os
import traceback

import pandas as pd

from core import paths
from core.load import load_summaries
from analysis.common import *


def bandit():
    d = load("bandit")
    h = d[d.task == "mp10"]
    t = cells(h, ["label"], ["total_reward", "cum_pseudo_regret", "cum_regret", "frac_optimal_last50"], 2)
    save_table(t.drop(columns="n"), "bandit_headline", "Bandit, min_pulls=10: mean [95% bootstrap CI], 10 seeds")
    c = cells(d, ["task", "label"], ["cum_pseudo_regret"], 2)
    save_table(wide(c, "task", "cum_pseudo_regret"), "bandit_sweep", "Cumulative pseudo-regret by forced-exploration floor (min_pulls)")
    comps = [("connectome - qlearning", ("connectome", "-"), ("qlearning", "-")),
             ("connectome - mlp", ("connectome", "-"), ("mlp", "-"))]
    save_table(paired_table(d, "cum_pseudo_regret", comps), "bandit_paired", "Paired differences in cumulative pseudo-regret (negative = connectome better)", 3)


def comparison():
    d = load("comparison")
    c = cells(d, ["task", "label"], ["eval_acc_final", "eval_acc_pre"], 3)
    save_table(wide(c, "task", "eval_acc_final"), "comparison_final_acc", "Number comparison: held-out accuracy after training")
    save_table(wide(c, "task", "eval_acc_pre"), "comparison_pre_acc", "Number comparison: held-out accuracy before training")
    comps = [("connectome - mlp", ("connectome", "fair"), ("mlp", "fair")),
             ("connectome - linear", ("connectome", "fair"), ("linear", "fair")),
             ("connectome[sleep50] - connectome", ("connectome", "sleep50"), ("connectome", "fair")),
             ("connectome - untrained connectome", ("connectome", "fair"), ("connectome_notrain", "control"))]
    save_table(paired_table(d, "eval_acc_final", comps), "comparison_paired", "Paired differences in held-out accuracy", 3)


def gridworld():
    d = load("gridworld")
    d["collapsed_f"] = d.collapsed.astype(float)
    c = cells(d, ["task", "label"], ["overall_success", "final20", "path_efficiency", "final_policy_acc"], 3)
    for m, cap in (("overall_success", "success over the whole run"), ("final20", "success in the last 20 episodes"),
                   ("path_efficiency", "path efficiency (optimal/actual steps, successes only)"),
                   ("final_policy_acc", "greedy-policy accuracy over all states at the end")):
        save_table(wide(c, "task", m), f"gridworld_{m}", f"Gridworld: {cap}")
    r = d.groupby(["task", "label"]).agg(k=("collapsed_f", "sum"), n=("seed", "count")).reset_index()
    r["cell"] = r.k.astype(int).astype(str) + "/" + r.n.astype(str)
    save_table(wide(r, "task", "cell"), "gridworld_collapse", "Gridworld: runs that peaked >=0.5 then fell by >0.3 (of 10)")
    t = d[d.n_trained.notna()].groupby(["task", "label"]).n_trained.median().reset_index()
    save_table(wide(t, "task", "n_trained"), "gridworld_trains", "Median number of train() calls / updates")
    ref = ("qlearning", "fair")
    others = [("connectome", "fair"), ("connectome", "opt"), ("connectome", "gate_visit"), ("connectome", "gate_error"),
              ("connectome", "gate_rank"), ("mlp", "fair"), ("kc_delta", "diagnostic"), ("rand_delta", "diagnostic")]
    comps = [(f"{label(*o)} - qlearning", o, ref) for o in others]
    comps += [("connectome [gate_rank] - connectome [opt]", ("connectome", "gate_rank"), ("connectome", "opt")),
              ("kc_delta - rand_delta (does the wiring help?)", ("kc_delta", "diagnostic"), ("rand_delta", "diagnostic"))]
    comps += [("connectome [gate_error] - connectome [opt]", ("connectome", "gate_error"), ("connectome", "opt")),
              ("connectome [gate_visit] - connectome [opt]", ("connectome", "gate_visit"), ("connectome", "opt"))]
    comps += [("kc_delta - rand_delta, alpha 0.2 (best at 316)", ("kc_delta", "diagnostic_a0.2"), ("rand_delta", "diagnostic_a0.2"))]
    save_table(paired_table(d, "final20", comps), "gridworld_paired", "Paired differences in last-20-episode success", 3)
    save_table(paired_table(d, "overall_success", comps), "gridworld_paired_overall",
               "Paired differences in success over the whole run", 3)
    d["never_f"] = (d.final20 < 0.5).astype(float)
    f = d.groupby(["task", "label"]).agg(k=("never_f", "sum"), n=("seed", "count")).reset_index()
    f["cell"] = f.k.astype(int).astype(str) + "/" + f.n.astype(str)
    save_table(wide(f, "task", "cell"), "gridworld_failed_final", "Gridworld: runs ending below 0.5 success in the last 20 episodes (of 10)")


def tictactoe():
    d = load("tictactoe")
    rl = d[d.task.str.startswith("rl_")].copy()
    mets = ["eval_random_win", "eval_heuristic_draw", "eval_heuristic_loss", "coverage_acc", "last100_win", "last100_draw"]
    for task, g in rl.groupby("task"):
        c = cells(g, ["label"], mets, 3)
        save_table(c.drop(columns="n"), f"tictactoe_{task}", f"Tic-tac-toe, {task}: greedy evaluation after training, minimax coverage, last-100 training win rate")
    r = rl[rl.task != "rl_selfplay"].assign(cf=rl.collapsed.astype(float)).groupby(["task", "label"]).agg(k=("cf", "sum"), n=("seed", "count")).reset_index()
    r["cell"] = r.k.astype(int).astype(str) + "/" + r.n.astype(str)
    save_table(wide(r, "task", "cell"), "tictactoe_collapse",
               "Tic-tac-toe: collapsed training runs (of 10). Self-play excluded: a drop in one learner's win rate can be the other learner improving.")
    im = d[d.task == "imitation"]
    save_table(cells(im, ["label"], ["coverage_acc_pre", "coverage_acc"], 3).drop(columns="n"), "tictactoe_imitation",
               "Imitation: minimax-optimal move coverage over all 4520 reachable states")
    dr = d[d.agent == "connectome"]
    save_table(wide(cells(dr, ["task", "label"], ["final_drift_pct"], 2), "task", "final_drift_pct"), "tictactoe_drift", "Final weight drift (%)")
    comps = []
    for tier in ("fair", "opt", "sym"):
        comps += [(f"connectome - mlp ({tier})", ("connectome", tier), ("mlp", tier)),
                  (f"connectome - qlearning ({tier})", ("connectome", tier), ("qlearning", tier))]
    comps += [("connectome [opt] - connectome", ("connectome", "opt"), ("connectome", "fair"))]
    comps += [("connectome [gate_error] - connectome [opt]", ("connectome", "gate_error"), ("connectome", "opt")),
              ("connectome [gate_visit] - connectome [opt]", ("connectome", "gate_visit"), ("connectome", "opt")),
              ("connectome [opt_sleep100] - connectome [opt]", ("connectome", "opt_sleep100"), ("connectome", "opt"))]
    save_table(paired_table(d, "coverage_acc", comps), "tictactoe_paired", "Paired differences in minimax coverage accuracy", 3)


def addition():
    d = load("addition")
    c = cells(d, ["task", "label"], ["eval_acc_final", "eval_acc_pre"], 3)
    save_table(wide(c, "task", "eval_acc_final"), "addition_final_acc", "Addition: held-out accuracy after training (chance = 0.5)")
    save_table(wide(c, "task", "eval_acc_pre"), "addition_pre_acc", "Addition: held-out accuracy before training")
    k = ("connectome", "fair")
    comps = [("connectome - mlp [raw]", k, ("mlp", "raw")), ("connectome - linear [raw]", k, ("linear", "raw")),
             ("connectome - mlp [feat] (oracle features)", k, ("mlp", "feat")),
             ("connectome - linear [feat] (oracle features)", k, ("linear", "feat")),
             ("connectome [gate_error] - connectome", ("connectome", "gate_error"), k),
             ("connectome - untrained connectome", k, ("connectome_notrain", "control"))]
    save_table(paired_table(d, "eval_acc_final", comps), "addition_paired", "Paired differences in held-out accuracy", 3)
    g = load("addition_diag")
    cols = [c for c in g.columns if c.startswith("jaccard_diff")]
    long = g.melt(id_vars=["task", "seed"], value_vars=cols, var_name="name", value_name="jaccard")
    long["d"] = long.name.str.replace("jaccard_diff", "").astype(int)
    cc = cells(long, ["task", "d"], ["jaccard"], 3)
    save_table(wide(cc, "d", "jaccard", col="task"), "addition_kc_overlap", "Jaccard overlap of KC codes: correct triple vs c off by d (row d=0 is self-overlap)")
    save_table(cells(g, ["task"], ["jaccard_random_mean"], 3).drop(columns="n"), "addition_kc_random", "Random-triple baseline overlap")


TUNE_METRIC = {"comparison": "eval_acc_final", "gridworld": "overall_success",
               "tictactoe": "coverage_acc", "addition": "eval_acc_final", "bandit": "cum_pseudo_regret"}


def tuning():
    root, rows = os.path.join(paths.RUNS_DIR, "_tuning"), []
    bytask = []
    for exp, met in TUNE_METRIC.items():
        try:
            d = load_summaries(exp, root=root)
        except FileNotFoundError:
            continue
        d = d[d.status == "ok"]
        dg = d[d.task != "maze316_rs"] if exp == "gridworld" else d
        g = dg.groupby(["agent", "tier"])[met].agg(["mean", "count"]).reset_index()
        g.insert(0, "experiment", exp)
        g["metric"] = met
        rows.append(g)
        bytask.append(d.groupby(["agent", "tier", "task"])[met].mean().reset_index().assign(experiment=exp))
    save_table(pd.concat(rows), "tuning_grids",
               "Hyperparameter search on tuning seeds (mean over tasks and seeds; gridworld pooled over maze26/72/113, maze316 only in tuning_by_task)", 3)
    save_table(pd.concat(bytask), "tuning_by_task", "Tuning results by task", 3)


def main():
    for fn in (bandit, comparison, gridworld, tictactoe, addition, tuning):
        print(f"\n===================== {fn.__name__} =====================")
        try:
            fn()
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
