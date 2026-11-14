"""python -m analysis.punish_tables"""
import pandas as pd

from analysis.common import load, cells, wide, paired_table, save_table
from analysis.validate import read_manifests

SHARES = ["share_rank0_last200", "share_rank1_last200", "share_rank2_last200", "share_rank3_last200"]


def main():
    m0 = read_manifests("bandit_punish")
    print("status:", m0.status.value_counts().to_dict(), "| commits:", m0.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m0.repo_dirty.fillna(False).sum()))
    new = load("bandit_punish")
    old = load("bandit_perm")
    old = old[old.task.isin(new.task.unique()) &
              (((old.agent == "connectome") & (old.tier == "balanced")) | old.agent.isin(["qlearning", "mlp"]))].copy()
    d = pd.concat([new, old], ignore_index=True)
    d["pick_best"] = (d.top_arm == d.best_arm).astype(float)

    j = new[new.tier == "balanced_reward"][["task", "seed", "cum_pseudo_regret"]].merge(
        old[(old.agent == "connectome")][["task", "seed", "cum_pseudo_regret"]], on=["task", "seed"], suffixes=("_new", "_perm"))
    print(f"\nreproduction check vs runs/bandit_perm ({len(j)} matched runs): max |regret difference| = "
          f"{float((j.cum_pseudo_regret_new - j.cum_pseudo_regret_perm).abs().max()):.3g} (must be 0)")

    mets = ["cum_pseudo_regret", "frac_optimal_last50", "pick_best", "frac_best_all"]
    c = cells(d, ["task", "label"], mets, 2)
    for met in mets:
        save_table(wide(c, "task", met), f"bandit_punish_{met}", f"4-armed bandit, reward-only vs reward+punishment: {met}")
    save_table(cells(new, ["task", "label"], SHARES, 2).drop(columns="n"), "bandit_punish_choice_share",
               "Share of the last 200 pulls going to the best / 2nd / 3rd / worst arm (rank0 = best)", 2)
    comps = [("connectome [punish] - connectome [reward-only]", ("connectome", "balanced_punish"), ("connectome", "balanced_reward")),
             ("connectome [punish] - qlearning", ("connectome", "balanced_punish"), ("qlearning", "-")),
             ("connectome [reward-only] - qlearning", ("connectome", "balanced_reward"), ("qlearning", "-")),
             ("connectome [punish] - mlp", ("connectome", "balanced_punish"), ("mlp", "-"))]
    for met in ("cum_pseudo_regret", "frac_optimal_last50"):
        save_table(paired_table(d, met, comps), f"bandit_punish_paired_{met}",
                   f"4-armed bandit: paired differences in {met} (matched on task, seed)", 3)


if __name__ == "__main__":
    main()
