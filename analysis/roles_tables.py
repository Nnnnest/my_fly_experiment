"""python -m analysis.roles_tables"""
import pandas as pd

from analysis.common import load, cells, save_table
from analysis.validate import read_manifests


def main():
    m0 = read_manifests("tictactoe_roles")
    print("status:", m0.status.value_counts().to_dict(), "| commits:", m0.repo_commit.value_counts().to_dict(),
          "| dirty:", int(m0.repo_dirty.fillna(False).sum()))
    r = load("tictactoe_roles")
    main_runs = load("tictactoe")
    main_runs = main_runs[main_runs.task == "rl_heuristic"]

    keys = ["eval_heuristic_win", "eval_heuristic_draw", "eval_heuristic_loss", "coverage_acc", "last100_win"]
    j = main_runs[["agent", "tier", "seed"] + keys].merge(
        r[["agent", "tier", "seed"] + keys], on=["agent", "tier", "seed"], suffixes=("_main", "_roles"))
    print(f"\nreproduction check ({len(j)} matched runs): max |main - replay| per key (must be 0)")
    for k in keys:
        print(f"  {k}: {float((j[k + '_main'] - j[k + '_roles']).abs().max()):.3g}")

    cols = ["first_win", "first_draw", "first_loss", "second_win", "second_draw", "second_loss"]
    save_table(cells(r, ["label"], cols, 3).drop(columns="n"), "tictactoe_roles",
               "Greedy evaluation against the heuristic opponent by role (200 games per role), mean of 10 seeds")
    g = r[r.label == "connectome [gate_visit]"].sort_values("seed")
    print("\nconnectome [gate_visit] per seed:")
    print(g[["seed", "first_move_greedy", "second_reply_greedy"] + cols].round(3).to_string(index=False))
    sig = ((r.first_draw > 0.95) & (r.second_loss > 0.95)) | ((r.second_draw > 0.95) & (r.first_loss > 0.95))
    print("\nseeds showing 'one role always draws, the other always loses' (>0.95 each), by label:")
    print(r.assign(sig=sig).groupby("label").sig.sum().astype(int).to_string())


if __name__ == "__main__":
    main()
