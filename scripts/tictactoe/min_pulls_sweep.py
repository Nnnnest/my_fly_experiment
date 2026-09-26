"""Sweep min_pulls for Tier 1-opt (symmetry + cache) against the heuristic
opponent, to test the untested hypothesis in notes/summary_addendum.txt
section 7: that symmetry's 0% heuristic-opponent win rate (vs Tier 1
fair's small-but-nonzero 0.002-0.092) comes from the forced-exploration
floor being satisfied much faster once canonical-state collapse packs
more visits per distinct board -- letting all three agents exploit
prematurely against a fully deterministic opponent, with no varied
counter-examples available to correct from.

If the hypothesis is right, raising min_pulls should recover SOME
draw/win rate as min_pulls grows; if it's flat at 0% regardless, the
explanation is something else.

    python min_pulls_sweep.py
    python min_pulls_sweep.py --min-pulls 5 10 20 40 --episodes 1000

NOTE: assumes agents/tictactoe/connectome_tictactoe_agent_v2.py exposes a
class named ConnectomeTicTacToeAgentV2(min_pulls=...) matching the other
two agents' constructor shape (same as Tier 1-opt's own construction
elsewhere in the project) -- adjust the import/class name below if it
differs.
"""
import argparse
import csv
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR), os.path.dirname(os.path.dirname(_THIS_DIR))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.tictactoe.qlearning_tictactoe_agent import QLearningTicTacToeAgent
from agents.tictactoe.mlp_tictactoe_agent import MLPTicTacToeAgent
from agents.tictactoe.connectome_tictactoe_agent_v2 import ConnectomeTicTacToeAgentV2
from agents.shared.symmetry_wrapper import SymmetryWrapper
from opponents.opponents import heuristic_opponent
from run_tictactoe import run_block, sample_early_boards

RESULTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "..", "results", "tictactoe")
N_EPISODES = 1000
MIN_PULLS_VALUES = [5, 10, 20, 40]


def outcome_rates(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    wins = sum(int(r["outcome"]) == 1 for r in rows)
    draws = sum(int(r["outcome"]) == 0 for r in rows)
    losses = n - wins - draws
    tail = rows[-100:]
    last_win = sum(int(r["outcome"]) == 1 for r in tail) / max(1, len(tail))
    last_draw = sum(int(r["outcome"]) == 0 for r in tail) / max(1, len(tail))
    return dict(win=wins / n, draw=draws / n, loss=losses / n,
                last100_win=last_win, last100_draw=last_draw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pulls", type=int, nargs="+", default=MIN_PULLS_VALUES)
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    args = ap.parse_args()

    calibration_boards = sample_early_boards(300)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "min_pulls_sweep.csv")

    with open(summary_path, "w", newline="") as sf:
        writer = csv.writer(sf)
        writer.writerow(["agent", "min_pulls", "win", "draw", "loss", "last100_win", "last100_draw"])

        for mp in args.min_pulls:
            print(f"=== min_pulls={mp} ===")
            agents = {
                "qlearning_sym": SymmetryWrapper(QLearningTicTacToeAgent(min_pulls=mp)),
                "mlp_sym": SymmetryWrapper(MLPTicTacToeAgent(min_pulls=mp)),
                "connectome_sym_cache": SymmetryWrapper(ConnectomeTicTacToeAgentV2(min_pulls=mp)),
            }
            if hasattr(agents["connectome_sym_cache"], "calibrate_baseline"):
                agents["connectome_sym_cache"].calibrate_baseline(calibration_boards)

            for name, agent in agents.items():
                tag = f"minpulls{mp}"
                run_block(agent, name, heuristic_opponent, f"heuristic_{tag}",
                          n_episodes=args.episodes)
                path = os.path.join(RESULTS_DIR, f"tictactoe_{name}_heuristic_{tag}.csv")
                rates = outcome_rates(path)
                writer.writerow([name, mp, rates["win"], rates["draw"], rates["loss"],
                                 rates["last100_win"], rates["last100_draw"]])
                sf.flush()
                print(f"  {name}: win={rates['win']:.3f} draw={rates['draw']:.3f} "
                      f"last100_win={rates['last100_win']:.3f} last100_draw={rates['last100_draw']:.3f}")

    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
