"""Targeted rerun: connectome agent only, with periodic fly.sleep() calls
every SLEEP_EVERY episodes, to test whether the library's own homeostatic
downscaling counteracts the win-rate decay seen in the first full run
(win_pct_last100 well below win_pct overall across all three opponent
types -- see results/tictactoe_summary.csv). Suspected mechanism: the same
one documented for the 113-state/hops=2 grid-world maze breakdown --
non-decaying, whole-graph training updates drift weights faster than the
agent can consolidate, compounded here by tic-tac-toe touching a much
larger board population per run than the maze did.

Writes tictactoe_connectome_<opponent>_sleep100.csv and
tictactoe_connectome_selfplay_sleep100.csv alongside (not overwriting) the
original connectome results, each with a drift_pct column, so
results/view_tictactoe.py picks both runs up for direct comparison --
win-rate curve AND drift-over-time side by side.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")

from agents.gridworld.connectome_tictactoe_agent import ConnectomeTicTacToeAgent
from opponents.opponents import random_opponent, heuristic_opponent
from run_tictactoe import run_block, run_self_play, sample_early_boards

SLEEP_EVERY = 100


def main():
    calibration_boards = sample_early_boards(300)

    agent = ConnectomeTicTacToeAgent()
    agent.calibrate_baseline(calibration_boards)
    for opponent_name, opponent_fn in [("random", random_opponent), ("heuristic", heuristic_opponent)]:
        run_block(agent, "connectome", opponent_fn, opponent_name, sleep_every=SLEEP_EVERY)

    run_self_play(ConnectomeTicTacToeAgent, "connectome",
                   calibration_boards=calibration_boards, sleep_every=SLEEP_EVERY)


if __name__ == "__main__":
    main()
