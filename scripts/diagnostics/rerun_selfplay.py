import sys
import os

"""Rerun ONLY the self-play blocks for all three agents, after fixing a
role-assignment bug in run_self_play: agent instance `a1` was always
mark=+1 (always moved first / played X), `a2` always second -- no
alternation, unlike every other block in this project (run_block already
alternated who plays first each episode). That gave a1 a permanent
structural first-move advantage confounding every self-play result to
some degree -- most visibly in the connectome-with-sleep run, where a1's
win rate jumped to 88-99%, which looked like a striking "sleep fixes the
decay" result but is at least partly this bug rather than a genuine effect
of sleep() (reinforced by vs-random/vs-heuristic, unaffected by this bug,
showing sleep changed almost nothing there).

vs-random and vs-heuristic blocks do NOT need rerunning -- only self-play.
Overwrites the previous (confounded) tictactoe_*_selfplay*.csv files.
"""

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


from agents.qlearning_tictactoe_agent import QLearningTicTacToeAgent
from agents.mlp_tictactoe_agent import MLPTicTacToeAgent
from agents.connectome_tictactoe_agent import ConnectomeTicTacToeAgent
from run_tictactoe import run_self_play, sample_early_boards

SLEEP_EVERY = 100


def main():
    calibration_boards = sample_early_boards(300)

    for agent_name, agent_cls in [
        ("qlearning", QLearningTicTacToeAgent),
        ("mlp", MLPTicTacToeAgent),
        ("connectome", ConnectomeTicTacToeAgent),
    ]:
        cb = calibration_boards if agent_name == "connectome" else None
        run_self_play(agent_cls, agent_name, calibration_boards=cb)

    # corrected connectome-with-sleep self-play, for a clean comparison
    # against the corrected no-sleep connectome self-play above
    run_self_play(ConnectomeTicTacToeAgent, "connectome",
                   calibration_boards=calibration_boards, sleep_every=SLEEP_EVERY)


if __name__ == "__main__":
    main()
