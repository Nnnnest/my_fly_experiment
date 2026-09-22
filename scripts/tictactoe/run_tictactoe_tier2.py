import os, sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", ".."))

from agents.tictactoe.connectome_tictactoe_agent_v2 import ConnectomeTicTacToeAgentV2
from agents.shared.symmetry_wrapper import SymmetryWrapper
from opponents.opponents import random_opponent, heuristic_opponent
from run_tictactoe import sample_early_boards, run_block, run_self_play


def main():
    calibration_boards = sample_early_boards(300)
    opponents = {"random": random_opponent, "heuristic": heuristic_opponent}

    for gate in ("error", "visit"):
        name = f"connectome_t2_{gate}"
        agent = SymmetryWrapper(ConnectomeTicTacToeAgentV2(gate=gate))
        agent.calibrate_baseline(calibration_boards)
        for opp_name, opp_fn in opponents.items():
            run_block(agent, name, opp_fn, opp_name)
        run_self_play(lambda g=gate: SymmetryWrapper(ConnectomeTicTacToeAgentV2(gate=g)),
                       name, calibration_boards=calibration_boards)


if __name__ == "__main__":
    main()
