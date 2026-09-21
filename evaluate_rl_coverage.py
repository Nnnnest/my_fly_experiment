"""Evaluate the RL-trained connectome (from play_tictactoe.py's saved
model) on the exact same full-coverage check used for the imitation-
trained one (train_imitation_tictactoe.py), for a direct, apples-to-apples
comparison. This script only evaluates -- run play_tictactoe.py first so
models/connectome_* exists.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.connectome_tictactoe_agent import ConnectomeTicTacToeAgent
from tictactoe_coverage_eval import full_coverage_accuracy

MODELS_DIR = os.path.join(_THIS_DIR, "models")


def main():
    weights_path = os.path.join(MODELS_DIR, "connectome_weights.npz")
    baseline_path = os.path.join(MODELS_DIR, "connectome_baseline.npy")
    if not (os.path.exists(weights_path) and os.path.exists(baseline_path)):
        print("no RL-trained connectome found -- run play_tictactoe.py first "
              "(it trains and saves models/connectome_* on its first run).")
        return

    agent = ConnectomeTicTacToeAgent()
    agent.load(os.path.join(MODELS_DIR, "connectome"))

    acc, correct, total = full_coverage_accuracy(agent)
    print(f"RL-trained connectome full-coverage accuracy: {correct}/{total} = {acc:.1f}%")
    print("compare directly against train_imitation_tictactoe.py's printed accuracy "
          "for the imitation-trained connectome -- same states, same metric.")


if __name__ == "__main__":
    main()
