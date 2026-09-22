import os, sys, csv

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_THIS_DIR, "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts", "diagnostics"))  # tictactoe_coverage_eval.py
sys.path.insert(0, os.path.join(_ROOT, "scripts", "tictactoe"))     # tictactoe_solver.py

from agents.tictactoe.connectome_tictactoe_agent import ConnectomeTicTacToeAgent
from agents.shared.symmetry_wrapper import SymmetryWrapper
from tictactoe_coverage_eval import full_coverage_accuracy
from tictactoe_solver import enumerate_reachable_states

MODELS_DIR = os.path.join(_ROOT, "models", "tictactoe")
OUT_PATH = os.path.join(_ROOT, "results", "tictactoe", "tier3_imitation_summary.csv")

# tag -> was this model trained through SymmetryWrapper?
# full_coverage_accuracy does NOT canonicalize internally (confirmed: it
# calls agent.best_action on the raw board) -- a symmetry-trained model
# MUST be wrapped here too, or every value() call sees an orientation it
# was never trained on. Previous version of this script did NOT do this.
TAGS = {
    "connectome_imitation": False,
    "connectome_imitation_shuffle": False,
    "connectome_imitation_sleep500": False,
    "connectome_imitation_symmetry": True,
    "connectome_imitation_shuffle_symmetry": True,
}


def main():
    states = enumerate_reachable_states()
    rows = []
    for tag, used_symmetry in TAGS.items():
        prefix = os.path.join(MODELS_DIR, tag)
        if not os.path.exists(prefix + "_weights.npz"):
            continue
        base = ConnectomeTicTacToeAgent()
        base.load(prefix)
        agent = SymmetryWrapper(base) if used_symmetry else base
        acc, correct, total = full_coverage_accuracy(agent, states)
        rows.append((tag, round(acc, 1), correct, total))
        print(f"{tag} (symmetry={used_symmetry}): {correct}/{total} = {acc:.1f}%")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "full_coverage_accuracy_pct", "correct", "total"])
        w.writerows(rows)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
