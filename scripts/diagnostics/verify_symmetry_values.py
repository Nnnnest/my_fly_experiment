"""Action path must agree with value path. python -m scripts.diagnostics.verify_symmetry_values"""
import random

from core import paths
from agents.tictactoe.qlearning_tictactoe_agent import QLearningTicTacToeAgent
from agents.shared.symmetry_wrapper import SymmetryWrapper
from agents.shared.tictactoe_solver import enumerate_reachable_states
from agents.shared.tictactoe_symmetry import canonicalize
from agents.shared.tictactoe_eval import to_perspective, afterstate

rng = random.Random(0)
inner = QLearningTicTacToeAgent()
w = SymmetryWrapper(inner)
states = enumerate_reachable_states()
rng.shuffle(states)
bad = n = 0
for raw, mover in states[:1500]:
    mine = to_perspective(raw, mover)
    legal = [a for a in range(9) if mine[a] == 0]
    for a in legal:
        key = canonicalize(afterstate(mine, a))[0]
        if key not in inner.V:
            inner.V[key] = rng.random()
            inner.visits[key] = inner.min_pulls
    vals = [w.value(afterstate(mine, a)) for a in legal]
    top = max(vals)
    for pick in (w.best_action(mine, legal), w.choose_action(mine, legal, 0.0)):
        n += 1
        bad += vals[legal.index(pick)] < top - 1e-12
print(f"mismatches: {bad}/{n}")
assert bad == 0
print("OK")
