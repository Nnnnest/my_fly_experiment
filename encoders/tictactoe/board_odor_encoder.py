"""Compositional board -> ALPN-vector encoder for the tic-tac-toe
afterstate agents.

Why compositional and not one-hot-per-board (as grid world used for
(state, action) pairs): tic-tac-toe has ~5,478 reachable states, far more
than the ~302-468 live ALPN indices available (hops=1/2) -- there's no
room for a unique pattern per board. Instead, each of the 9 cells gets its
own dedicated slice of the live-index budget, and that slice's activation
encodes just that cell's mark (+1 self / -1 opponent / 0 empty). A full
board is the sum of its 9 cells' slices -- this reuses the same 9 slices
for every board instead of needing one slice per board.

CELL_DRIVE and GROUP_SIZE are NOT validated against the real network yet --
run tictactoe_diagnostic.py and adjust them based on what it reports before
trusting any experiment results.
"""
import os

import numpy as np
import itertools

N_CELLS = 9
N_ALPN_TOTAL = 685  # mb mode's ALPN count, per README/summary.txt

_HERE = os.path.dirname(os.path.abspath(__file__))
_MY_EXPERIMENTS =  os.path.join(os.path.dirname(__file__), "..", "..", "..")         # my_experiments/
_REPO_ROOT = os.path.dirname(_MY_EXPERIMENTS)     # one level up, in case it lives there instead
_CANDIDATES = [
    os.path.join(_MY_EXPERIMENTS, "results", "gridworld", "live_alpn_indices.npy"),
    os.path.join(_REPO_ROOT, "results", "gridworld", "live_alpn_indices.npy"),
]

for _candidate in _CANDIDATES:
    if os.path.exists(_candidate):
        LIVE_ALPN = np.load(_candidate)
        break
else:
    raise FileNotFoundError(
        "couldn't find live_alpn_indices.npy in either of:\n  "
        + "\n  ".join(_CANDIDATES)
        + "\n(the same file the bandit/grid-world agents use, hops=1). "
        "If it lives somewhere else, edit _CANDIDATES above."
    )

GROUP_SIZE = len(LIVE_ALPN) // N_CELLS
CELL_GROUPS = [LIVE_ALPN[i * GROUP_SIZE:(i + 1) * GROUP_SIZE] for i in range(N_CELLS)]

# Starting guess only -- tune against tictactoe_diagnostic.py's output
# (innate-bias spread and cross-board interference delta).
CELL_DRIVE = 1.0


def board_to_vector(board, n_alpn=N_ALPN_TOTAL):
    """board: length-9 sequence, already from the acting player's own
    perspective (+1 self, -1 opponent, 0 empty) -- i.e. the output of
    env.board_from_perspective(mark)."""
    vec = np.zeros(n_alpn)
    for cell_idx, mark in enumerate(board):
        if mark != 0:
            vec[CELL_GROUPS[cell_idx]] = mark * CELL_DRIVE
    return vec


def board_baseline_features(board):
    """9 raw marks + all C(9,2)=36 pairwise products, for fitting the
    connectome's untrained-MB_pref baseline model (used by
    calibrate_baseline() in connectome_tictactoe_agent.py and by the R^2
    check in tictactoe_diagnostic.py -- kept in one place so both use the
    exact same feature definition).

    Deliberately NOT the 685-dim board_to_vector: that vector is constant
    within each of the 9 cell groups, so a linear fit on it is
    mathematically equivalent to fitting on just the 9 raw marks -- more
    parameters than degrees of freedom, and no way to represent any
    interaction between cells. The pairwise products let the baseline
    model capture cell-cell interactions (e.g. two marks on the same line)
    that the recurrent nonlinear forward pass likely produces, without
    needing a much larger calibration sample."""
    marks = list(board)
    feats = list(marks)
    for i, j in itertools.combinations(range(9), 2):
        feats.append(marks[i] * marks[j])
    return np.array(feats, dtype=np.float64)
