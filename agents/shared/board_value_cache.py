"""Exact fast value evaluator for FlyBrainAPI(mode="mb") at hops=1, generalized
from mb_value_cache.MBValueCache to a fixed list of arbitrary board vectors
(tic-tac-toe's afterstate formulation) instead of gridworld's
state*n_actions+action pair indexing. Same exactness argument as
MBValueCache: at hops=1 in mb mode, train() only touches KC->MBON weights,
so MB_pref is linear in those weights for any fixed ALPN-drive pattern --
verified against brain.step() the same way (see .verify()).
"""
import numpy as np
from agents.shared.mb_value_cache import MBValueCache


class _BoardEncoderAdapter:
    """Wraps a fixed (n_boards, n_alpn) pattern array so MBValueCache --
    which expects encoder.patterns / encoder.n_actions -- works unmodified.
    n_actions=1 makes values_state(board_id) == values_pairs([board_id])."""
    def __init__(self, patterns):
        self.patterns = patterns
        self.n_actions = 1


def enumerate_afterstates():
    """Every board the tic-tac-toe agent can ever call value()/train() on:
    one ply after any reachable non-terminal (board, mover) state, from the
    MOVER's own perspective (i.e. exactly board_from_perspective(mover) --
    what choose_action/update_episode actually produce). Includes TERMINAL
    (won/lost/drawn) afterstates, which enumerate_reachable_states()
    excludes but choose_action() does evaluate (a winning move must be
    visible)."""
    from envs.tictactoe_env import TicTacToe
    from tictactoe_solver import enumerate_reachable_states
    seen = set()
    for board, mover in enumerate_reachable_states():
        for a in range(9):
            if board[a] != 0:
                continue
            b = list(board)
            b[a] = mover
            seen.add(tuple(mover * x for x in b))  # board_from_perspective(mover)
    return list(seen)


def build_board_cache(brain, boards, board_to_vector, kc_frac=0.05):
    """boards: list of length-9 board tuples, already in whatever
    orientation you'll look values up with -- if you canonicalize with
    SymmetryWrapper, canonicalize THESE first and canonicalize every later
    lookup the same way, or ids won't match.
    Returns (cache, board_to_id)."""
    patterns = np.stack([board_to_vector(b) for b in boards]).astype(np.float32)
    cache = MBValueCache(brain, _BoardEncoderAdapter(patterns), kc_frac=kc_frac)
    return cache, {b: i for i, b in enumerate(boards)}
