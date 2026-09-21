"""Exact minimax solver for tic-tac-toe -- the ground-truth optimal policy,
used to generate labeled (state -> correct move) training data for the
imitation-learning comparison (train_imitation_tictactoe.py): instead of
learning through trial-and-error reward signals (the RL approach used
everywhere else in this project), the agent is just shown the correct
answer for every state directly.

Negamax with memoization -- board space is small enough (~5,478 reachable
states) that this solves instantly and exactly, no pruning needed.
"""
from functools import lru_cache

from envs.tictactoe_env import winner_of


@lru_cache(maxsize=None)
def minimax(board, player):
    """board: length-9 tuple, RAW marks (+1/-1/0, NOT perspective-shifted).
    player: whose turn it is (+1 or -1). Returns (value, best_action) from
    `player`'s perspective: value +1 win / 0 draw / -1 loss, both sides
    playing optimally from here on. best_action is one optimal move
    (there may be several tied-optimal moves; this returns whichever
    negamax happens to find first)."""
    w = winner_of(board)
    if w is not None:
        if w == 0:
            return 0, None
        return (1 if w == player else -1), None

    best_a, best_v = None, -2
    for a in range(9):
        if board[a] != 0:
            continue
        b = list(board)
        b[a] = player
        v, _ = minimax(tuple(b), -player)
        v = -v  # negate: that value was from the OTHER player's perspective
        if v > best_v:
            best_v, best_a = v, a
    return best_v, best_a


def optimal_action(board, player):
    """board: RAW marks. player: whose turn (+1/-1). Convenience wrapper
    when you only need the action, not the value."""
    _, a = minimax(board, player)
    return a


def enumerate_reachable_states():
    """All (raw_board, mover) pairs where the game is NOT yet over -- i.e.
    every position a real game could pause at and ask 'whose move is it,
    and what should they play'. Depth-first from the empty board."""
    seen = set()
    states = []

    def rec(board, mover):
        key = (board, mover)
        if key in seen:
            return
        seen.add(key)
        if winner_of(board) is not None:
            return
        states.append(key)
        for a in range(9):
            if board[a] == 0:
                b = list(board)
                b[a] = mover
                rec(tuple(b), -mover)

    rec((0,) * 9, 1)
    return states


if __name__ == "__main__":
    states = enumerate_reachable_states()
    print(f"{len(states)} reachable non-terminal states")
    val, _ = minimax((0,) * 9, 1)
    print(f"game value from an empty board, X to move: {val} (0 = draw with optimal play, as expected)")
