"""Opponent policies for the tic-tac-toe experiment (spec.txt Stage 3,
Experiment 3: random player / pre-built algorithm / another learning
agent -- the third is self-play, handled directly in run_tictactoe.py
rather than here).

Signature for both: fn(board, legal_actions, mark) -> action
  board: env.board_from_perspective(mark) -- the board as this opponent
         sees it (its own marks are +1, the other player's are -1)
  mark:  which raw mark (+1/-1, i.e. X/O) this opponent is currently
         playing -- unused by both policies below since board is already
         perspective-normalized, kept in the signature for symmetry with
         the agents' choose_action() and in case a future opponent needs it
"""
import random

from envs.tictactoe_env import winner_of


def random_opponent(board, legal_actions, mark):
    return random.choice(legal_actions)


def heuristic_opponent(board, legal_actions, mark):
    """Win if possible, else block, else center, else corner, else random.
    Because board is already from this opponent's own perspective, 'win'
    always means checking for +1 and 'block' always means checking for -1,
    regardless of whether this opponent is actually X or O."""
    for a in legal_actions:
        b = list(board)
        b[a] = 1
        if winner_of(b) == 1:
            return a
    for a in legal_actions:
        b = list(board)
        b[a] = -1
        if winner_of(b) == -1:
            return a
    if 4 in legal_actions:
        return 4
    corners = [c for c in (0, 2, 6, 8) if c in legal_actions]
    if corners:
        return random.choice(corners)
    return random.choice(legal_actions)
