"""Tic-tac-toe environment.

Board cells: 0 = empty, +1 = X, -1 = O. X always moves first. There is no
"illegal move" path in normal play -- step() is only ever called with an
action drawn from legal_actions() -- but it still raises if misused, since
a silent no-op would be worse for debugging than a crash.
"""
import numpy as np

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
]


def winner_of(board):
    """board: length-9 sequence of -1/0/+1. Returns +1, -1, 0 (draw), or
    None (game not over)."""
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    if 0 not in board:
        return 0
    return None


class TicTacToe:
    def __init__(self):
        self.board = None
        self.to_move = None
        self.reset()

    def reset(self):
        self.board = [0] * 9
        self.to_move = 1  # X starts every episode; who the *agent* plays as
        return tuple(self.board)  # is handled by the caller, not here.

    def legal_actions(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def step(self, action):
        """Applies `action` for whichever mark is currently due to move.
        Returns (board_after, outcome, done). `outcome` is from the
        perspective of the player who just moved: +1 win, 0 draw/ongoing,
        -1 is not reachable here (a player can't lose on their own move) --
        losses only happen from the *other* player's next move, which the
        caller derives by negating this outcome."""
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"illegal action {action}, legal={legal}, board={self.board}")
        mover = self.to_move
        self.board[action] = mover
        w = winner_of(self.board)
        self.to_move *= -1
        if w is None:
            return tuple(self.board), 0, False
        outcome = 1 if w == mover else 0  # w == 0 -> draw -> 0; w==mover -> win
        return tuple(self.board), outcome, True

    def board_from_perspective(self, player):
        """Board as seen by `player`: player's own marks read as +1,
        opponent's as -1, regardless of whether player is actually X or O.
        This is what lets a single agent's learned associations transfer
        between games where it moves first vs. second."""
        return tuple(player * v for v in self.board)

    def render(self):
        sym = {1: "X", -1: "O", 0: "."}
        rows = [" ".join(sym[self.board[r * 3 + c]] for c in range(3)) for r in range(3)]
        return "\n".join(rows)
