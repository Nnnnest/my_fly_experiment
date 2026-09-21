"""Wraps any tic-tac-toe agent (qlearning/mlp/connectome) so every board it
sees is canonicalized to one of its 8 equivalent orientations first (see
tictactoe_symmetry.py). Same value/choose_action/best_action/
update_episode/save/load interface as the wrapped agent, so it's a
drop-in replacement anywhere in this project:

    agent = SymmetryWrapper(ConnectomeTicTacToeAgent())

instead of

    agent = ConnectomeTicTacToeAgent()

-- nothing else in run_tictactoe.py / play_tictactoe.py /
tictactoe_training_curve.py needs to change.
"""
from tictactoe_symmetry import canonicalize, action_to_canonical, action_from_canonical


class SymmetryWrapper:
    def __init__(self, inner):
        self.inner = inner

    def value(self, board_after):
        canon, _ = canonicalize(board_after)
        return self.inner.value(canon)

    def choose_action(self, board, legal_actions, epsilon):
        canon, name = canonicalize(board)
        canon_legal = [action_to_canonical(a, name) for a in legal_actions]
        canon_a = self.inner.choose_action(canon, canon_legal, epsilon)
        return action_from_canonical(canon_a, name)

    def best_action(self, board, legal_actions):
        canon, name = canonicalize(board)
        canon_legal = [action_to_canonical(a, name) for a in legal_actions]
        canon_a = self.inner.best_action(canon, canon_legal)
        return action_from_canonical(canon_a, name)

    def update_episode(self, afterstates, outcome):
        canon_afterstates = [canonicalize(s)[0] for s in afterstates]
        self.inner.update_episode(canon_afterstates, outcome)

    def calibrate_baseline(self, sample_boards):
        """Connectome-specific -- canonicalize the calibration sample too,
        so the fitted baseline model matches the orientation distribution
        value()/choose_action() actually feed it at runtime (all
        canonical). Without this override, __getattr__ below would forward
        the RAW sample_boards straight to the inner agent, calibrating
        against a different distribution than it's evaluated on."""
        canon_boards = [canonicalize(b)[0] for b in sample_boards]
        return self.inner.calibrate_baseline(canon_boards)

    def save(self, path):
        return self.inner.save(path)

    def load(self, path):
        return self.inner.load(path)

    def __getattr__(self, name):
        # forward anything else (sleep, drift_pct, fly, ...) straight to
        # the wrapped agent
        return getattr(self.inner, name)
