"""Wraps a tic-tac-toe agent so every board is canonicalized to one of its 8 equivalent orientations.
Values and visit counts live under the canonical form of the AFTERSTATE (what update_episode stores)."""
import random

from agents.shared.tictactoe_symmetry import canonicalize
from agents.shared.tie_break import argmax_random


class SymmetryWrapper:
    def __init__(self, inner):
        self.inner = inner

    def _keys(self, board, legal_actions):
        keys = []
        for a in legal_actions:
            b = list(board)
            b[a] = 1                                  # board is already from the mover's perspective
            keys.append(canonicalize(tuple(b))[0])
        return keys

    def value(self, board_after):
        return self.inner.value(canonicalize(board_after)[0])

    def choose_action(self, board, legal_actions, epsilon):
        keys = self._keys(board, legal_actions)
        under = [a for a, k in zip(legal_actions, keys) if self.inner._visits(k) < self.inner.min_pulls]
        if under:
            return random.choice(under)
        if random.random() < epsilon:
            return random.choice(legal_actions)
        return argmax_random(legal_actions, [self.inner.value(k) for k in keys])

    def best_action(self, board, legal_actions):
        keys = self._keys(board, legal_actions)
        return argmax_random(legal_actions, [self.inner.value(k) for k in keys])

    def update_episode(self, afterstates, outcome):
        self.inner.update_episode([canonicalize(s)[0] for s in afterstates], outcome)

    def calibrate_baseline(self, sample_boards):
        return self.inner.calibrate_baseline([canonicalize(b)[0] for b in sample_boards])

    def save(self, path):
        return self.inner.save(path)

    def load(self, path):
        return self.inner.load(path)

    def __getattr__(self, name):
        return getattr(self.inner, name)
