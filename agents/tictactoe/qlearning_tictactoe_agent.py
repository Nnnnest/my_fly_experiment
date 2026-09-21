"""Tabular V(afterstate) agent for tic-tac-toe.

Learns a value function over *resulting boards* (afterstates), not
state-action pairs -- see project notes on why this formulation was
chosen (the (state, action) one-hot encoding used for grid world doesn't
scale to tic-tac-toe's ~5,478 reachable states).

V(board) is always evaluated from the perspective of the player who just
moved into that board (i.e. board is already in "me=+1, opp=-1" form via
env.board_from_perspective). This lets one table serve as both players,
same trick the env file itself calls out.
"""
import random


class QLearningTicTacToeAgent:
    def __init__(self, alpha=0.1, min_pulls=5):
        self.V = {}          # board(tuple) -> float
        self.visits = {}     # board(tuple) -> int, forced exploration + decaying step size
        self.alpha = alpha
        self.min_pulls = min_pulls

    def value(self, board_after):
        return self.V.get(board_after, 0.0)

    def _visits(self, board_after):
        return self.visits.get(board_after, 0)

    def choose_action(self, board, legal_actions, epsilon):
        # forced exploration floor, same standard as grid world: try
        # under-visited afterstates before trusting epsilon-greedy
        afterstates = []
        for a in legal_actions:
            b = list(board)
            b[a] = 1  # board is already from the acting player's own perspective
            afterstates.append((a, tuple(b)))

        under_visited = [a for a, s in afterstates if self._visits(s) < self.min_pulls]
        if under_visited:
            return random.choice(under_visited)

        if random.random() < epsilon:
            return random.choice(legal_actions)

        best_a, best_v = None, float("-inf")
        for a, s in afterstates:
            v = self.value(s)
            if v > best_v:
                best_a, best_v = a, v
        return best_a

    def update_episode(self, afterstates, outcome):
        """afterstates: list of this agent's own afterstates from the
        episode, in order. outcome: +1 win, -1 loss, 0 draw, from this
        agent's perspective. Monte Carlo backup -- every afterstate in the
        episode gets the same terminal signal, matching the sign-gate
        training convention forced on the connectome agent (kept identical
        across all three agents for a fair comparison, not just applied to
        the connectome)."""
        target = float(outcome)
        for s in afterstates:
            self.visits[s] = self._visits(s) + 1
            n = self.visits[s]
            # sample-average while under the forced-exploration floor,
            # fixed step size after -- tune alpha if this doesn't converge
            # cleanly against the other two agents
            lr = 1.0 / n if n <= self.min_pulls else self.alpha
            old = self.value(s)
            self.V[s] = old + lr * (target - old)

    def best_action(self, board, legal_actions):
        """Pure greedy -- no forced exploration, no epsilon. For
        play/eval against a human or another trained agent, not training
        (choose_action is for training)."""
        best_a, best_v = None, float("-inf")
        for a in legal_actions:
            b = list(board)
            b[a] = 1
            v = self.value(tuple(b))
            if v > best_v:
                best_a, best_v = a, v
        return best_a

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"V": self.V, "visits": self.visits}, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.V = d["V"]
        self.visits = d["visits"]
