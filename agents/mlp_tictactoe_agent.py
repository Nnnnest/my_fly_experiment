"""Feedforward V(afterstate) agent for tic-tac-toe -- Dense(64)-Dense(64)-1,
a single-scalar-output shape, not the spec's original
State -> Dense(64) -> Dense(64) -> Actions head. That head was designed for
per-action Q(s,a) outputs, which doesn't fit the afterstate formulation the
connectome needs (see run_tictactoe.py module docstring / project notes for
why afterstates were adopted). Using the same single-output shape here
means all three agents are literally fitting the same regression target,
which matters for a fair comparison.

Carries over ONE of the two grid-world MLP fixes: an experience replay
buffer, since a single gradient step under-reinforces a rare early win
relative to a tabular write. The OTHER grid-world fix -- a frozen target
network -- doesn't apply here: that fix addressed a moving-target problem
from Bellman bootstrapping (train on reward + gamma * next-state estimate,
itself changing every update). This agent uses Monte Carlo backup instead
(fixed terminal-outcome targets, see update_episode), which has no moving
target, so a target network would be inert complexity, not a fix.

Requires torch. If that's not available in your environment, the buffer +
choose_action logic is unchanged if you swap ValueNet for a hand-rolled
numpy MLP with manual backprop.
"""
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class ValueNet(nn.Module):
    def __init__(self, in_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPTicTacToeAgent:
    def __init__(self, lr=1e-3, buffer_size=5000, batch_size=32, min_pulls=5):
        self.net = ValueNet()
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.visits = {}
        self.min_pulls = min_pulls

    @staticmethod
    def _to_tensor(board):
        return torch.tensor(board, dtype=torch.float32)

    def value(self, board_after):
        with torch.no_grad():
            return self.net(self._to_tensor(board_after).unsqueeze(0)).item()

    def _visits(self, board_after):
        return self.visits.get(board_after, 0)

    def choose_action(self, board, legal_actions, epsilon):
        afterstates = []
        for a in legal_actions:
            b = list(board)
            b[a] = 1
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
        target = float(outcome)
        for s in afterstates:
            self.visits[s] = self._visits(s) + 1
            self.buffer.append((s, target))

        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)

        preds = self.net(states)
        loss = nn.functional.mse_loss(preds, targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def best_action(self, board, legal_actions):
        """Pure greedy -- no forced exploration, no epsilon. For
        play/eval against a human or another trained agent, not training."""
        best_a, best_v = None, float("-inf")
        for a in legal_actions:
            b = list(board)
            b[a] = 1
            v = self.value(tuple(b))
            if v > best_v:
                best_a, best_v = a, v
        return best_a

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
