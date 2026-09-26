"""Classical baselines for the addition binary-judgment task, matching the
connectome agent's per-round online-update interface (predict/update) --
same fairness discipline as every other baseline in this project."""
import numpy as np
import torch
import torch.nn as nn


class LinearAdditionAgent:
    """Online linear model on the same 9 features as the connectome's
    baseline regression, trained by gradient steps toward +1/-1 targets --
    matches the connectome's binary reward/punish framing (not full
    logistic regression), same 'plain linear' comparison point used
    elsewhere in the project."""
    def __init__(self, lr=0.05, seed=0):
        self.w = np.zeros(10)  # 9 features + bias
        self.lr = lr

    @staticmethod
    def _feat(a, b, c):
        diff = a + b - c
        return np.array([a, b, c, a * b, a * c, b * c, diff, diff ** 2, a + b, 1.0])

    def predict(self, a, b, c):
        return float(self._feat(a, b, c) @ self.w) > 0.0

    def update(self, a, b, c, label):
        target = 1.0 if label else -1.0
        x = self._feat(a, b, c)
        err = target - x @ self.w
        self.w += self.lr * err * x / (np.dot(x, x) + 1e-6)


class MLPAdditionAgent:
    """Dense(64)-Dense(64)-1 binary classifier, sigmoid output, raw (a,b,c)
    input -- same shape family as mlp_tictactoe_agent.py's baseline.
    Trained one example at a time (BCE loss); no replay buffer, since
    addition triples aren't sequential/correlated the way grid-world
    transitions are."""
    def __init__(self, lr=1e-3, seed=0):
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _x(self, a, b, c):
        return torch.tensor([[float(a), float(b), float(c)]])

    def predict(self, a, b, c):
        with torch.no_grad():
            logit = self.net(self._x(a, b, c))
        return logit.item() > 0.0

    def update(self, a, b, c, label):
        target = torch.tensor([[1.0 if label else 0.0]])
        logit = self.net(self._x(a, b, c))
        loss = nn.functional.binary_cross_entropy_with_logits(logit, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
