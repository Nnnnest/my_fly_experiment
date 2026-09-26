"""Classical baselines for number comparison, mirroring the bandit's
linear/MLP baseline pairing. Input is just (a, b) directly (2 features) --
no need to route through the 685-dim ALPN vector, since the comparison
itself doesn't depend on that representation the way the connectome's
does. Output: P(a > b), online logistic/gradient update per round, same
per-round training cadence as run_comparison.py's connectome loop.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LinearCompareAgent:
    """Logistic regression on [a, b, a*b, 1]."""

    def __init__(self, lr=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, size=4)
        self.lr = lr

    def _feats(self, a, b):
        return np.array([a, b, a * b, 1.0])

    def _p(self, a, b):
        z = float(self._feats(a, b) @ self.w)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, a, b):
        p = self._p(a, b)
        return ("A" if p >= 0.5 else "B"), p

    def train_on(self, a, b):
        y = 1.0 if a > b else 0.0
        p = self._p(a, b)
        grad = (p - y) * self._feats(a, b)
        self.w -= self.lr * grad


class CompareNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPCompareAgent:
    def __init__(self, lr=1e-3, seed=0):
        torch.manual_seed(seed)
        self.net = CompareNet()
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    def predict(self, a, b):
        x = torch.tensor([[float(a), float(b)]], dtype=torch.float32)
        with torch.no_grad():
            p = torch.sigmoid(self.net(x)).item()
        return ("A" if p >= 0.5 else "B"), p

    def train_on(self, a, b):
        x = torch.tensor([[float(a), float(b)]], dtype=torch.float32)
        y = torch.tensor([1.0 if a > b else 0.0], dtype=torch.float32)
        logit = self.net(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
