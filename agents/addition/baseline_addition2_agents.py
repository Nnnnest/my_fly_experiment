"""Classical baselines for two-digit addition, given the SAME digit-level
inputs as the connectome (at, au, bt, bu, ct, cu) -- not the raw two-digit
numbers -- so all three agents solve the identical positional-weighting
problem, same fairness rule as Method B."""
import numpy as np
import torch
import torch.nn as nn


class LinearAddition2Agent:
    def __init__(self, lr=0.05, seed=0):
        self.w = np.zeros(14)  # 13 features (see encoder.features) + bias
        self.lr = lr

    @staticmethod
    def _feat(a, b, c):
        at, au = a // 10, a % 10
        bt, bu = b // 10, b % 10
        ct, cu = c // 10, c % 10
        units_sum = au + bu
        carry = 1.0 if units_sum >= 10 else 0.0
        diff = a + b - c
        return np.array([at, au, bt, bu, ct, cu, at * bt, au * bu,
                         units_sum, carry, diff, diff ** 2, a + b, 1.0])

    def predict(self, a, b, c):
        return float(self._feat(a, b, c) @ self.w) > 0.0

    def update(self, a, b, c, label):
        target = 1.0 if label else -1.0
        x = self._feat(a, b, c)
        err = target - x @ self.w
        self.w += self.lr * err * x / (np.dot(x, x) + 1e-6)


class MLPAddition2Agent:
    """Dense(64)-Dense(64)-1, input = the 6 digits (at,au,bt,bu,ct,cu)."""
    def __init__(self, lr=1e-3, seed=0):
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _x(self, a, b, c):
        at, au = a // 10, a % 10
        bt, bu = b // 10, b % 10
        ct, cu = c // 10, c % 10
        return torch.tensor([[float(at), float(au), float(bt), float(bu), float(ct), float(cu)]])

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
