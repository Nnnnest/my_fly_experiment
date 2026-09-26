"""Classical baselines for circle-counting, regression versions of
linear_shape_agent.py / mlp_shape_agent.py -- MSE loss against the raw
count (not normalized; these have no scale constraint the way the
connectome's x_teach_output does), single scalar output, round+clip to
predict a count.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LinearCountAgent:
    def __init__(self, n_pixels, n_max=4, lr=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, size=n_pixels)
        self.b = 0.0
        self.lr = lr
        self.n_max = n_max

    def _raw(self, bitmap):
        return float(bitmap.reshape(-1) @ self.w + self.b)

    def predict(self, bitmap):
        est = self._raw(bitmap)
        return int(round(float(np.clip(est, 0, self.n_max)))), est

    def train_on(self, bitmap, count):
        x = bitmap.reshape(-1)
        pred = self._raw(bitmap)
        err = pred - count
        self.w -= self.lr * err * x
        self.b -= self.lr * err


class CountNet(nn.Module):
    def __init__(self, n_pixels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_pixels, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPCountAgent:
    def __init__(self, n_pixels, n_max=4, lr=1e-3, buffer_size=2000,
                 batch_size=32, min_buffer_before_train=64, seed=0):
        torch.manual_seed(seed)
        self.net = CountNet(n_pixels)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.min_buffer_before_train = min_buffer_before_train
        self.n_max = n_max

    def predict(self, bitmap):
        x = torch.tensor(bitmap.reshape(-1), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            est = self.net(x).item()
        return int(round(float(np.clip(est, 0, self.n_max)))), est

    def train_on(self, bitmap, count):
        self.buffer.append((bitmap.reshape(-1).copy(), float(count)))
        if len(self.buffer) < self.min_buffer_before_train:
            return
        batch = random.sample(self.buffer, min(self.batch_size, len(self.buffer)))
        xs = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        preds = self.net(xs)
        loss = nn.functional.mse_loss(preds, ys)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
