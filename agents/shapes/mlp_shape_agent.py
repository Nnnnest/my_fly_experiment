"""Dense(64)-Dense(64)-Dense(n_classes) MLP baseline for the shape
experiment -- same architecture convention as mlp_gridworld_agent.py /
mlp_tictactoe_agent.py, cross-entropy classification head instead of a
Q-value or single-scalar regression head, with a replay buffer for the
same reason mlp_tictactoe_agent.py uses one (a single gradient step
under-reinforces one example relative to a tabular/linear write).
"""
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class ShapeNet(nn.Module):
    def __init__(self, n_pixels, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_pixels, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class MLPShapeAgent:
    def __init__(self, n_pixels, n_classes=5, lr=1e-3, buffer_size=2000,
                 batch_size=32, min_buffer_before_train=64, seed=0):
        torch.manual_seed(seed)
        self.net = ShapeNet(n_pixels, n_classes)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.min_buffer_before_train = min_buffer_before_train
        self.n_pixels = n_pixels

    def predict(self, bitmap):
        x = torch.tensor(bitmap.reshape(-1), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.net(x)
        return int(torch.argmax(logits[0]).item()), logits[0].numpy()

    def train_on(self, bitmap, label):
        self.buffer.append((bitmap.reshape(-1).copy(), label))
        if len(self.buffer) < self.min_buffer_before_train:
            return
        batch = random.sample(self.buffer, min(self.batch_size, len(self.buffer)))
        xs = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        ys = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.long)
        logits = self.net(xs)
        loss = nn.functional.cross_entropy(logits, ys)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
