import numpy as np
import torch
import torch.nn as nn
import copy
from collections import deque

class QNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)

class MLPGridAgent:
    def __init__(self, n_states, n_actions, lr=0.01, gamma=0.95, epsilon=0.1,
                 min_pulls=5, target_sync_every=10, buffer_size=5000,
                 batch_size=32, min_buffer_before_train=200, seed=0):
        torch.manual_seed(seed)
        self.n_states = n_states
        self.n_actions = n_actions
        self.net = QNet(n_states, n_actions)
        self.target_net = copy.deepcopy(self.net)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.min_pulls = min_pulls
        self.counts = np.zeros((n_states, n_actions))
        self.rng = np.random.default_rng(seed)
        self.target_sync_every = target_sync_every
        self.update_count = 0
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.min_buffer_before_train = min_buffer_before_train

    def _one_hot_batch(self, states):
        v = torch.zeros(len(states), self.n_states)
        v[torch.arange(len(states)), states] = 1.0
        return v

    def select_action(self, state):
        under_explored = np.where(self.counts[state] < self.min_pulls)[0]
        if len(under_explored) > 0:
            return int(under_explored[0])
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            q = self.net(self._one_hot_batch([state]))
        return int(torch.argmax(q[0]).item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(self, state, action, reward, next_state, done):
        self.counts[state, action] += 1
        self.buffer.append((state, action, reward, next_state, done))

        if len(self.buffer) < self.min_buffer_before_train:
            return  # not enough data yet to sample a meaningful batch

        batch = [self.buffer[i] for i in self.rng.integers(len(self.buffer), size=self.batch_size)]
        states, actions, rewards, next_states, dones = zip(*batch)

        q = self.net(self._one_hot_batch(states))
        q_taken = q[torch.arange(len(batch)), torch.tensor(actions)]

        with torch.no_grad():
            q_next = self.target_net(self._one_hot_batch(next_states))
            max_next = torch.max(q_next, dim=1).values
            targets = torch.tensor(rewards, dtype=torch.float32) + \
                self.gamma * max_next * (1 - torch.tensor(dones, dtype=torch.float32))

        loss = torch.mean((q_taken - targets) ** 2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.update_count += 1
        if self.update_count % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.net.state_dict())
