import numpy as np

class QLearningGridAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1, min_pulls=5, seed=0):
        self.q = np.zeros((n_states, n_actions))
        self.counts = np.zeros((n_states, n_actions))
        self.min_pulls = min_pulls
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98

    def select_action(self, state):
        under_explored = np.where(self.counts[state] < self.min_pulls)[0]
        if len(under_explored) > 0:
            return int(under_explored[0])
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q[state]))

    def update(self, state, action, reward, next_state, done):
        self.counts[state, action] += 1
        target = reward if done else reward + self.gamma * np.max(self.q[next_state])
        self.q[state, action] += self.alpha * (target - self.q[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
