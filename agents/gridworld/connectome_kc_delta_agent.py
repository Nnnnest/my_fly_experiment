import numpy as np

from core import paths
from core.brain_pool import fresh_brain
from agents.shared.sparse_encoder import SparseStateActionEncoder
from agents.shared.mb_value_cache import MBValueCache


class ConnectomeKCDeltaAgent:
    """Fixed state representation (real KC code, or a random code of the same size/sparsity as the
    control) + delta-rule Q readout. NOT connectome learning: train()/DAN gating are never used."""

    def __init__(self, traversable_state_ids, n_actions, live_indices_path, epsilon=0.1, min_pulls=5,
                 seed=0, k=6, max_overlap=1, alpha=0.5, gamma=0.95, features="kc"):
        if features not in ("kc", "random"):
            raise ValueError("features must be 'kc' or 'random'")
        self.state_map = {sid: i for i, sid in enumerate(traversable_state_ids)}
        n_states = len(traversable_state_ids)
        self.n_states, self.n_actions, self.alpha, self.gamma = n_states, n_actions, alpha, gamma
        self.rng = np.random.default_rng(seed)
        brain = fresh_brain()
        enc = SparseStateActionEncoder(n_states, 1, live_indices_path, k=k, max_overlap=max_overlap, seed=seed)
        cache = MBValueCache(brain, enc)
        res = cache.verify(n=min(16, n_states))
        if not res["ok"]:
            raise RuntimeError(f"KC code cache disagrees with brain.step: {res}")
        codes = cache.codes.astype(np.float64)
        if features == "random":
            n_kc = codes.shape[1]
            n_act = int(round((codes > 0).sum(axis=1).mean()))
            rc = np.zeros_like(codes)
            for i in range(n_states):
                rc[i, self.rng.choice(n_kc, size=n_act, replace=False)] = 1.0
            codes = rc
        self.X = codes / np.maximum(np.linalg.norm(codes, axis=1, keepdims=True), 1e-12)
        self.W = np.zeros((n_actions, self.X.shape[1]))
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, 0.01, 0.98
        self.min_pulls = min_pulls
        self.counts = np.zeros((n_states, n_actions))
        self.n_trained, self.n_skipped = 0, 0

    def _c(self, state):
        return self.state_map[state]

    def _q(self, s):
        return self.W @ self.X[s]

    def greedy_action(self, state):
        return int(np.argmax(self._q(self._c(state))))

    def select_action(self, state):
        s = self._c(state)
        under = np.where(self.counts[s] < self.min_pulls)[0]
        if len(under) > 0:
            return int(under[0])
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self._q(s)))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(self, state, action, reward, next_state, done):
        s = self._c(state)
        self.counts[s, action] += 1
        target = reward if done else reward + self.gamma * np.max(self._q(self._c(next_state)))
        self.W[action] += self.alpha * (target - self._q(s)[action]) * self.X[s]
        self.n_trained += 1
