import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_THIS_DIR, "..", "..", ".."), os.path.join(_THIS_DIR, "..", "..")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from fly_api import FlyBrainAPI
from agents.shared.paths import FLY_ROOT
from agents.shared.state_action_odor_encoder import StateActionOdorEncoder
import numpy as np

class ConnectomeGridAgent:
    def __init__(self, traversable_state_ids, n_actions, live_indices_path, bfs_dist,
                 hops=1, epsilon=0.1, min_pulls=5, seed=0):
        """traversable_state_ids: list of env state_id() values the agent can
        actually occupy — NOT env.n_states (which counts wall cells too).
        This keeps the ALPN index budget spent only on reachable cells."""
        self.state_map = {sid: i for i, sid in enumerate(traversable_state_ids)}
        n_compact_states = len(traversable_state_ids)

        self.brain = FlyBrainAPI(mode="mb", path=FLY_ROOT)
        self.encoder = StateActionOdorEncoder(n_compact_states, n_actions, live_indices_path, seed=seed)
        self.n_actions = n_actions
        self.bfs_dist = bfs_dist
        self.hops = hops
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.min_pulls = min_pulls
        self.counts = np.zeros((n_compact_states, n_actions))
        self.rng = np.random.default_rng(seed)

        self.baseline = np.zeros((n_compact_states, n_actions))
        for s in range(n_compact_states):
            for a in range(n_actions):
                self.baseline[s, a] = self.brain.step(
                    odor=self.encoder.encode(s, a), hops=self.hops
                )["MB_pref"]

    def _c(self, state):
        return self.state_map[state]

    def _corrected_values(self, state):
        s = self._c(state)
        vals = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            raw = self.brain.step(odor=self.encoder.encode(s, a), hops=self.hops)["MB_pref"]
            vals[a] = raw - self.baseline[s, a]
        return vals

    def select_action(self, state):
        s = self._c(state)
        under_explored = np.where(self.counts[s] < self.min_pulls)[0]
        if len(under_explored) > 0:
            return int(under_explored[0])
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self._corrected_values(state)))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(self, state, action, reward, next_state, done):
        s = self._c(state)
        self.counts[s, action] += 1
        odor = self.encoder.encode(s, action)

        if done and reward > 0:
            self.brain.train(odor=odor, reward=1.0, hops=self.hops)
            return

        if self.bfs_dist[next_state] < self.bfs_dist[state]:
            self.brain.train(odor=odor, reward=1.0, hops=self.hops)
        else:
            self.brain.train(odor=odor, punish=1.0, hops=self.hops)
