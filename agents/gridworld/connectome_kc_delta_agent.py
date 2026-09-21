import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared")
import numpy as np
from fly_api import FlyBrainAPI
from sparse_encoder import SparseStateActionEncoder
from mb_value_cache import MBValueCache


class ConnectomeKCDeltaAgent:
    """Connectome as a FIXED state representation + a delta-rule Q readout.

    What is and is not the connectome here:
      * each STATE gets a sparse ALPN stimulus; the real ALPN->KC wiring turns it into a
        sparse Kenyon-cell code (top 5%, ~250 of ~5000 KCs). That code is computed exactly
        (MBValueCache) and never changes.
      * Q(s, a) = w_a . code(s): one weight vector per action, trained with a normalized
        delta rule on the temporal-difference error (Q-learning target, same shaped reward
        and gamma as the classical baselines).
      * The readout weights are NOT connectome synapses and train() / DAN gating is NOT used.
        This tests "is the connectome's KC code a good representation for RL", not "does its
        own reward-gated plasticity learn". Report it as a separate, clearly labelled variant.

    kc_frac sets code sparsity (default 0.05 = the library's top-5% rule; smaller values are a protocol
    choice and apply equally to the random control).

    features="random" is the CONTROL: same number of KCs, same sparsity, but random binary
    codes with no connectome structure. If it performs as well, the real wiring adds nothing
    as a feature map for this task.
    """

    def __init__(self, traversable_state_ids, n_actions, live_indices_path, hops=1,
                 epsilon=0.1, min_pulls=5, seed=0, k=6, max_overlap=1,
                 alpha=0.5, gamma=0.95, features="kc", kc_frac=0.05):
        if features not in ("kc", "random"):
            raise ValueError("features must be 'kc' or 'random'")
        self.state_map = {sid: i for i, sid in enumerate(traversable_state_ids)}
        n_states = len(traversable_state_ids)
        self.n_states, self.n_actions = n_states, n_actions
        self.alpha, self.gamma = alpha, gamma
        self.rng = np.random.default_rng(seed)

        brain = FlyBrainAPI(mode="mb")
        enc = SparseStateActionEncoder(n_states, 1, live_indices_path, k=k,
                                       max_overlap=max_overlap, seed=seed)
        cache = MBValueCache(brain, enc)   # library-default 5% sparsity: verify against brain.step
        res = cache.verify(n=min(16, n_states))
        if not res["ok"]:
            raise RuntimeError(f"KC code cache disagrees with brain.step: {res}")
        if kc_frac != 0.05:
            # protocol choice: keep only the top kc_frac of KCs per state (sparser = less cross-talk
            # between states). Applied identically to the random control.
            cache = MBValueCache(brain, enc, kc_frac=kc_frac)
        self.kc_frac = kc_frac
        codes = cache.codes.astype(np.float64)          # (n_states, n_kc)
        if features == "random":
            n_kc = codes.shape[1]
            n_act = int(round((codes > 0).sum(axis=1).mean()))
            rc = np.zeros_like(codes)
            for i in range(n_states):
                rc[i, self.rng.choice(n_kc, size=n_act, replace=False)] = 1.0
            codes = rc
        self.features = features
        self.X = codes / np.maximum(np.linalg.norm(codes, axis=1, keepdims=True), 1e-12)
        self.W = np.zeros((n_actions, self.X.shape[1]))

        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, 0.01, 0.98
        self.min_pulls = min_pulls
        self.counts = np.zeros((n_states, n_actions))
        self.n_trained = 0

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
        delta = target - self._q(s)[action]
        self.W[action] += self.alpha * delta * self.X[s]
        self.n_trained += 1
