import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from fly_api import FlyBrainAPI
from state_action_odor_encoder import StateActionOdorEncoder
from sparse_encoder import SparseStateActionEncoder
from mb_value_cache import MBValueCache, train_no_readout, calibrate_own_effect


class ConnectomeGridAgentV2:
    """Same task interface and shaping as ConnectomeGridAgent. Changes:

    encoding="khot": sparse random ALPN subsets per (state, action) instead of one
        ALPN per pair, so hops=1 works for larger mazes.
    use_cache=True (hops=1 only): values come from MBValueCache (exact, self-verified)
        instead of 4 brain.step() calls per decision; train()'s unused readout is skipped.
    gate: WHEN to call brain.train() (train() itself has a fixed step size, so the only
        lever the agent has is how often it applies it):
        "none"  always train (the old behaviour)
        "visit" train with probability (min_pulls/n)^power after n visits of a pair
        "error" train a pair only while its own value is on the wrong side of a margin
        "rank"  train only while the RANKING at that state is wrong: a rewarded action
                must beat every other action by a margin, a punished action must lose to
                some other action by a margin. Learning stops once the greedy policy is
                right, so there is no ongoing drift. Needs the cache (hops=1).
    margin_frac: margin = margin_frac * (value change caused by ONE train), measured on
        the circuit at start-up (weights restored afterwards).
    max_pair_trains: optional cap on how many times any single pair may be trained
        (stops persistent conflicts from being retrained on every visit).
    train_decay=True is kept as an alias for gate="visit".
    """

    def __init__(self, traversable_state_ids, n_actions, live_indices_path, bfs_dist,
                 hops=1, epsilon=0.1, min_pulls=5, seed=0,
                 encoding="khot", k=6, max_overlap=1,
                 use_cache=True, gate=None, train_decay=None, margin_frac=0.05,
                 train_power=1.0, train_floor=0.05, verify_every=500, max_pair_trains=None):
        if gate is None:
            gate = "visit" if train_decay else "none"
        if gate not in ("none", "visit", "error", "rank"):
            raise ValueError("gate must be none/visit/error/rank")
        self.gate = gate
        self.state_map = {sid: i for i, sid in enumerate(traversable_state_ids)}
        n_states = len(traversable_state_ids)
        self.n_states, self.n_actions = n_states, n_actions
        self.brain = FlyBrainAPI(mode="mb")
        if encoding == "khot":
            self.encoder = SparseStateActionEncoder(n_states, n_actions, live_indices_path,
                                                    k=k, max_overlap=max_overlap, seed=seed)
        elif encoding == "onehot":
            self.encoder = StateActionOdorEncoder(n_states, n_actions, live_indices_path, seed=seed)
        else:
            raise ValueError("encoding must be 'khot' or 'onehot'")
        self.bfs_dist, self.hops = bfs_dist, hops
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, 0.01, 0.98
        self.min_pulls = min_pulls
        self.counts = np.zeros((n_states, n_actions))
        self.rng = np.random.default_rng(seed)
        self.train_power, self.train_floor = train_power, train_floor
        self.verify_every = verify_every
        self.n_trained = self.n_skipped = self.n_updates = 0
        self.max_pair_trains = max_pair_trains  # optional per-pair training budget (error/rank gates)
        self.train_counts = np.zeros((n_states, n_actions), dtype=np.int64)

        self.cache = None
        if use_cache and hops == 1:
            self.cache = MBValueCache(self.brain, self.encoder)
            res = self.cache.verify(n=24)
            if not res["ok"]:
                warnings.warn(f"value cache disagrees with brain.step ({res}); using slow path")
                self.cache = None
            elif not res["kc_sorted"]:
                warnings.warn("brain.KC is not sorted: the library pairs sorted-rank weights with "
                              "unsorted-order activations. The cache mirrors this; check the wiring.")
        if gate in ("error", "rank") and self.cache is None:
            raise ValueError(f"gate='{gate}' needs the value cache (hops=1, use_cache=True)")
        self._dirty = False

        if self.cache is not None:
            self.baseline = self.cache.values_all().reshape(n_states, n_actions)
        else:
            self.baseline = np.zeros((n_states, n_actions))
            for s in range(n_states):
                for a in range(n_actions):
                    self.baseline[s, a] = self.brain.step(
                        odor=self.encoder.encode(s, a), hops=self.hops)["MB_pref"]

        self.tau_r = self.tau_p = 0.0
        if gate in ("error", "rank"):
            d_r, d_p = calibrate_own_effect(self.brain, self.cache, self.encoder)
            self.tau_r, self.tau_p = margin_frac * d_r, margin_frac * d_p
            print(f"[agent] gate={gate}: one-train effect reward={d_r:.2f} punish={d_p:.2f}; "
                  f"margins {self.tau_r:.2f}/{self.tau_p:.2f} (margin_frac={margin_frac})", flush=True)

    def _c(self, state):
        return self.state_map[state]

    def _corrected_values(self, state):
        s = self._c(state)
        if self.cache is not None:
            if self._dirty:
                self.cache.refresh()
                self._dirty = False
            return self.cache.values_state(s) - self.baseline[s]
        vals = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            raw = self.brain.step(odor=self.encoder.encode(s, a), hops=self.hops)["MB_pref"]
            vals[a] = raw - self.baseline[s, a]
        return vals

    def greedy_action(self, state):
        return int(np.argmax(self._corrected_values(state)))

    def select_action(self, state):
        s = self._c(state)
        under = np.where(self.counts[s] < self.min_pulls)[0]
        if len(under) > 0:
            return int(under[0])
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _train_prob(self, n):
        if n <= self.min_pulls:
            return 1.0
        return max(self.train_floor, (self.min_pulls / n) ** self.train_power)

    def _satisfied(self, state, action, target_reward):
        """Gate check for 'error' / 'rank': is this pair already where it should be?"""
        vals = self._corrected_values(state)
        if self.gate == "error":
            v = vals[action]
        else:  # rank: margin over the best OTHER action
            v = vals[action] - np.max(np.delete(vals, action))
        return v >= self.tau_r if target_reward else v <= -self.tau_p

    def _train(self, odor, **kw):
        b = self.brain
        if self.cache is not None:
            train_no_readout(b, odor, hops=self.hops, **kw)
        else:
            b.train(odor=odor, hops=self.hops, **kw)
        self._dirty = True
        self.n_trained += 1

    def update(self, state, action, reward, next_state, done):
        s = self._c(state)
        self.counts[s, action] += 1
        self.n_updates += 1
        odor = self.encoder.encode(s, action)
        goal = bool(done and reward > 0)
        target_reward = goal or bool(self.bfs_dist[next_state] < self.bfs_dist[state])

        skip = False
        if self.gate == "visit":
            skip = (not goal) and self.rng.random() >= self._train_prob(self.counts[s, action])
        elif self.gate in ("error", "rank"):
            skip = self._satisfied(state, action, target_reward)
        if (not skip and self.max_pair_trains is not None
                and self.train_counts[s, action] >= self.max_pair_trains):
            skip = True
        if skip:
            self.n_skipped += 1
        else:
            self.train_counts[s, action] += 1
            if target_reward:
                self._train(odor, reward=1.0)
            else:
                self._train(odor, punish=1.0)

        if self.cache is not None and self.verify_every and self.n_updates % self.verify_every == 0:
            res = self.cache.verify(n=3, seed=self.n_updates)
            if not res["ok"]:
                if self.gate in ("error", "rank"):
                    raise RuntimeError(f"cache drifted from brain.step ({res}); gate needs the cache")
                warnings.warn(f"cache drifted from brain.step ({res}); switching to slow path")
                self.cache = None
