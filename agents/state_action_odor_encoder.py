import numpy as np

class StateActionOdorEncoder:
    """One distinct ALPN one-hot pattern per (state, action) pair."""
    def __init__(self, n_states, n_actions, live_indices_path, n_target=685, mag=5.0, seed=0):
        live = np.load(live_indices_path)
        n_pairs = n_states * n_actions
        assert n_pairs <= len(live), (
            f"{n_pairs} (state,action) pairs exceeds {len(live)} live ALPN indices — "
            "reduce n_states, or widen hops and rescan live_alpn_indices.npy"
        )
        rng = np.random.default_rng(seed)
        chosen = rng.choice(live, size=n_pairs, replace=False)

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_target = n_target
        self.mag = mag
        self.patterns = np.zeros((n_pairs, n_target), dtype=np.float32)
        for pair_id, idx in enumerate(chosen):
            self.patterns[pair_id, idx] = mag

    def _pair_id(self, state, action):
        return state * self.n_actions + action

    def encode(self, state, action):
        return self.patterns[self._pair_id(state, action)]
