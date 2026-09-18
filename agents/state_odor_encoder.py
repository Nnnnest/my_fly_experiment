import numpy as np

class StateOdorEncoder:
    def __init__(self, n_states, live_indices_path, n_target=685, mag=5.0, seed=0):
        live = np.load(live_indices_path)
        rng = np.random.default_rng(seed)
        assert n_states <= len(live), "more states than live ALPN indices — widen hops or reduce grid size"
        self.chosen = rng.choice(live, size=n_states, replace=False)
        self.n_target = n_target
        self.mag = mag
        self.patterns = np.zeros((n_states, n_target), dtype=np.float32)
        for s, idx in enumerate(self.chosen):
            self.patterns[s, idx] = mag

    def encode(self, state):
        return self.patterns[state]
