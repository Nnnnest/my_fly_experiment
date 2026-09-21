import numpy as np


class SparseStateActionEncoder:
    """k-hot ALPN code per (state, action) pair.

    Drop-in replacement for StateActionOdorEncoder (same attributes and
    encode(state, action)). Instead of one dedicated ALPN per pair (which caps
    the maze at len(live) // n_actions states), each pair gets a random subset
    of k live ALPN positions, with pairwise overlap capped at `max_overlap`.
    The number of usable pairs is then limited by combinatorics, not by the
    number of live ALPNs, so hops=1 can be kept for larger mazes.

    live_indices: path to a .npy file OR an array of live ALPN positions.
    """

    def __init__(self, n_states, n_actions, live_indices, k=6, max_overlap=1,
                 n_target=685, mag=5.0, seed=0, max_tries=5000):
        live = np.load(live_indices) if isinstance(live_indices, str) else np.asarray(live_indices)
        live = live.astype(np.int64)
        L = len(live)
        n_pairs = n_states * n_actions
        if k > L:
            raise ValueError(f"k={k} exceeds {L} live ALPN positions")
        rng = np.random.default_rng(seed)
        P = np.zeros((n_pairs, L), dtype=bool)
        for i in range(n_pairs):
            for _ in range(max_tries):
                cand = rng.choice(L, size=k, replace=False)
                if i == 0 or P[:i][:, cand].sum(axis=1).max() <= max_overlap:
                    P[i, cand] = True
                    break
            else:
                raise RuntimeError(
                    f"could not place pair {i}/{n_pairs} with k={k}, max_overlap={max_overlap} "
                    f"in {L} live positions; raise max_overlap, lower k, or widen the live set")
        self.n_states, self.n_actions = n_states, n_actions
        self.n_target, self.mag = n_target, mag
        self.k, self.max_overlap = k, max_overlap
        self.patterns = np.zeros((n_pairs, n_target), dtype=np.float32)
        for i in range(n_pairs):
            self.patterns[i, live[P[i]]] = mag

    def _pair_id(self, state, action):
        return state * self.n_actions + action

    def encode(self, state, action):
        return self.patterns[self._pair_id(state, action)]
