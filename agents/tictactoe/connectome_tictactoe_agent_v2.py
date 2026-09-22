import os
import random
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.join(_THIS_DIR, "..", ".."), os.path.join(_THIS_DIR, "..", "..", "..")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from fly_api import FlyBrainAPI
from agents.shared.paths import FLY_ROOT
from encoders.tictactoe.board_odor_encoder import board_to_vector, board_baseline_features
from agents.shared.mb_value_cache import train_no_readout
from agents.shared.board_value_cache import build_board_cache, enumerate_afterstates


class ConnectomeTicTacToeAgentV2:
    """Same public interface as ConnectomeTicTacToeAgent (value/
    choose_action/update_episode/best_action/save/load/drift_pct/sleep).
    Tier-1 equivalent: use_cache=True, gate="none" -- must match
    ConnectomeTicTacToeAgent's behavior exactly (same train() calls, same
    weights trajectory), just faster. VERIFY this before trusting any
    gate="error"/"visit" result against it.

    gate: same semantics as ConnectomeGridAgentV2 --
      "none"  train on every non-draw afterstate (v1 behavior)
      "visit" train with probability (min_pulls/n)^train_power after
              min_pulls visits of this exact afterstate
      "error" skip training an afterstate whose cached value is already
              past a calibrated margin in the right direction
    """

    def __init__(self, hops=1, min_pulls=5, use_cache=True, gate="none",
                 margin_frac=0.05, train_power=1.0, train_floor=0.05, seed=0):
        if gate not in ("none", "visit", "error"):
            raise ValueError("gate must be none/visit/error")
        self.fly = FlyBrainAPI(mode="mb", path=FLY_ROOT)
        self.hops = hops
        self.min_pulls = min_pulls
        self.gate = gate
        self.train_power, self.train_floor = train_power, train_floor
        self.visits = {}
        self._baseline_coef = None
        self.rng = np.random.default_rng(seed)
        self.n_trained = self.n_skipped = self.n_cache_miss = 0

        self.cache = None
        self.board_to_id = None
        if use_cache and hops == 1:
            boards = enumerate_afterstates()
            self.cache, self.board_to_id = build_board_cache(self.fly, boards, board_to_vector)
            res = self.cache.verify(n=24)
            if not res["ok"]:
                import warnings
                warnings.warn(f"tic-tac-toe value cache disagrees with fly.step ({res}); using slow path")
                self.cache = None

        self.tau = 0.0
        if gate == "error" and self.cache is not None:
            self.tau = margin_frac * self._calibrate_own_effect()

    def _calibrate_own_effect(self, n_probe=8, seed=12345):
        w0 = self.fly.wM.copy()
        rng = np.random.default_rng(seed)
        ids = rng.choice(len(self.board_to_id), size=min(n_probe, len(self.board_to_id)), replace=False)
        deltas = []
        for bid in ids:
            self.cache.refresh(); v0 = self.cache.values_pairs([bid])[0]
            train_no_readout(self.fly, self.cache.encoder.patterns[bid], hops=self.hops, reward=1.0)
            self.cache.refresh(); v1 = self.cache.values_pairs([bid])[0]
            self.fly.wM[:] = w0
            deltas.append(abs(v1 - v0))
        self.cache.refresh()
        return max(float(np.median(deltas)), 1e-9)

    def calibrate_baseline(self, sample_boards):
        """Same contract as v1: call once, before any train(). Uses the
        cache when available (near-instant vs. one fly.step() per board)."""
        X, y = [], []
        for board in sample_boards:
            X.append(board_baseline_features(board))
            if self.cache is not None and board in self.board_to_id:
                y.append(float(self.cache.values_pairs([self.board_to_id[board]])[0]))
            else:
                self.n_cache_miss += 1
                y.append(self.fly.step(odor=board_to_vector(board), hops=self.hops)["MB_pref"])
        X = np.array(X)
        X_aug = np.hstack([X, np.ones((len(X), 1))])
        coef, *_ = np.linalg.lstsq(X_aug, np.array(y), rcond=None)
        self._baseline_coef = coef

    def _baseline(self, board_after):
        feats = board_baseline_features(board_after)
        return float(np.append(feats, 1.0) @ self._baseline_coef)

    def value(self, board_after):
        if self.cache is not None and board_after in self.board_to_id:
            raw = float(self.cache.values_pairs([self.board_to_id[board_after]])[0])
        else:
            self.n_cache_miss += 1
            raw = self.fly.step(odor=board_to_vector(board_after), hops=self.hops)["MB_pref"]
        return raw - self._baseline(board_after)

    def _visits(self, board_after):
        return self.visits.get(board_after, 0)

    def choose_action(self, board, legal_actions, epsilon):
        afterstates = []
        for a in legal_actions:
            b = list(board); b[a] = 1
            afterstates.append((a, tuple(b)))
        under_visited = [a for a, s in afterstates if self._visits(s) < self.min_pulls]
        if under_visited:
            return random.choice(under_visited)
        if random.random() < epsilon:
            return random.choice(legal_actions)
        best_a, best_v = None, float("-inf")
        for a, s in afterstates:
            v = self.value(s)
            if v > best_v:
                best_a, best_v = a, v
        return best_a

    def _train_prob(self, n):
        if n <= self.min_pulls:
            return 1.0
        return max(self.train_floor, (self.min_pulls / n) ** self.train_power)

    def _skip(self, board_after, outcome):
        if self.gate == "none":
            return False
        if self.gate == "visit":
            return self.rng.random() >= self._train_prob(self._visits(board_after))
        if self.gate == "error" and self.cache is not None:
            v = self.value(board_after)
            return v >= self.tau if outcome > 0 else v <= -self.tau
        return False

    def update_episode(self, afterstates, outcome):
        for s in afterstates:
            self.visits[s] = self._visits(s) + 1
            if outcome == 0:
                continue  # draws: no clean sign, same as v1
            if self._skip(s, outcome):
                self.n_skipped += 1
                continue
            self.n_trained += 1
            vec = board_to_vector(s)
            if self.cache is not None:
                train_no_readout(self.fly, vec, hops=self.hops,
                                  **({"reward": 1.0} if outcome > 0 else {"punish": 1.0}))
                self.cache.refresh()
            elif outcome > 0:
                self.fly.train(odor=vec, reward=1.0, hops=self.hops)
            else:
                self.fly.train(odor=vec, punish=1.0, hops=self.hops)

    def best_action(self, board, legal_actions):
        best_a, best_v = None, float("-inf")
        for a in legal_actions:
            b = list(board); b[a] = 1
            v = self.value(tuple(b))
            if v > best_v:
                best_a, best_v = a, v
        return best_a

    def save(self, path_prefix):
        self.fly.save_weights(path_prefix + "_weights.npz")
        if self._baseline_coef is not None:
            np.save(path_prefix + "_baseline.npy", self._baseline_coef)

    def load(self, path_prefix):
        self.fly.load_weights(path_prefix + "_weights.npz")
        self._baseline_coef = np.load(path_prefix + "_baseline.npy")
        if self.cache is not None:
            self.cache.refresh()

    def drift_pct(self):
        num = float(np.abs(self.fly.wM - self.fly.wM0).mean())
        den = float(np.abs(self.fly.wM0).mean()) + 1e-9
        return 100.0 * num / den

    def sleep(self, **kwargs):
        r = self.fly.sleep(**kwargs)
        if self.cache is not None:
            self.cache.refresh()
        return r
