"""Connectome V(afterstate) agent for tic-tac-toe, via
FlyBrainAPI(mode="mb").

Reads MB_pref for the vector-encoded afterstate as the value signal.
Trained with train(reward=1.0) / train(punish=1.0) as fixed-magnitude
sign gates -- Monte Carlo backup only, no Bellman bootstrapping (see
grid-world's 113-state/hops=2 breakdown: bootstrapped targets fed into a
sign-gate trainer produced runaway, then decaying, values).

Baseline correction -- IMPORTANT, read before using:
The bandit fixed innate odor bias by subtracting each arm's untrained
MB_pref, computed once before any training happened anywhere. A *lazy*
"cache the first time this board is seen" baseline would be wrong here,
because whole-graph plasticity means training on board A shifts the
untrained-looking reading of board B too -- so a board's "baseline"
depends on when in the run you happened to first see it. calibrate_baseline()
must be called once, with a representative sample of boards, BEFORE any
train() call, and fits a linear model (board vector -> untrained MB_pref)
from that one clean pass. Every later value() lookup subtracts this
model's prediction, not a per-board cached number.
"""
import os
import random
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
from fly_api import FlyBrainAPI
from agents.shared.paths import FLY_ROOT
from encoders.tictactoe.board_odor_encoder import board_to_vector, board_baseline_features


class ConnectomeTicTacToeAgent:
    def __init__(self, hops=1, min_pulls=5):
        self.fly = FlyBrainAPI(mode="mb", path=FLY_ROOT)
        self.hops = hops
        self.min_pulls = min_pulls
        self.visits = {}
        self._baseline_coef = None  # set by calibrate_baseline()

    def calibrate_baseline(self, sample_boards):
        """Call once, before any train() call, with a representative
        sample of boards (e.g. sample_early_boards() from
        run_tictactoe.py -- a few hundred random legal boards a few plies
        deep is enough to fit the baseline model). Uses
        board_baseline_features (9 marks + 36 pairwise products), NOT the
        raw 685-dim board_to_vector -- see that function's docstring for
        why (diagnosed empirically: R^2=0.862 on the raw vector, expected
        to improve with interaction terms)."""
        X, y = [], []
        for board in sample_boards:
            feats = board_baseline_features(board)
            X.append(feats)
            y.append(self.fly.step(odor=board_to_vector(board), hops=self.hops)["MB_pref"])
        X = np.array(X)
        X_aug = np.hstack([X, np.ones((len(X), 1))])
        coef, *_ = np.linalg.lstsq(X_aug, np.array(y), rcond=None)
        self._baseline_coef = coef

    def _baseline(self, board_after):
        if self._baseline_coef is None:
            raise RuntimeError("call calibrate_baseline() before using this agent")
        feats = board_baseline_features(board_after)
        feats_aug = np.append(feats, 1.0)
        return float(feats_aug @ self._baseline_coef)

    def value(self, board_after):
        vec = board_to_vector(board_after)
        raw = self.fly.step(odor=vec, hops=self.hops)["MB_pref"]
        return raw - self._baseline(board_after)

    def _visits(self, board_after):
        return self.visits.get(board_after, 0)

    def choose_action(self, board, legal_actions, epsilon):
        afterstates = []
        for a in legal_actions:
            b = list(board)
            b[a] = 1
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

    def update_episode(self, afterstates, outcome):
        for s in afterstates:
            self.visits[s] = self._visits(s) + 1
            vec = board_to_vector(s)
            if outcome > 0:
                self.fly.train(odor=vec, reward=1.0, hops=self.hops)
            elif outcome < 0:
                self.fly.train(odor=vec, punish=1.0, hops=self.hops)
            # draws: skip training -- no clean reward/punish sign to assign,
            # and the bandit/grid-world lessons say the sign-gate trainer
            # shouldn't be fed anything but a real +/- signal

    def best_action(self, board, legal_actions):
        """Pure greedy -- no forced exploration, no epsilon. For
        play/eval against a human or another trained agent, not training."""
        best_a, best_v = None, float("-inf")
        for a in legal_actions:
            b = list(board)
            b[a] = 1
            v = self.value(tuple(b))
            if v > best_v:
                best_a, best_v = a, v
        return best_a

    def save(self, path_prefix):
        """Saves the fly weights (path_prefix + '_weights.npz') and the
        baseline regression coefficients (path_prefix + '_baseline.npy')
        -- both are needed to reproduce this agent's exact value() outputs
        on reload."""
        self.fly.save_weights(path_prefix + "_weights.npz")
        if self._baseline_coef is not None:
            np.save(path_prefix + "_baseline.npy", self._baseline_coef)

    def load(self, path_prefix):
        self.fly.load_weights(path_prefix + "_weights.npz")
        self._baseline_coef = np.load(path_prefix + "_baseline.npy")

    def drift_pct(self):
        """Mean |wM - wM0| / mean |wM0|, as a percent -- same convention
        the library itself reports (README's "drift 0.12%", x_report's
        EI_drift_%). Instrumentation to check whether a declining win rate
        late in a run tracks cumulative whole-graph weight drift from
        unconsolidated, non-decaying training updates (the same mechanism
        suspected in the 113-state/hops=2 grid-world maze breakdown)."""
        num = float(np.abs(self.fly.wM - self.fly.wM0).mean())
        den = float(np.abs(self.fly.wM0).mean()) + 1e-9
        return 100.0 * num / den

    def sleep(self, **kwargs):
        """Pass-through to FlyBrainAPI.sleep() (homeostatic downscaling,
        SHY). The library's own mechanism for the exact runaway-drift
        problem non-decaying training updates can cause -- call
        periodically (e.g. every N episodes, see run_tictactoe.py's
        sleep_every) to test whether it counteracts the decay pattern seen
        at tic-tac-toe's scale."""
        return self.fly.sleep(**kwargs)
