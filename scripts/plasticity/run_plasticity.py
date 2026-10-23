"""Plasticity sweep (revision plan, priority 3). Inputs, feedback, exploration and evaluation are unchanged; only the
native update is modified, from OUTSIDE the simulator (fly_api.py is not touched):
  g (strength)  : after each train() call the KC->MBON weights are set to w_old + g * (w_new - w_old), clipped at 0.05.
                  g = 1 is the unmodified rule. This scales the whole change, including train()'s own renormalisation.
  f (frequency) : each train() call is executed with probability f (otherwise skipped).
g = 1, f = 1 must reproduce the main-experiment runs exactly (see the checks in the README of this script's instructions).
    python -m scripts.plasticity.run_plasticity --smoke
    python -m scripts.plasticity.run_plasticity --workers 14"""
import numpy as np

from core import paths
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from encoders.addition.addition_encoder import AdditionEncoder
from encoders.addition.addition_encoder_2digit import TwoDigitAdditionEncoder
from agents.shared.addition_common import sample_batch, sample_triple
from agents.shared.mb_value_cache import train_no_readout
from agents.gridworld.connectome_gridworld_agent_v2 import ConnectomeGridAgentV2
from agents.comparison.connectome_compare_agent import ConnectomeCompareAgent
from agents.addition.connectome_addition_agent import ConnectomeAdditionAgent
import scripts.gridworld.run_gridworld as rg
import scripts.comparison.run_comparison as rc
import scripts.addition.run_addition as ra

EXPERIMENT = "plasticity"
GRID_G = [0.25, 0.5, 1.0, 2.0]
GRID_F = [0.25, 0.5, 1.0]
MAZE_COMBOS = [(0.1, 1.0), (0.3, 1.0), (1.0, 1.0), (1.0, 0.3), (1.0, 0.1)]   # ungated k-hot maze: (g, f)
MAZE_SEEDS = [0, 1, 2, 3, 4]
SMALL_TASKS = {
    "cmp_knob_n49": {"family": "cmp", "kind": "knob", "n_max": 49, "rounds": 400},
    "cmp_digits_5": {"family": "cmp", "kind": "digits", "n_digits": 5, "rounds": 400},
    "add_single": {"family": "add", "width": "single", "rounds": 800},
    "add_two": {"family": "add", "width": "two", "rounds": 800},
}


def scaled_train(brain, odor, g, **kw):
    km = getattr(brain, "_km_idx", None)
    if km is None:
        km = brain._km_idx = np.where(brain.km_mask)[0]
    w0 = brain.wM[km].copy()
    train_no_readout(brain, odor, hops=1, **kw)
    if g != 1.0:
        brain.wM[km] = np.maximum(w0 + g * (brain.wM[km] - w0), 0.05)


class _Plastic:
    def _init_plastic(self, g, f, seed):
        self.g, self.f = float(g), float(f)
        self._prng = np.random.default_rng([seed, 77])
        self.n_trained = self.n_skipped = 0

    def _skip(self):
        if self.f < 1.0 and self._prng.random() >= self.f:
            self.n_skipped += 1
            return True
        self.n_trained += 1
        return False


class ScaledCompare(_Plastic, ConnectomeCompareAgent):
    def __init__(self, encoder, g, f, seed):
        ConnectomeCompareAgent.__init__(self, encoder)
        self._init_plastic(g, f, seed)

    def train_on(self, a, b):
        if self._skip():
            return
        kw = {"reward": 1.0} if a > b else {"punish": 1.0}
        scaled_train(self.brain, self.encoder.encode(a, b), self.g, **kw)
        self.cache.refresh()


class ScaledAddition(_Plastic, ConnectomeAdditionAgent):
    def __init__(self, encoder, width, g, f, seed):
        ConnectomeAdditionAgent.__init__(self, encoder, width)
        self._init_plastic(g, f, seed)

    def update(self, a, b, c, label):
        if self._skip():
            return
        kw = {"reward": 1.0} if label else {"punish": 1.0}
        scaled_train(self.brain, self.encoder.encode(a, b, c), self.g, **kw)
        self.cache.refresh()


class ScaledGrid(ConnectomeGridAgentV2):
    def __init__(self, *args, g=1.0, f=1.0, plas_seed=0, **kw):
        super().__init__(*args, **kw)
        self.g, self.f = float(g), float(f)
        self._prng = np.random.default_rng([plas_seed, 77])

    def _train(self, odor, **kw):
        if self.f < 1.0 and self._prng.random() >= self.f:
            self.n_skipped += 1
            return
        scaled_train(self.brain, odor, self.g, **kw)
        self._dirty = True
        self.n_trained += 1


_orig_make_agent = rg.make_agent


def _make_agent(spec, env, trav_ids, bfs):
    cfg = spec.config
    if "plastic" in cfg:
        return ScaledGrid(trav_ids, 4, paths.LIVE_ALPN, bfs, hops=1, min_pulls=rg.MIN_PULLS, seed=spec.seed,
                          g=cfg["g"], f=cfg["f"], plas_seed=spec.seed, **cfg["conn"])
    return _orig_make_agent(spec, env, trav_ids, bfs)


rg.make_agent = _make_agent    # run_gridworld.run_one resolves make_agent by name at call time


def run_cmp(spec):
    cfg, seed = spec.config, spec.seed
    enc, sample, _ = rc.build(cfg, seed)
    r_eval, r_cal, r_tr = (np.random.default_rng([seed, k]) for k in (1, 2, 3))
    eval_pairs = [sample(r_eval) for _ in range(rc.N_EVAL)]
    agent = ScaledCompare(enc, cfg["g"], cfg["f"], seed)
    agent.calibrate_baseline([sample(r_cal) for _ in range(rc.N_CALIB)])
    pre = rc.acc(agent, eval_pairs)
    for _ in range(cfg["rounds"]):
        a, b = sample(r_tr)
        agent.train_on(a, b)
    return {"eval_acc_pre": pre, "eval_acc_final": rc.acc(agent, eval_pairs), "final_drift_pct": agent.drift_pct(),
            "n_trained": agent.n_trained, "n_skipped": agent.n_skipped}


def run_add(spec):
    cfg, seed = spec.config, spec.seed
    two = cfg["width"] == "two"
    op_max = 49 if two else 9
    rngs = {k: np.random.default_rng([seed, i]) for i, k in enumerate(["cal", "eval", "train", "pool", "probe"], 1)}
    eval_set = sample_batch(rngs["eval"], op_max, ra.N_EVAL, two)
    live = paths.live_alpn()
    enc = TwoDigitAdditionEncoder(live, seed=seed) if two else AdditionEncoder(live, n_max=9, seed=seed)
    agent = ScaledAddition(enc, "two" if two else "single", cfg["g"], cfg["f"], seed)
    agent.calibrate_baseline([t[:3] for t in sample_batch(rngs["cal"], op_max, ra.N_CALIB, two)])
    pre = ra.accuracy(agent, eval_set)
    for _ in range(cfg["rounds"]):
        a, b, c, label = sample_triple(rngs["train"], op_max, rngs["train"].random() < 0.5, two)
        agent.update(a, b, c, label)
    return {"eval_acc_pre": pre, "eval_acc_final": ra.accuracy(agent, eval_set), "final_drift_pct": agent.drift_pct(),
            "n_trained": agent.n_trained, "n_skipped": agent.n_skipped}


def run_one(spec, log):
    fam = spec.config["family"]
    if fam == "maze":
        return rg.run_one(spec, log)
    return run_cmp(spec) if fam == "cmp" else run_add(spec)


def main():
    args = common_parser(__doc__).parse_args()
    if args.tuning:
        raise SystemExit("no tuning mode")
    specs = []
    small = {"cmp_digits_5": SMALL_TASKS["cmp_digits_5"], "add_single": SMALL_TASKS["add_single"]} if args.smoke \
        else SMALL_TASKS
    combos = [(1.0, 1.0), (0.5, 0.5)] if args.smoke else [(g, f) for g in GRID_G for f in GRID_F]
    for name, t in small.items():
        t = dict(t, rounds=60) if args.smoke else t
        for g, f in combos:
            for seed in resolve_seeds(args):
                specs.append(RunSpec(EXPERIMENT, name, "connectome", f"g{g}_f{f}", seed, {**t, "g": g, "f": f}))
    maze = "maze26" if args.smoke else "maze72"
    _, _, max_steps, episodes = rg.MAZES[maze]
    for g, f in ([(1.0, 1.0)] if args.smoke else MAZE_COMBOS):
        for seed in (args.seeds or ([0] if args.smoke else MAZE_SEEDS)):
            cfg = {"family": "maze", "plastic": True, "maze": maze, "start": "rs", "max_steps": max_steps,
                   "episodes": 30 if args.smoke else episodes, "gamma": rg.GAMMA, "min_pulls": rg.MIN_PULLS,
                   "g": g, "f": f, "conn": {"encoding": "khot", "gate": "none"}}
            specs.append(RunSpec(EXPERIMENT, f"{maze}_rs", "connectome", f"g{g}_f{f}", seed, cfg))
    run_all(specs, run_one, args,
            cost=lambda s: (s.config["episodes"] * s.config["max_steps"] if s.config["family"] == "maze"
                            else s.config["rounds"]))


if __name__ == "__main__":
    main()
