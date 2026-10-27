"""The disjoint splits must (1) contain no evaluation pair (either orientation) in the training or calibration
streams, and (2) reuse the evaluation pairs of the main comparison experiment for the same seed.
    python -m scripts.diagnostics.verify_disjoint_split"""
import numpy as np

from core import paths  # noqa: F401
from core.config import FINAL_SEEDS
from scripts.comparison.run_comparison import build, N_EVAL, ROUNDS
from scripts.comparison.run_comparison_disjoint import TASKS, make_split, MAX_MULT

for task, tcfg in TASKS.items():
    n_unique = []
    for seed in FINAL_SEEDS:
        enc, scale, ev, cal, tr, forb = make_split(tcfg, seed, MAX_MULT * ROUNDS)
        _, sample, _ = build(tcfg, seed)
        r = np.random.default_rng([seed, 1])
        ref = [sample(r) for _ in range(N_EVAL)]
        assert ref == ev, f"{task} seed {seed}: evaluation pairs differ from the main experiment"
        assert all(p not in forb for p in tr + cal), f"{task} seed {seed}: overlap"
        assert not (set(tr) & set(ev)), f"{task} seed {seed}: exact overlap"
        n_unique.append(len(set(tr)))
    print(f"{task}: OK, 10 seeds; unique training pairs in the 5x stream: {min(n_unique)} to {max(n_unique)} "
          f"(of {MAX_MULT * ROUNDS})")
print("OK")
