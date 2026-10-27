"""python -m analysis.eval_overlap
Replays the sampling code of the main number-comparison and addition experiments (same random streams, no agents
are run) and measures how many evaluation items also occur in the online training stream. Use it to verify the
overlap percentages quoted in the paper.
 comparison: eval pairs = stream [seed, 1] (200), training stream = stream [seed, 3]; 400 rounds (main) and
             2000 (= 5x, as used by the 5x ridge readouts).
 addition:   eval triples = stream [seed, 2] (300), training stream = stream [seed, 3]; 800 rounds (main) and
             4000 (5x). Imitation pools are not replayed."""
import numpy as np
import pandas as pd

from core import paths  # noqa: F401
from core.config import FINAL_SEEDS
from analysis.common import cells, save_table
from scripts.comparison.run_comparison import build, KNOB_N, DIGITS, N_EVAL
from agents.shared.addition_common import sample_batch, sample_triple


def comparison_rows():
    tasks = {f"knob_n{n}": {"kind": "knob", "n_max": n} for n in KNOB_N}
    tasks.update({f"digits_{d}": {"kind": "digits", "n_digits": d} for d in DIGITS})
    rows = []
    for task, cfg in tasks.items():
        for seed in FINAL_SEEDS:
            _, sample, _ = build(cfg, seed)
            r_eval, r_tr = np.random.default_rng([seed, 1]), np.random.default_rng([seed, 3])
            ev = [sample(r_eval) for _ in range(N_EVAL)]
            stream = [sample(r_tr) for _ in range(2000)]
            for n in (400, 2000):
                tr = set(stream[:n])
                tr_either = tr | {(b, a) for a, b in tr}
                rows.append({"task": task, "n_train": n, "seed": seed,
                             "frac_exact": float(np.mean([p in tr for p in ev])),
                             "frac_either_orientation": float(np.mean([p in tr_either for p in ev]))})
    return pd.DataFrame(rows)


def addition_rows():
    rows = []
    for width in ("single", "two"):
        two, op_max = width == "two", 49 if width == "two" else 9
        for seed in FINAL_SEEDS:
            rngs = {k: np.random.default_rng([seed, i])
                    for i, k in enumerate(["cal", "eval", "train", "pool", "probe"], 1)}
            ev = sample_batch(rngs["eval"], op_max, 300, two)
            stream = []
            for _ in range(4000):
                a, b, c, label = sample_triple(rngs["train"], op_max, rngs["train"].random() < 0.5, two)
                stream.append((a, b, c))
            for n in (800, 4000):
                tr = set(stream[:n])
                seen = np.array([t[:3] in tr for t in ev])
                pos = np.array([t[3] for t in ev])
                rows.append({"task": f"addition_{width}", "n_train": n, "seed": seed,
                             "frac_all": float(seen.mean()), "frac_positives": float(seen[pos].mean())})
    return pd.DataFrame(rows)


def main():
    c = comparison_rows()
    t = cells(c, ["task", "n_train"], ["frac_exact", "frac_either_orientation"], 3).drop(columns="n")
    save_table(t, "eval_overlap_comparison",
               "Fraction of evaluation pairs that also occur in the online training stream (main experiment replay)", 2)
    a = addition_rows()
    t = cells(a, ["task", "n_train"], ["frac_all", "frac_positives"], 3).drop(columns="n")
    save_table(t, "eval_overlap_addition",
               "Fraction of evaluation triples (all / positives) that also occur in the online training stream", 2)


if __name__ == "__main__":
    main()
