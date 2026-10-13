"""KC-code overlap: does the code of the correct triple differ from 'off by d' triples?
    python -m scripts.addition.run_addition_diagnostic --smoke
    python -m scripts.addition.run_addition_diagnostic"""
import numpy as np

from core import paths
from core.brain_pool import fresh_brain, value_fn
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from encoders.addition.addition_encoder import AdditionEncoder
from encoders.addition.addition_encoder_2digit import TwoDigitAdditionEncoder

DIFFS = [0, 1, -1, 2, -2, 3, -3]


def run_one(spec, log):
    two, seed = spec.task == "two", spec.seed
    op_max = 49 if two else 9
    fresh_brain()
    cache, _ = value_fn()
    live = paths.live_alpn()
    enc = TwoDigitAdditionEncoder(live, seed=seed) if two else AdditionEncoder(live, n_max=9, seed=seed)
    rng = np.random.default_rng([seed, 6])

    def codes(triples):
        return cache._codes(np.stack([enc.encode(*t) for t in triples])) > 0

    def jac(A, B):
        inter, union = (A & B).sum(1), (A | B).sum(1)
        return np.where(union > 0, inter / np.maximum(union, 1), 1.0)

    # random-triple baseline
    rt = [(int(rng.integers(0, op_max + 1)), int(rng.integers(0, op_max + 1)), int(rng.integers(0, 2 * op_max + 1)))
          for _ in range(60)]
    B = codes(rt).astype(np.float32)
    inter = B @ B.T
    sz = B.sum(1)
    union = sz[:, None] + sz[None, :] - inter
    j = np.where(union > 0, inter / np.maximum(union, 1e-9), 1.0)[np.triu_indices(len(B), 1)]
    log.log(0, "jaccard_random_mean", j.mean())
    log.log(0, "jaccard_random_p95", np.quantile(j, 0.95))
    log.log(0, "empty_code_frac", float((sz == 0).mean()))
    # diff sweep: correct triple vs c moved away from the true sum
    pairs = [(int(rng.integers(0, op_max + 1)), int(rng.integers(0, op_max + 1))) for _ in range(spec.config["n_pairs"])]
    out = {"jaccard_random_mean": float(j.mean())}
    for d in DIFFS:
        base, other = [], []
        for a, b in pairs:
            c = a + b - d
            if 0 <= c <= 2 * op_max:
                base.append((a, b, a + b))
                other.append((a, b, c))
        v = float(jac(codes(base), codes(other)).mean())
        log.log(0, f"jaccard_diff{d:+d}", v)
        out[f"jaccard_diff{d:+d}"] = v
    return out


def main():
    args = common_parser(__doc__).parse_args()
    n = 10 if args.smoke else 40
    specs = [RunSpec("addition_diag", w, "connectome", "diagnostic", s, {"n_pairs": n})
             for w in ("single", "two") for s in resolve_seeds(args)]
    run_all(specs, run_one, args)


if __name__ == "__main__":
    main()
