"""Feature-map quality probe: real connectome KC codes vs random sparse codes.

Why: kc_delta (real KC codes + delta rule) did WORSE than rand_delta (random sparse codes of
the same size and sparsity) on the 31x21 maze. This probe measures, without running any
learning, how the two feature maps differ, using quantities that control how a linear
readout learns:

  cos_mean / cos_p99 / cos_max : similarity between different states' unit-norm codes; an
                                 update at one state moves the value of every other state
                                 by (step size x error x cosine)
  eff_units                    : effective number of KCs per code, (sum x)^2 / sum x^2; low =
                                 the code leans on a few dominant KCs
  kc_load_max / kc_load_p99    : how many states use the busiest KCs (hubs); ideal is ~ mean
  kc_used                      : fraction of KCs used by at least one state
  lam_max / lam_min            : extreme eigenvalues of the state-similarity (Gram) matrix;
                                 1/lam_max is a rough ceiling for a stable delta-rule step
                                 size, lam_min sets how slowly the hardest direction learns
  eff_rank                     : (sum lam)^2 / sum lam^2, how many independent state
                                 directions the codes really provide

    python feature_probe_v2.py                       # 113, 300, 600 states
    python feature_probe_v2.py --states 113 300 --k 6 --max-overlap 1
"""
import sys, os, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "agents"))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from fly_api import FlyBrainAPI
from sparse_encoder import SparseStateActionEncoder
from mb_value_cache import MBValueCache, ensure_live_hops1


def stats(C):
    n = len(C)
    X = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    G = X @ X.T
    off = G[np.triu_indices(n, 1)]
    lam = np.linalg.eigvalsh(G)
    lam = np.clip(lam, 0.0, None)
    load = (C > 0).sum(axis=0)
    used = load > 0
    return dict(
        cos_mean=float(off.mean()), cos_p99=float(np.quantile(off, 0.99)), cos_max=float(off.max()),
        eff_units=float(((C.sum(1) ** 2) / np.maximum((C ** 2).sum(1), 1e-12)).mean()),
        kc_load_max=int(load.max()), kc_load_p99=float(np.quantile(load[used], 0.99)),
        kc_load_mean=float(load[used].mean()), kc_used=float(used.mean()),
        lam_max=float(lam.max()), lam_min=float(lam[lam > 1e-9].min()) if (lam > 1e-9).any() else 0.0,
        eff_rank=float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, nargs="+", default=[113, 300, 600])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-overlap", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    brain = FlyBrainAPI(mode="mb")
    live = ensure_live_hops1(os.path.join(SCRIPT_DIR, "live_alpn_indices.npy"), brain)
    rng = np.random.default_rng(a.seed)
    for ns in a.states:
        try:
            enc = SparseStateActionEncoder(ns, 1, live, k=a.k, max_overlap=a.max_overlap, seed=a.seed)
        except RuntimeError as e:
            print(f"--- {ns} states: cannot place codes ({e})\n")
            continue
        kc = MBValueCache(brain, enc).codes.astype(np.float64)
        n_kc = kc.shape[1]
        n_act = int(round((kc > 0).sum(axis=1).mean()))
        rnd = np.zeros_like(kc)
        for i in range(ns):
            rnd[i, rng.choice(n_kc, size=n_act, replace=False)] = 1.0
        print(f"--- {ns} states (KCs={n_kc}, active per code={n_act}) ---")
        rows = {"kc": stats(kc), "random": stats(rnd)}
        keys = list(rows["kc"].keys())
        print(f"    {'metric':14s} {'kc':>12s} {'random':>12s}")
        for k_ in keys:
            fmt = (lambda v: f"{v:12.4f}") if isinstance(rows["kc"][k_], float) else (lambda v: f"{v:12d}")
            print(f"    {k_:14s} {fmt(rows['kc'][k_])} {fmt(rows['random'][k_])}")
        for nm in ("kc", "random"):
            print(f"    rough stable step size ~ 1/lam_max: {nm}={1.0 / rows[nm]['lam_max']:.3f}")
        print()
    print("Reading: if kc has lower eff_units, higher kc_load_max/cos_max, larger lam_max or smaller eff_rank than random,\n"
          "the real codes are a worse feature map (hub KCs, more overlap) and that explains kc_delta < rand_delta.\n"
          "If they look the same, the gap is tuning/noise: sweep --alpha before concluding anything.")


if __name__ == "__main__":
    main()
