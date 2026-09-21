"""Capacity probe (real circuit, a few minutes per maze size).

Question it answers: at what number of (state, action) pairs does each scheme stop being
able to hold the task, and is the limit the KC representation or the learning rule?

For each maze size it reports
  * ceiling  - best possible linear readout on the fixed KC codes (unconstrained perceptron)
  * online   - policy accuracy of the real train() rule after supervised sweeps, for the
               always-train baseline and error-driven gates at several margins
               (peak accuracy over the sweeps, final accuracy, and trains used)

Reading it:
  ceiling high, online low  -> representation is fine, the sign-gated rule is the bottleneck
                               (a different readout/learning rule could fix it)
  ceiling low               -> the KC codes cannot hold this many pairs (capacity of the circuit)
  online rank/error ~ ceiling up to N states -> N is the usable capacity to report

    python capacity_probe_v2.py                          # rank gate, margins .03 .05 .1 .2, 4 sizes
    python capacity_probe_v2.py --dims 19x13 --sweeps 60 --margins 0.05 0.1 --gates rank error
"""
import sys, os, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "agents"))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from fly_api import FlyBrainAPI
from mb_value_cache import MBValueCache, train_no_readout, calibrate_own_effect, ensure_live_hops1
import diagnose_gridworld_v2 as D


def run_online(brain, task, gate, mf, sweeps, seed=0):
    """Supervised sweeps with one gate; returns (peak_acc, peak_sweep, final_acc, total_trains)."""
    out = D.sweep_test(brain, None, n_sweeps=sweeps, task=task, seed=seed,
                       configs=[(f"{gate} m={mf}", gate, 0.0, mf)] if gate != "none" else [("always", "none", 0.0)],
                       quiet=True)
    hist = list(out.values())[0]
    accs = [a for _, a, _ in hist]
    return max(accs), int(np.argmax(accs)) + 1, accs[-1], sum(n for _, _, n in hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", nargs="+", default=["9x7", "13x9", "15x11", "19x13"])
    ap.add_argument("--sweeps", type=int, default=30)
    ap.add_argument("--margins", type=float, nargs="+", default=[0.03, 0.05, 0.1, 0.2])
    ap.add_argument("--gates", nargs="+", default=["rank"], choices=["rank", "error"])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-overlap", type=int, default=1)
    ap.add_argument("--skip-always", action="store_true", help="skip the slow always-train baseline")
    a = ap.parse_args()

    brain = FlyBrainAPI(mode="mb")
    live = ensure_live_hops1(os.path.join(SCRIPT_DIR, "live_alpn_indices.npy"), brain)
    w0 = brain.wM.copy()
    print(f"live ALPN={len(live)}; sweeps={a.sweeps}\n")
    rows = []
    for dim in a.dims:
        w, h = (int(x) for x in dim.lower().split("x"))
        task = D.build_task(brain, live, maze=(w, h), k=a.k, max_overlap=a.max_overlap)
        ns = task["ns"]
        cl = D.ceiling_test(task)
        print(f"--- maze {dim}: states={ns}, pairs={ns * 4} ---")
        print(f"    ceiling (unconstrained linear readout): acc={cl['accuracy']:.2f} "
              f"({'converged' if cl['converged'] else 'not converged'}, {cl['epochs']} epochs)")
        res = {}
        if not a.skip_always:
            brain.wM[:] = w0
            res["always"] = run_online(brain, task, "none", 0, a.sweeps)
        for gate in a.gates:
            for mf in a.margins:
                brain.wM[:] = w0
                res[f"{gate} m={mf}"] = run_online(brain, task, gate, mf, a.sweeps)
        for name, (pk, pks, fin, tr) in res.items():
            print(f"    {name:12s} peak={pk:.2f} (sweep {pks:2d})  final={fin:.2f}  trains={tr}")
        rows.append((dim, ns, cl["accuracy"], {k: v[0] for k, v in res.items()}, {k: v[2] for k, v in res.items()}))
        brain.wM[:] = w0
        print()

    print("=== Summary: states | ceiling | best online peak | best online FINAL (which scheme) ===")
    for dim, ns, ceil, peaks, finals in rows:
        bp = max(peaks, key=peaks.get); bf = max(finals, key=finals.get)
        print(f"  {dim:6s} states={ns:4d}  ceiling={ceil:.2f}  peak={peaks[bp]:.2f} ({bp})  final={finals[bf]:.2f} ({bf})")


if __name__ == "__main__":
    main()
