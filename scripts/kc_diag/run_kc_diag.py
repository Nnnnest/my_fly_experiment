"""k-hot selection and zero-input diagnostics (essential_experiments_plan.md, experiment 3).
Replays the online addition experiment of scripts/addition/run_addition.py (same random streams, same agent,
same evaluation set) with per-update instrumentation of the plasticity mask, in three tiers:
  reproduce          native train(), nothing skipped (must equal the main 'connectome fair' runs exactly)
  exclude_zero       updates are skipped when no KC is active (US drive included)
  exclude_mismatch   updates are skipped when the plasticity mask is larger than the readout code
                     (fewer than k active KCs: the k-th largest activity is 0 and the mask selects every KC)
The evaluation set is never changed. Statistics of the evaluation inputs are computed from the input-only KC code.
    python -m scripts.kc_diag.run_kc_diag --smoke
    python -m scripts.kc_diag.run_kc_diag --workers 14"""
import numpy as np

from core import paths
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from encoders.addition.addition_encoder import AdditionEncoder
from encoders.addition.addition_encoder_2digit import TwoDigitAdditionEncoder
from agents.shared.addition_common import sample_batch, sample_triple
from agents.shared.kc_native import mask_stats, kc_activity
from agents.addition.connectome_addition_agent_probe import ProbeAdditionAgent
import scripts.addition.run_addition as ra

EXPERIMENT = "kc_diag"
TIERS = [("reproduce", None), ("exclude_zero", "zero"), ("exclude_mismatch", "mismatch")]


def freq(n_pos, n_plast, n_read, n_kc, k, prefix):
    m = lambda x: float(np.mean(x)) if len(x) else float("nan")
    return {f"{prefix}frac_zero": m(n_pos == 0), f"{prefix}frac_lt_k": m(n_pos < k),
            f"{prefix}frac_mismatch": m(n_plast > n_read), f"{prefix}frac_all_kc": m(n_plast == n_kc),
            f"{prefix}frac_none": m(n_plast == 0), f"{prefix}frac_same": m(n_plast == n_read),
            f"{prefix}frac_tie_extended": m((n_plast > k) & (n_plast == n_read)),
            f"{prefix}mean_pos": m(n_pos), f"{prefix}mean_plast": m(n_plast), f"{prefix}mean_read": m(n_read)}


def run_one(spec, log):
    cfg, seed = spec.config, spec.seed
    two = cfg["width"] == "two"
    op_max = 49 if two else 9
    rngs = {k: np.random.default_rng([seed, i]) for i, k in enumerate(["cal", "eval", "train", "pool", "probe"], 1)}
    eval_set = sample_batch(rngs["eval"], op_max, cfg["n_eval"], two)
    live = paths.live_alpn()
    enc = TwoDigitAdditionEncoder(live, seed=seed) if two else AdditionEncoder(live, n_max=9, seed=seed)
    agent = ProbeAdditionAgent(enc, cfg["width"], skip=cfg["skip"], record=True)
    agent.calibrate_baseline([t[:3] for t in sample_batch(rngs["cal"], op_max, ra.N_CALIB, two)])   # BEFORE training
    pre = ra.accuracy(agent, eval_set)
    ev = mask_stats(kc_activity(agent.cache, np.stack([enc.encode(*t[:3]) for t in eval_set])))
    n_kc, k = int(ev["n_kc"]), int(ev["k"])
    for rnd in range(1, cfg["rounds"] + 1):
        a, b, c, label = sample_triple(rngs["train"], op_max, rngs["train"].random() < 0.5, two)
        agent.update(a, b, c, label)
        r = agent.rows[-1]
        for key in ("n_pos", "n_plast", "n_read", "n_sel", "n_changed"):
            log.log(rnd, key, r[key])
    final = ra.accuracy(agent, eval_set)
    R = {key: np.array([r[key] for r in agent.rows], float) for key in agent.rows[0]}
    ex = R["skipped"] == 0
    out = {"eval_acc_pre": pre, "eval_acc_final": final, "n_kc": n_kc, "k": k,
           "n_trained": agent.n_trained, "n_skipped": agent.n_skipped, "n_cache_disagree": agent.n_cache_disagree,
           "mean_n_sel_edges": float(R["n_sel"][ex].mean()) if ex.any() else float("nan"),
           "mean_n_changed_edges": float(R["n_changed"][ex].mean()) if ex.any() else float("nan"),
           "mean_rel_change": float(R["rel_change"][ex].mean()) if ex.any() else float("nan"),
           "final_drift_pct": float(agent.drift_pct())}
    out.update(freq(R["n_pos"], R["n_plast"], R["n_read"], n_kc, k, "train_"))
    pos = R["label"] > 0.5
    out["train_frac_zero_pos"] = float((R["n_pos"][pos] == 0).mean()) if pos.any() else float("nan")
    out["train_frac_zero_neg"] = float((R["n_pos"][~pos] == 0).mean()) if (~pos).any() else float("nan")
    out.update(freq(ev["n_pos"], ev["n_plast"], ev["n_read"], n_kc, k, "eval_"))
    return out


def main():
    args = common_parser(__doc__).parse_args()
    if args.tuning:
        raise SystemExit("no tuning mode")
    base = {"rounds": 80 if args.smoke else ra.ROUNDS, "n_eval": 60 if args.smoke else ra.N_EVAL}
    widths = ["single"] if args.smoke else ["single", "two"]
    specs = [RunSpec(EXPERIMENT, f"{w}_online", "connectome", tier, seed, {**base, "width": w, "skip": skip})
             for w in widths for tier, skip in TIERS for seed in resolve_seeds(args)]
    run_all(specs, run_one, args, cost=lambda s: s.config["rounds"])


if __name__ == "__main__":
    main()
