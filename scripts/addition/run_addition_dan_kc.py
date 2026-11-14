"""Training-time versus input-only KC activity (addition).
During train() the forward pass includes the dopaminergic (US) drive; the exact cache and all fitted readouts use
odour-only KC codes. This experiment quantifies what the DAN drive does to KC activity, to the top-k plasticity mask,
to the MBON readout and to the weights changed by one update, and runs one focused ablation in which the native update
uses the odour-only mask (everything else unchanged).
  tasks   single_measure, two_measure   (tier 'measure': diagnostic, 200 triples per seed)
          single_online,  two_online    (tiers 'native' = replica of the main runs, 'odor_mask' = ablation)
    python -m scripts.addition.run_addition_dan_kc --smoke
    python -m scripts.addition.run_addition_dan_kc --workers 14"""
import numpy as np

from core import paths
from core.brain_pool import fresh_brain, value_fn
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from encoders.addition.addition_encoder import AdditionEncoder
from encoders.addition.addition_encoder_2digit import TwoDigitAdditionEncoder
from agents.shared.addition_common import sample_batch
from agents.shared.mb_value_cache import train_no_readout
from agents.addition.connectome_addition_agent import ConnectomeAdditionAgent
import scripts.addition.run_addition as ra

EXPERIMENT = "addition_dan_kc"
KFRAC = 0.05
N_ITEMS = 200


def topk_mask(kcs):
    """Exactly the mask of FlyBrainAPI.train()/step(): kcs >= k-th largest value (ties are kept)."""
    k = max(1, int(len(kcs) * KFRAC))
    return kcs >= np.sort(kcs)[-k]


def sim_kc(brain, vec, rew=0.0, pun=0.0):
    """KC activity from the simulator's own forward pass, with optional US drive (as inside train())."""
    idx, val, _ = brain.encode(odor=vec, dan_rew=rew, dan_pun=pun)
    h = brain._forward_pure_mb(idx, val, hops=1, thr=0.0)
    return np.asarray(h[brain.KC], np.float64)


def mbon_readout(brain, kcs):
    m = topk_mask(kcs)
    ks = kcs * m
    w = brain.wM[brain.km_mask].astype(np.float64)
    r = np.bincount(brain.km_mi, weights=ks[brain.km_ki] * w, minlength=len(brain.MBON))
    return r, m


def train_odor_mask(brain, odor, **kw):
    """Native train() update, but the KC top-k plasticity mask comes from the ODOUR-ONLY forward pass.
    DAN activity (the gate) still comes from the real forward pass with US drive. fly_api.py is not touched."""
    idx0, val0, _ = brain.encode(odor=odor)
    orig = brain._forward_pure_mb

    def fwd(idx, val, hops=1, thr=0.0):
        h = np.array(orig(idx, val, hops=hops, thr=thr))
        h0 = orig(idx0, val0, hops=hops, thr=thr)
        h[brain.KC] = h0[brain.KC]
        return h

    brain._forward_pure_mb = fwd
    try:
        train_no_readout(brain, odor, hops=1, **kw)
    finally:
        del brain._forward_pure_mb


def apply_update(brain, vec, w0, odor_mask, **kw):
    brain.wM[:] = w0
    brain._km_tag = set()
    (train_odor_mask if odor_mask else lambda b, o, **k: train_no_readout(b, o, hops=1, **k))(brain, vec, **kw)
    w = brain.wM.copy()
    brain.wM[:] = w0
    brain._km_tag = set()
    return w


def compare(k0, kd, r0, rd, m0, md, c):
    inter, union = int((m0 & md).sum()), int((m0 | md).sum())
    tol = 1e-6 * max(float(k0.max()), 1e-12)
    n0 = max(float(np.linalg.norm(k0)), 1e-12)
    r0n = max(float(np.linalg.norm(r0)), 1e-12)
    return {"frac_kc_changed": float((np.abs(kd - k0) > tol).mean()),
            "cont_rel_l2": float(np.linalg.norm(kd - k0) / n0),
            "cont_max_abs": float(np.abs(kd - k0).max()),
            "jaccard": inter / union if union else 1.0,
            "mask_differs": float(inter != union),
            "n_sel": float(md.sum()),
            "n_sel_delta": float(md.sum() - m0.sum()),
            "mbon_rel_l2": float(np.linalg.norm(rd - r0) / r0n),
            "mbpref_abs_diff": float(abs(c @ rd - c @ r0))}


def run_measure(spec, log):
    cfg, seed = spec.config, spec.seed
    two = cfg["width"] == "two"
    op_max = 49 if two else 9
    brain = fresh_brain()
    cache, _ = value_fn()
    live = paths.live_alpn()
    enc = TwoDigitAdditionEncoder(live, seed=seed) if two else AdditionEncoder(live, n_max=9, seed=seed)
    triples = sample_batch(np.random.default_rng([seed, 12]), op_max, cfg["n_items"], two)
    w0 = brain.wM.copy()
    c = cache.c
    res = {"reward": [], "punish": [], "as_trained": []}
    base = {"none_n_sel": [], "mbpref_scale": [], "cache_mask_jaccard": [], "cache_max_abs": []}
    for a, b, cc, label in triples:
        vec = enc.encode(a, b, cc)
        k0 = sim_kc(brain, vec)
        r0, m0 = mbon_readout(brain, k0)
        code = cache._codes(vec[None, :])[0].astype(np.float64)
        pm, qm = code > 0, (k0 * m0) > 0
        base["none_n_sel"].append(float(m0.sum()))
        base["mbpref_scale"].append(float(abs(c @ r0)))
        base["cache_mask_jaccard"].append(float((pm & qm).sum() / max((pm | qm).sum(), 1)))
        base["cache_max_abs"].append(float(np.abs(code - k0 * m0).max()))
        item = {}
        for cond, kw, rew, pun in (("reward", {"reward": 1.0}, 1.0, 0.0), ("punish", {"punish": 1.0}, 0.0, 1.0)):
            kd = sim_kc(brain, vec, rew=rew, pun=pun)
            rd, md = mbon_readout(brain, kd)
            m = compare(k0, kd, r0, rd, m0, md, c)
            w_nat = apply_update(brain, vec, w0, False, **kw)
            w_odo = apply_update(brain, vec, w0, True, **kw)
            upd = float(np.abs(w_nat - w0).sum())
            m.update(n_changed_native=float((w_nat != w0).sum()), n_changed_odor=float((w_odo != w0).sum()),
                     n_edges_differ=float((w_nat != w_odo).sum()),
                     rel_update_diff=float(np.abs(w_nat - w_odo).sum() / max(upd, 1e-12)))
            item[cond] = m
            res[cond].append(m)
        res["as_trained"].append(item["reward"] if label else item["punish"])
    out = {"n_items": len(triples), "frac_positive": float(np.mean([t[3] for t in triples]))}
    for k, v in base.items():
        out[k] = float(np.mean(v))
    for cond, rows in res.items():
        for key in rows[0]:
            vals = np.array([r[key] for r in rows], float)
            out[f"{cond}_{key}"] = float(vals.mean())
        out[f"{cond}_jaccard_min"] = float(min(r["jaccard"] for r in rows))
    for k, v in out.items():
        log.log(0, k, v, split="eval")
    return out


class OdorMaskAddition(ConnectomeAdditionAgent):
    """Native agent whose plasticity mask is the odour-only top-k set (readout cache unchanged)."""

    def update(self, a, b, c, label):
        kw = {"reward": 1.0} if label else {"punish": 1.0}
        train_odor_mask(self.brain, self.encoder.encode(a, b, c), **kw)
        self.cache.refresh()


_orig_make_agent = ra.make_agent


def _make_agent(spec, op_max, two, rngs):
    if spec.config.get("odor_mask"):
        live = paths.live_alpn()
        enc = TwoDigitAdditionEncoder(live, seed=spec.seed) if two else AdditionEncoder(live, n_max=9, seed=spec.seed)
        agent = OdorMaskAddition(enc, "two" if two else "single")
        agent.calibrate_baseline([t[:3] for t in sample_batch(rngs["cal"], op_max, ra.N_CALIB, two)])
        return agent
    return _orig_make_agent(spec, op_max, two, rngs)


ra.make_agent = _make_agent     # ra.run_one resolves make_agent by name at call time


def run_one(spec, log):         # wrapper so that spawned workers import THIS module (and therefore the patch above)
    return run_measure(spec, log) if "n_items" in spec.config else ra.run_one(spec, log)


def main():
    args = common_parser(__doc__).parse_args()
    if args.tuning:
        raise SystemExit("no tuning mode: constants are those of the main addition experiment")
    small = args.smoke
    base = {"rounds": 80 if small else ra.ROUNDS, "reps": 1 if small else ra.REPS, "pool": 40 if small else ra.POOL,
            "n_eval": 60 if small else ra.N_EVAL, "eval_every": ra.EVAL_EVERY, "train": True}
    widths = ["single"] if small else ["single", "two"]
    specs = []
    for w in widths:
        for seed in resolve_seeds(args):
            specs.append(RunSpec(EXPERIMENT, f"{w}_measure", "connectome", "measure", seed,
                                 {"width": w, "n_items": 20 if small else N_ITEMS}))
            for tier, extra in (("native", {}), ("odor_mask", {"odor_mask": True})):
                specs.append(RunSpec(EXPERIMENT, f"{w}_online", "connectome", tier, seed,
                                     {**base, "width": w, "mode": "online", **extra}))
    run_all(specs, run_one, args, cost=lambda s: 4 if "n_items" in s.config else 2)


if __name__ == "__main__":
    main()
