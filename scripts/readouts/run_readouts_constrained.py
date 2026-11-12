"""Constrained alternative readouts on identical KC codes (essential_experiments_plan.md, experiment 1).
Per task and seed, on the SAME calibration set, training stream and held-out evaluation set:
  native              the existing agent (train() on the KC->MBON weights)
  ridge_{1x,5x}       signed ridge on the KC codes (reference; = acc_ridge_kc_* of the readouts experiments)
  nonneg_{1x,5x}      one non-negative weight per KC, free bias
  anat_{1x,5x}        two non-negative channels per KC (approach / avoid), a KC feeds a channel only if it has a
                      KC->MBON edge to that population (real circuit), free bias
  rule_native         the native rule re-implemented on the same codes (must equal 'native' exactly)
  rule_no_renorm / rule_push_pull / rule_decay / rule_mask_active
                      the native rule with ONE property relaxed (see agents/shared/kc_native.apply_rule)
  rule_relax_all      all four relaxations together (interpretation aid only)
1x = same number of training examples as the online agent, 5x = five times more.
Splits: main (streams of the main addition experiment) and disjoint (held-out operand pairs).
    python -m scripts.readouts.run_readouts_constrained --smoke
    python -m scripts.readouts.run_readouts_constrained --tuning --workers 14   # choose N0_DECAY (disjoint split)
    python -m analysis.constrained_tables --tuning                              # edit N0_DECAY, commit
    python -m scripts.readouts.run_readouts_constrained --workers 14"""
import numpy as np

from core import paths
from core.runner import RunSpec, common_parser, resolve_seeds, run_all
from encoders.addition.addition_encoder import AdditionEncoder
from encoders.addition.addition_encoder_2digit import TwoDigitAdditionEncoder
from agents.shared.addition_common import sample_batch, sample_triple
from agents.shared.addition_split import split_pairs, draw
from agents.shared.readouts import prep_l2, accuracy, ridge_fit_select
from agents.shared.constrained_readouts import nnls_fit_select, nonneg_design, anatomy_design, edge_flags
from agents.addition.connectome_addition_agent_probe import ProbeAdditionAgent
from scripts.readouts.run_readouts import codes_of, LAMBDAS
from scripts.addition.run_addition_disjoint import HELD_FRAC
import scripts.addition.run_addition as ra

EXPERIMENT = "readouts_constrained"
ROUNDS, BIG = 800, 5
N0_DECAY = 10.0            # PLACEHOLDER: set from `analysis.constrained_tables --tuning`, then commit
TUNE_N0 = [3.0, 10.0, 30.0, 100.0]
WIDTHS, SPLITS = ("single", "two"), ("main", "disjoint")


def rules():
    return {"native": {}, "no_renorm": {"renorm": False}, "push_pull": {"push_pull": True},
            "decay": {"decay_n0": N0_DECAY}, "mask_active": {"mask_active": True},
            "relax_all": {"renorm": False, "push_pull": True, "decay_n0": N0_DECAY, "mask_active": True}}


def streams(cfg, seed):
    two = cfg["width"] == "two"
    op_max = 49 if two else 9
    rngs = {k: np.random.default_rng([seed, i]) for i, k in enumerate(["cal", "eval", "train", "pool", "probe"], 1)}
    n = cfg["big"] * cfg["rounds"]
    if cfg["split"] == "main":          # same streams as scripts/readouts/run_readouts.run_add
        eval_set = sample_batch(rngs["eval"], op_max, ra.N_EVAL, two)
        cal = [t[:3] for t in sample_batch(rngs["cal"], op_max, ra.N_CALIB, two)]
        train = [sample_triple(rngs["train"], op_max, rngs["train"].random() < 0.5, two) for _ in range(n)]
    else:                               # same streams as scripts/addition/run_addition_disjoint
        train_pairs, held = split_pairs(op_max, seed, HELD_FRAC)
        eval_set = draw(rngs["eval"], held, ra.N_EVAL, op_max, two)
        cal = [t[:3] for t in draw(rngs["cal"], train_pairs, ra.N_CALIB, op_max, two)]
        train = draw(rngs["train"], train_pairs, n, op_max, two)
    live = paths.live_alpn()
    enc = TwoDigitAdditionEncoder(live, seed=seed) if two else AdditionEncoder(live, n_max=9, seed=seed)
    return enc, eval_set, cal, train


def run_one(spec, log):
    cfg, seed = spec.config, spec.seed
    rounds = cfg["rounds"]
    enc, eval_set, cal, train = streams(cfg, seed)

    def fresh(rule=None):
        agent = ProbeAdditionAgent(enc, cfg["width"], rule=rule)
        agent.calibrate_baseline(cal)                                  # BEFORE any training
        return agent

    def run_online(rule):
        agent = fresh(rule)
        for a, b, c, label in train[:rounds]:
            agent.update(a, b, c, label)
        return agent, ra.accuracy(agent, eval_set)

    if cfg["mode"] == "tune":
        return {f"acc_rule_decay_n{n0}": run_online({"decay_n0": n0})[1] for n0 in TUNE_N0}

    agent = fresh(None)
    out = {"acc_untrained": ra.accuracy(agent, eval_set)}
    for a, b, c, label in train[:rounds]:
        agent.update(a, b, c, label)
    out["acc_native"] = ra.accuracy(agent, eval_set)
    w_native = agent.brain.wM.copy()

    # offline readouts on the SAME codes
    cache, brain = agent.cache, agent.brain
    Xtr = np.stack([enc.encode(*t[:3]) for t in train])
    Xev = np.stack([enc.encode(*t[:3]) for t in eval_set])
    ytr = np.array([float(t[3]) for t in train])
    yev = np.array([float(t[3]) for t in eval_set])
    Ktr, Kev = prep_l2(codes_of(cache, Xtr)), prep_l2(codes_of(cache, Xev))
    has_app, has_avo = edge_flags(brain, cache)
    out.update(n_kc=int(len(brain.KC)), n_km_edges=int(brain.km_mask.sum()),
               n_kc_app=int(has_app.sum()), n_kc_avo=int(has_avo.sum()))
    rng = np.random.default_rng([seed, 91])                            # same sequence as run_readouts.readouts()
    for tag, n in (("1x", rounds), ("5x", len(Ktr))):
        w, lam = ridge_fit_select(Ktr[:n], ytr[:n], LAMBDAS, rng)
        out[f"acc_ridge_{tag}"], out[f"lam_ridge_{tag}"] = accuracy(Kev, yev, w), float(lam)
    designs = {"nonneg": (nonneg_design(Ktr), nonneg_design(Kev)),
               "anat": (anatomy_design(Ktr, has_app, has_avo), anatomy_design(Kev, has_app, has_avo))}
    for j, (name, ((Ztr, nn), (Zev, _))) in enumerate(designs.items()):
        rng = np.random.default_rng([seed, 92 + j])
        for tag, n in (("1x", rounds), ("5x", len(Ztr))):
            w, lam = nnls_fit_select(Ztr[:n], ytr[:n], LAMBDAS, rng, nn)
            out[f"acc_{name}_{tag}"], out[f"lam_{name}_{tag}"] = accuracy(Zev, yev, w), float(lam)
            out[f"nnz_{name}_{tag}"] = int((w[nn] > 1e-9).sum())
    out["n_params_ridge"] = out["n_params_nonneg"] = int(Ktr.shape[1])
    out["n_params_anat"] = int(has_app.sum() + has_avo.sum() + 1)

    # native rule and one-at-a-time relaxations, same codes, same stream
    for name, rule in rules().items():
        ag, acc = run_online(rule)
        out[f"acc_rule_{name}"] = acc
        out[f"drift_rule_{name}"] = float(ag.drift_pct())
        if name == "native":
            out["max_dw_replica"] = float(np.abs(ag.brain.wM - w_native).max())
    return out


def main():
    args = common_parser(__doc__).parse_args()
    small = args.smoke
    base = {"rounds": 60 if small else ROUNDS, "big": BIG, "mode": "tune" if args.tuning else "full"}
    widths = ["single"] if small else WIDTHS
    splits = ("disjoint",) if (small or args.tuning) else SPLITS
    specs = [RunSpec(EXPERIMENT, f"{w}_{sp}", "readouts_constrained", "all", seed, {**base, "width": w, "split": sp})
             for w in widths for sp in splits for seed in resolve_seeds(args)]
    run_all(specs, run_one, args, cost=lambda s: s.config["rounds"] * 8)


if __name__ == "__main__":
    main()
