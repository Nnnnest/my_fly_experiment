import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from core import paths
from core.config import FINAL_SEEDS, TUNING_SEEDS


@dataclass
class RunSpec:
    experiment: str
    task: str      # task variant, e.g. "mp10", "maze113"
    agent: str     # "qlearning" | "mlp" | "connectome" | ...
    tier: str      # "fair" | "opt" | "assisted" | "imitation" | "-"
    seed: int
    config: dict = field(default_factory=dict)

    @property
    def run_id(self):
        return f"{self.task}__{self.agent}__{self.tier}__s{self.seed:04d}"

    @property
    def config_hash(self):
        blob = json.dumps({"task": self.task, "agent": self.agent, "tier": self.tier,
                           "config": self.config}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]


class RunLogger:
    """Long format: every task logs (step, split, metric, value)."""
    def __init__(self):
        self.rows = []

    def log(self, step, metric, value, split="train"):
        self.rows.append((int(step), split, metric, float(value)))


_GIT = {}


def _git(path):
    if path not in _GIT:
        def run(*a):
            return subprocess.run(["git", "-C", path, *a], capture_output=True, text=True).stdout.strip()
        _GIT[path] = {"commit": run("rev-parse", "HEAD") or None,
                      "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    return _GIT[path]


def _versions():
    v = {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}
    try:
        import torch
        v["torch"] = torch.__version__
        v["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    return v


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def _dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def execute(spec, run_fn, root, overwrite=False):
    d = os.path.join(root, spec.experiment, spec.run_id)
    mpath = os.path.join(d, "manifest.json")
    if os.path.exists(mpath) and not overwrite:
        with open(mpath) as f:
            old = json.load(f)
        if old.get("config_hash") != spec.config_hash:
            raise RuntimeError(f"{d} exists with a DIFFERENT config; use --overwrite or change the task name")
        if old.get("status") == "ok":
            return "cached"
    os.makedirs(d, exist_ok=True)
    manifest = {"experiment": spec.experiment, "run_id": spec.run_id, "task": spec.task,
                "agent": spec.agent, "tier": spec.tier, "seed": spec.seed, "config": spec.config,
                "config_hash": spec.config_hash, "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "repo": _git(paths.REPO_ROOT), "simulator": _git(paths.FLY_ROOT), **_versions()}
    _dump(mpath, manifest)            # stays "running" if interrupted with Ctrl-C
    seed_everything(spec.seed)
    log, t0, summary = RunLogger(), time.time(), {}
    try:
        summary = run_fn(spec, log) or {}
        manifest["status"] = "ok"
    except Exception:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
    manifest["runtime_s"] = round(time.time() - t0, 2)
    with open(os.path.join(d, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "split", "metric", "value"])
        w.writerows(log.rows)
    _dump(os.path.join(d, "summary.json"), summary)
    _dump(mpath, manifest)
    return manifest["status"]


def common_parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--tuning", action="store_true", help="tuning seeds, stored in runs/_tuning")
    ap.add_argument("--smoke", action="store_true", help="tiny run, stored in runs/_smoke")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    return ap


def resolve_seeds(args):
    if args.seeds:
        return args.seeds
    if args.smoke:
        return [0]
    return TUNING_SEEDS if args.tuning else FINAL_SEEDS


def _init_worker():
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def run_all(specs, run_fn, args, cost=None):
    root = os.path.join(paths.RUNS_DIR, "_smoke" if args.smoke else "_tuning" if args.tuning else "")
    if cost:                                   # longest jobs first: avoids a long tail at the end
        specs = sorted(specs, key=cost, reverse=True)
    n, counts, t_start = len(specs), {}, time.time()

    def report(i, rid, status):
        counts[status] = counts.get(status, 0) + 1
        print(f"[{i}/{n}] {rid}: {status} (t={time.time() - t_start:.0f}s)", flush=True)

    _init_worker()
    if args.workers <= 1:
        for i, s in enumerate(specs, 1):
            report(i, s.run_id, execute(s, run_fn, root, args.overwrite))
    else:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[k] = "1"                # inherited by the spawned workers
        ex = ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker)
        try:
            futs = {ex.submit(execute, s, run_fn, root, args.overwrite): s for s in specs}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    status = f.result()
                except Exception as e:
                    status = f"error: {e}"
                report(i, futs[f].run_id, status)
        except KeyboardInterrupt:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown()
    print("done:", counts, "| output:", root)
