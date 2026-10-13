import json
import os

import pandas as pd

from core import paths


def load_manifests(experiment, root=None):
    root = root or paths.RUNS_DIR
    base = os.path.join(root, experiment)
    rows = []
    for rid in sorted(os.listdir(base)):
        mp = os.path.join(base, rid, "manifest.json")
        if os.path.exists(mp):
            with open(mp) as f:
                m = json.load(f)
            rows.append({k: m.get(k) for k in ("run_id", "task", "agent", "tier", "seed", "status", "runtime_s")})
    return pd.DataFrame(rows)


def load_metrics(experiment, metric=None, root=None):
    """Metrics of all OK runs, joined with task/agent/tier/seed."""
    root = root or paths.RUNS_DIR
    man = load_manifests(experiment, root)
    frames = []
    for _, r in man[man.status == "ok"].iterrows():
        df = pd.read_csv(os.path.join(root, experiment, r.run_id, "metrics.csv"))
        if metric:
            df = df[df.metric == metric]
        for c in ("task", "agent", "tier", "seed"):
            df[c] = r[c]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def load_summaries(experiment, root=None):
    """One row per OK run: manifest columns + everything in summary.json."""
    root = root or paths.RUNS_DIR
    man = load_manifests(experiment, root)
    rows = []
    for _, r in man[man.status == "ok"].iterrows():
        with open(os.path.join(root, experiment, r.run_id, "summary.json")) as f:
            rows.append({**r.to_dict(), **json.load(f)})
    return pd.DataFrame(rows)
