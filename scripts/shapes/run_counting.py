"""Main experiment runner for the circle-counting stage.

Adds CLI knobs (--eta, --trials, --n-out) so the "basic improvement" test
-- push the learning rate/trial count harder now that output_drift_pct()
shows the real (not whole-brain-diluted) picture -- doesn't need a code
edit. Try e.g.:

    python scripts/shapes/run_counting.py --eta 1.0 --trials 6
    python scripts/shapes/run_counting.py --eta 1.5 --trials 8 --n-out 16

before concluding a tuning fix isn't enough and moving to tier2/imitation.

Logs BOTH drift_pct (whole-brain average, mostly meaningless here -- see
connectome_count_agent.py's output_drift_pct docstring) and
output_drift_pct (the ~512 edges that actually matter) so you can see
directly whether more eta/trials is moving the weights that matter.
"""
import sys
import os
import csv
import time
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_d = _HERE
for _ in range(6):
    if _d not in sys.path:
        sys.path.insert(0, _d)
    _d = os.path.dirname(_d)

import numpy as np

from encoders.shapes.circle_shapes import sample_random, circle_bitmap
from agents.shapes.connectome_count_agent import ConnectomeCountAgent
from agents.shapes.baseline_count_agents import LinearCountAgent, MLPCountAgent

N_ROUNDS = 400
N_MAX = 4
IMG_SIZE = 32
RADIUS = 3
SLEEP_EVERY = 50
PROGRESS_EVERY = 20
N_CALIB_PER_COUNT = 4


def _find_results_dir():
    d = _HERE
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "results")):
            return os.path.join(d, "results", "counting")
        d = os.path.dirname(d)
    return os.path.join(os.path.dirname(os.path.dirname(_HERE)), "results", "counting")


RESULTS_DIR = _find_results_dir()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, default=0.3, help="x_teach_output step size")
    ap.add_argument("--trials", type=int, default=3, help="x_teach_output trials per train_on call")
    ap.add_argument("--n-out", type=int, default=8, help="neurons in the 'count' output")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    n_pixels = IMG_SIZE * IMG_SIZE

    connectome = ConnectomeCountAgent(n_max=N_MAX, n_out=a.n_out, teach_eta=a.eta)
    calib_rng = np.random.default_rng(1)
    calib_bitmaps = [circle_bitmap(n, calib_rng, size=IMG_SIZE, radius=RADIUS)
                      for n in range(N_MAX + 1) for _ in range(N_CALIB_PER_COUNT)]
    connectome.calibrate_baseline(calib_bitmaps)  # BEFORE any train_on call

    agents = {
        "connectome": connectome,
        "linear": LinearCountAgent(n_pixels, n_max=N_MAX),
        "mlp": MLPCountAgent(n_pixels, n_max=N_MAX),
    }

    suffix = f"_{a.tag}" if a.tag else ""
    csv_path = os.path.join(RESULTS_DIR, f"counting_results{suffix}.csv")
    t0 = time.time()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "agent", "count", "predicted", "correct", "abs_err",
                    "drift_pct", "output_drift_pct"])

        for rnd in range(N_ROUNDS):
            count, bitmap = sample_random(rng, n_min=0, n_max=N_MAX, size=IMG_SIZE, radius=RADIUS)

            for name, agent in agents.items():
                pred, _ = agent.predict(bitmap)
                correct = int(pred == count)
                abs_err = abs(pred - count)
                drift = f"{agent.drift_pct():.4f}" if hasattr(agent, "drift_pct") else ""
                out_drift = (f"{agent.output_drift_pct():.4f}"
                             if hasattr(agent, "output_drift_pct") else "")
                w.writerow([rnd, name, count, pred, correct, abs_err, drift, out_drift])
                if name == "connectome":
                    agent.train_on(bitmap, count, trials=a.trials)
                else:
                    agent.train_on(bitmap, count)
            f.flush()

            if hasattr(agents["connectome"], "sleep") and (rnd + 1) % SLEEP_EVERY == 0:
                agents["connectome"].sleep()

            if (rnd + 1) % PROGRESS_EVERY == 0 or rnd == N_ROUNDS - 1:
                elapsed = time.time() - t0
                eta_s = elapsed / (rnd + 1) * (N_ROUNDS - rnd - 1)
                print(f"round {rnd + 1}/{N_ROUNDS} elapsed={elapsed:.0f}s eta={eta_s:.0f}s")

    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
