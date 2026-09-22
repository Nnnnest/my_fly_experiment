import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fly_api import FlyBrainAPI
from agents.shared.paths import FLY_ROOT
import numpy as np

rng = np.random.default_rng(0)
mb = FlyBrainAPI(mode="mb", path=FLY_ROOT)
probs = {"A": 0.3, "B": 0.7}

for t in range(150):
    for odor in ["A", "B"]:          # strict round robin — no policy, no argmax
        reward = float(rng.random() < probs[odor])
        mb.train(odor=odor, reward=reward)
    if t % 10 == 0:
        pa = mb.step(odor="A")["MB_pref"]
        pb = mb.step(odor="B")["MB_pref"]
        print(f"t={t:3d}  A={pa:8.2f}  B={pb:8.2f}  B_ahead={pb > pa}")
