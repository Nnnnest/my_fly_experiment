"""python -m scripts.diagnostics.verify_addition_agent"""
import numpy as np

from core import paths
from encoders.addition.addition_encoder import AdditionEncoder
from agents.addition.connectome_addition_agent import ConnectomeAdditionAgent
from agents.shared.addition_common import sample_batch

enc = AdditionEncoder(paths.live_alpn(), n_max=9, seed=0)
ag = ConnectomeAdditionAgent(enc, "single")
rng = np.random.default_rng(0)
tr = [t[:3] for t in sample_batch(rng, 9, 70)]
ref = [ag.brain.step(odor=enc.encode(*t), hops=1)["MB_pref"] for t in tr[:10]]
print("max |cache - step|:", float(np.abs(np.array(ref) - ag._raw_many(tr[:10])).max()), "(want < 1e-3)")
ag.calibrate_baseline(tr[:60])
ev = sample_batch(rng, 9, 300)
y = np.array([t[3] for t in ev])
print("untrained accuracy:", float((ag.predict_many([t[:3] for t in ev]) == y).mean()), "(want ~0.5)")
