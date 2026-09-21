import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fly_api import FlyBrainAPI

mb = FlyBrainAPI(mode="mb")

print("--- Before any training ---")
a0 = mb.step(odor="A")
b0 = mb.step(odor="B")
print("A: MB_pref =", a0.get("MB_pref"), " app/avo =", a0.get("app"), a0.get("avo"))
print("B: MB_pref =", b0.get("MB_pref"), " app/avo =", b0.get("app"), b0.get("avo"))

print("\n--- Training B with reward=1.0 x20, A untouched ---")
for _ in range(20):
    mb.train(odor="B", reward=1.0)

a1 = mb.step(odor="A")
b1 = mb.step(odor="B")
print("A: MB_pref =", a1.get("MB_pref"))
print("B: MB_pref =", b1.get("MB_pref"))
print("Did B overtake A?", b1.get("MB_pref") > a1.get("MB_pref"))
