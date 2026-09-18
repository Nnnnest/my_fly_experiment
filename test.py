# quick_check.py, in my_experiments/
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.connectome_agent import ConnectomeBanditAgent

agent = ConnectomeBanditAgent(2, ["A", "B"], epsilon=0.0, seed=0)  # epsilon=0: pure exploitation, no randomness
for i in range(20):
    arm = agent.select_action()
    reward = 1.0 if arm == 1 else 0.0   # pretend arm B (index 1) is always correct
    agent.update(arm, reward)
    print(f"trial {i}: chose arm {arm}, reward {reward}")
