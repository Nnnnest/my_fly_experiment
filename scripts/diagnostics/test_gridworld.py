import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.gridworld_env import GridWorld, ACTION_NAMES
import numpy as np

env = GridWorld()
print(f"start={env.start} goal={env.goal} n_states={env.n_states}")

s = env.reset()
print("Walking into a wall on purpose (up from start, should not move):")
s2, r, done = env.step(0)  # up — S is against the top wall, should bounce
print(f"  state {s} -> {s2}, reward={r}, done={done}, still at {env.pos}")
assert s2 == s, "agent moved through a wall"

print("\nRandom agent, 5 episodes:")
rng = np.random.default_rng(1)
for ep in range(5):
    s = env.reset()
    total_r = 0
    for t in range(env.max_steps):
        a = rng.integers(4)
        s, r, done = env.step(a)
        total_r += r
        if done:
            break
    print(f"  episode {ep}: {t+1} steps, total_reward={total_r}, reached_goal={env.pos == env.goal}")
