import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from envs.gridworld_env import GridWorld
env = GridWorld()
traversable = sum(1 for row in env.grid for cell in row if cell != "#")
print("traversable cells:", traversable, "/ n_states:", env.n_states)
