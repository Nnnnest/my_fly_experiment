import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents"))
from envs.gridworld_env import GridWorld
from envs.maze_generator import generate_maze
from agents.mlp_gridworld_agent import MLPGridAgent
import torch

MEDIUM_MAZE = generate_maze(15, 11, extra_connections=0.1, seed=0)
env = GridWorld(grid=MEDIUM_MAZE, max_steps=250, seed=0)
n_states = env.n_states
bfs_dist = env.bfs_distances_from_goal()

def shape_reward(raw_r, s, s2, gamma=0.95):
    return raw_r + (bfs_dist[s] - gamma * bfs_dist[s2])

TARGET_SYNC_EVERY = 10   # change this to try other values
LR = 0.01                # change this too, once sync interval is settled
N_EPISODES = 300

agent = MLPGridAgent(n_states, 4, lr=LR, min_pulls=5, target_sync_every=TARGET_SYNC_EVERY, seed=0)

print(f"target_sync_every={TARGET_SYNC_EVERY}, lr={LR}\n")

successes = []
for ep in range(N_EPISODES):
    s = env.reset()
    for t in range(env.max_steps):
        a = agent.select_action(s)
        s2, r, done = env.step(a)
        shaped_r = shape_reward(r, s, s2)
        agent.update(s, a, shaped_r, s2, done)
        s = s2
        if done:
            break
    successes.append(env.pos == env.goal)

    with torch.no_grad():
        q_sample = agent.net(agent._one_hot_batch([env.state_id(env.start)]))[0]
    has_nan = torch.isnan(q_sample).any().item()

    if ep >= 9:
        recent_rate = sum(successes[-10:]) / 10
        print(f"ep {ep:2d}: steps={t+1:3d}, reached_goal={str(env.pos==env.goal):5s}, "
              f"recent_success_rate={recent_rate:.2f}, "
              f"Q(start)={[round(x,2) for x in q_sample.tolist()]}, has_nan={has_nan}")
    else:
        print(f"ep {ep:2d}: steps={t+1:3d}, reached_goal={str(env.pos==env.goal):5s}")

print(f"\nfinal 20-episode success rate: {sum(successes[-20:])/20:.2f}")
