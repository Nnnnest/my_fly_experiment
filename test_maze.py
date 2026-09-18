import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.maze_generator import generate_maze, count_traversable
from envs.gridworld_env import GridWorld

for size_name, (w, h) in [("small", (11, 7)), ("medium", (15, 11)), ("large", (19, 13))]:
    rows = generate_maze(w, h, extra_connections=0.1, seed=0)
    env = GridWorld(grid=rows)
    n_trav = count_traversable(rows)
    bfs = env.bfs_distances_from_goal()
    shortest = bfs[env.state_id(env.start)]
    solvable = shortest != float("inf")
    print(f"{size_name}: {w}x{h}, traversable={n_trav}, pairs_needed={n_trav*4}, "
          f"shortest_path={shortest if solvable else 'N/A'}, solvable={solvable}")
