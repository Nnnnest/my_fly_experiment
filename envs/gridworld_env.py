import numpy as np
from collections import deque

GRID = [
    "##########",
    "#S.......#",
    "#.######.#",
    "#........#",
    "#......G.#",
    "##########",
]

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}

class GridWorld:
    def __init__(self, grid=GRID, step_penalty=-1.0, goal_reward=10.0, max_steps=100, seed=0):
        self.grid = [list(row) for row in grid]
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.start = self._find("S")
        self.goal = self._find("G")
        self.n_states = self.height * self.width

        self.pos = None
        self.steps = 0

    def _find(self, ch):
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell == ch:
                    return (r, c)
        raise ValueError(f"'{ch}' not found in grid")

    def _is_wall(self, r, c):
        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return True
        return self.grid[r][c] == "#"

    def state_id(self, pos):
        r, c = pos
        return r * self.width + c

    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self.state_id(self.pos)

    def step(self, action):
        dr, dc = ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if not self._is_wall(nr, nc):
            self.pos = (nr, nc)
        self.steps += 1

        done = self.pos == self.goal
        reward = self.goal_reward if done else self.step_penalty
        if self.steps >= self.max_steps:
            done = True
        return self.state_id(self.pos), reward, done

    def set_pos(self, pos):
        self.pos = pos
        self.steps = 0

    def traversable_states(self):
        states = []
        for r in range(self.height):
            for c in range(self.width):
                if not self._is_wall(r, c):
                    states.append((r, c))
        return states

    def bfs_distances_from_goal(self):
        """Walls-aware shortest-path distance from every traversable cell to
        the goal, via BFS on the grid graph. Replaces Manhattan distance,
        which cuts through walls and misleads shaping near corners."""
        dist = np.full(self.n_states, np.inf)
        goal_id = self.state_id(self.goal)
        dist[goal_id] = 0
        q = deque([self.goal])
        while q:
            r, c = q.popleft()
            d = dist[self.state_id((r, c))]
            for dr, dc in ACTIONS.values():
                nr, nc = r + dr, c + dc
                if not self._is_wall(nr, nc):
                    nid = self.state_id((nr, nc))
                    if dist[nid] > d + 1:
                        dist[nid] = d + 1
                        q.append((nr, nc))
        return dist
