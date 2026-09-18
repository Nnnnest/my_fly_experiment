import numpy as np

def generate_maze(width, height, extra_connections=0.1, seed=0):
    assert width % 2 == 1 and height % 2 == 1, \
        "width and height must be odd (standard requirement for this carving algorithm)"
    rng = np.random.default_rng(seed)
    grid = [["#"] * width for _ in range(height)]

    def carve(r, c):
        grid[r][c] = "."
        dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < height - 1 and 1 <= nc < width - 1 and grid[nr][nc] == "#":
                grid[r + dr // 2][c + dc // 2] = "."
                carve(nr, nc)

    carve(1, 1)
    grid[1][1] = "S"
    grid[height - 2][width - 2] = "G"

    # add loops: find wall cells that sit directly between two already-carved
    # cells (a removable connector) and open some of them — this only ADDS
    # paths on top of the guaranteed-connected spanning tree, never removes
    # the one path that's already guaranteed to exist.
    for r in range(2, height - 2):
        for c in range(2, width - 2):
            if grid[r][c] != "#":
                continue
            horiz_open = grid[r][c - 1] != "#" and grid[r][c + 1] != "#"
            vert_open = grid[r - 1][c] != "#" and grid[r + 1][c] != "#"
            if (horiz_open or vert_open) and rng.random() < extra_connections:
                grid[r][c] = "."

    grid[1][1] = "S"
    grid[height - 2][width - 2] = "G"
    return ["".join(row) for row in grid]


def count_traversable(rows):
    return sum(1 for row in rows for ch in row if ch != "#")
