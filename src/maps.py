"""Procedural map generation for MAPF training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .grid import Grid, Position


@dataclass
class MapConfig:
    width: int
    height: int
    n_agents: int
    obstacle_density: float
    map_type: str   # "maze" | "rooms" | "scatter"
    name: str


DIFFICULTY_LEVELS: dict[str, MapConfig] = {
    "easy":   MapConfig( 7,  7,  2, 0.10, "maze",    "easy"),
    "medium": MapConfig(11, 11,  4, 0.20, "rooms",   "medium"),
    "hard":   MapConfig(15, 15,  8, 0.28, "scatter", "hard"),
    "expert": MapConfig(20, 20, 12, 0.33, "rooms",   "expert"),
}


# ── Connectivity helper ──────────────────────────────────────────────────────

def bfs_reachable(grid: Grid, start: Position) -> set:
    """Return all positions reachable from start via grid.neighbors."""
    visited: set = {start}
    queue: deque = deque([start])
    while queue:
        pos = queue.popleft()
        for nb in grid.neighbors(*pos):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited


# ── Map generators ───────────────────────────────────────────────────────────

def _generate_maze(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """
    Iterative recursive-backtracker maze.
    Maze cells live at odd (x,y) indices; even-indexed cells are walls between them.
    """
    arr = np.ones((height, width), dtype=np.int8)

    def in_bounds(x: int, y: int) -> bool:
        return 0 < x < width and 0 < y < height

    arr[1][1] = 0
    stack = [(1, 1)]
    visited = {(1, 1)}

    while stack:
        cx, cy = stack[-1]
        candidates = []
        for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
            nx, ny = cx + dx, cy + dy
            if in_bounds(nx, ny) and (nx, ny) not in visited:
                candidates.append((nx, ny, dx, dy))

        if candidates:
            nx, ny, dx, dy = candidates[int(rng.integers(len(candidates)))]
            arr[cy + dy // 2][cx + dx // 2] = 0   # carve wall between cells
            arr[ny][nx] = 0                         # carve destination cell
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()

    # Randomly punch extra openings to reduce pure-corridor linearity
    extra = max(1, int(width * height * 0.04))
    for _ in range(extra):
        x = int(rng.integers(1, width - 1))
        y = int(rng.integers(1, height - 1))
        arr[y][x] = 0

    return arr


def _generate_rooms(
    width: int, height: int, rng: np.random.Generator, density: float
) -> np.ndarray:
    """Random rooms connected by L-shaped corridors."""
    arr = np.ones((height, width), dtype=np.int8)

    n_rooms = max(4, int(width * height * 0.012))
    rooms: List[Tuple[int, int, int, int]] = []  # (x, y, w, h)

    for _ in range(n_rooms * 8):
        rw = int(rng.integers(3, max(4, width // 3)))
        rh = int(rng.integers(3, max(4, height // 3)))
        rx = int(rng.integers(1, width - rw - 1))
        ry = int(rng.integers(1, height - rh - 1))

        # Reject rooms that overlap existing ones (with 1-cell margin)
        if any(
            rx < ex + ew + 1 and rx + rw + 1 > ex and
            ry < ey + eh + 1 and ry + rh + 1 > ey
            for ex, ey, ew, eh in rooms
        ):
            continue

        rooms.append((rx, ry, rw, rh))
        arr[ry: ry + rh, rx: rx + rw] = 0

        if len(rooms) >= n_rooms:
            break

    # Connect consecutive rooms with L-shaped corridors
    perm = rng.permutation(len(rooms))
    for i in range(len(perm) - 1):
        r1 = rooms[perm[i]]
        r2 = rooms[perm[i + 1]]
        x1 = r1[0] + r1[2] // 2
        y1 = r1[1] + r1[3] // 2
        x2 = r2[0] + r2[2] // 2
        y2 = r2[1] + r2[3] // 2
        for x in range(min(x1, x2), max(x1, x2) + 1):
            arr[y1][x] = 0
        for y in range(min(y1, y2), max(y1, y2) + 1):
            arr[y][x2] = 0

    return arr


def _generate_scatter(
    width: int, height: int, rng: np.random.Generator, density: float
) -> np.ndarray:
    """Random obstacle scatter; retries until ≥80% of free cells are connected."""
    for _ in range(30):
        arr = (rng.random((height, width)) < density).astype(np.int8)
        arr[0, :] = arr[-1, :] = arr[:, 0] = arr[:, -1] = 0   # clear border

        grid = Grid.from_array(arr)
        free = [(x, y) for y in range(height) for x in range(width) if arr[y][x] == 0]
        if len(free) < 2:
            continue

        reachable = bfs_reachable(grid, free[0])
        if len(reachable) >= len(free) * 0.80:
            return arr

    # Fallback: mostly empty map
    return np.zeros((height, width), dtype=np.int8)


# ── Public API ───────────────────────────────────────────────────────────────

def generate_map(
    config: MapConfig, seed: Optional[int] = None
) -> Tuple[Grid, np.ndarray]:
    """Generate a map according to *config*. Returns (Grid, raw_array)."""
    rng = np.random.default_rng(seed)

    if config.map_type == "maze":
        arr = _generate_maze(config.width, config.height, rng)
    elif config.map_type == "rooms":
        arr = _generate_rooms(config.width, config.height, rng, config.obstacle_density)
    else:
        arr = _generate_scatter(config.width, config.height, rng, config.obstacle_density)

    return Grid.from_array(arr), arr


def sample_positions(
    grid: Grid,
    n: int,
    rng: np.random.Generator,
    exclude: Optional[List[Position]] = None,
) -> Optional[List[Position]]:
    """
    Sample *n* distinct free positions, all within the same connected component.
    Returns None if fewer than n reachable free cells exist.
    """
    exclude_set = set(exclude or [])
    free = [
        (x, y)
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.is_free(x, y) and (x, y) not in exclude_set
    ]
    if len(free) < n:
        return None

    # Find the largest connected component among free cells
    reachable = bfs_reachable(grid, free[0])
    component = [p for p in free if p in reachable]
    if len(component) < n:
        return None

    idx = rng.choice(len(component), size=n, replace=False)
    return [component[i] for i in idx]
