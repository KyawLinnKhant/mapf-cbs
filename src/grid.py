"""Grid environment for MAPF."""

import numpy as np
from typing import List, Tuple, Set

Position = Tuple[int, int]

MOVES = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]  # wait + 4 directions


class Grid:
    def __init__(self, width: int, height: int, obstacles: List[Position] = None):
        self.width = width
        self.height = height
        self.obstacles: Set[Position] = set(obstacles or [])
        self._array = np.zeros((height, width), dtype=np.int8)
        for ox, oy in self.obstacles:
            self._array[oy][ox] = 1

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Grid":
        h, w = arr.shape
        obstacles = [(x, y) for y in range(h) for x in range(w) if arr[y][x] == 1]
        return cls(w, h, obstacles)

    def is_free(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles

    def neighbors(self, x: int, y: int) -> List[Position]:
        return [(x + dx, y + dy) for dx, dy in MOVES if self.is_free(x + dx, y + dy)]

    def to_array(self) -> np.ndarray:
        return self._array.copy()

    def __repr__(self) -> str:
        return f"Grid({self.width}x{self.height}, {len(self.obstacles)} obstacles)"
