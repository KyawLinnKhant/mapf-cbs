"""
Dynamic obstacle agents for MAPF environments.

These are non-learning agents that move autonomously each timestep.
They appear in the MARL agent observation (ch1 of the local crop) just
like other agents, enabling zero-shot generalisation from the trained policy.

CBS cannot handle them — it plans once on the static map and the plan is
invalidated at execution time as obstacles move into planned cells.

Patterns
--------
random_walk  : at each step, move to a uniformly random free neighbour (or stay).
               Models unpredictable autonomous machines / humans.
patrol       : advance along a fixed axis until hitting a wall, then reverse.
               Models forklifts / conveyor-belt carts on fixed routes.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from .grid import Grid, Position, MOVES


class DynamicObstacle:
    """A single non-learning moving obstacle."""

    def __init__(
        self,
        pos: Position,
        grid: Grid,
        pattern: str = "random_walk",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.pos = pos
        self.grid = grid
        self.pattern = pattern
        self._rng = rng if rng is not None else np.random.default_rng()
        self._patrol_dir: Tuple[int, int] = (0, 0)
        self._init_patrol()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_patrol(self) -> None:
        if self.pattern != "patrol":
            return
        # Choose initial direction along whichever axis has more free space
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        valid = [d for d in dirs if self.grid.is_free(self.pos[0]+d[0], self.pos[1]+d[1])]
        if valid:
            idx = int(self._rng.integers(len(valid)))
            self._patrol_dir = valid[idx]
        else:
            self._patrol_dir = (1, 0)

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, occupied: set | None = None) -> None:
        """
        Advance one timestep. `occupied` is the set of positions currently
        held by MARL agents — dynamic obstacles yield to MARL agents rather
        than pushing through them.
        """
        blocked = set(occupied or [])

        if self.pattern == "random_walk":
            self._step_random(blocked)
        elif self.pattern == "patrol":
            self._step_patrol(blocked)

    def _step_random(self, blocked: set) -> None:
        x, y = self.pos
        candidates: List[Position] = []
        for dx, dy in MOVES[1:]:          # skip wait — always try to move
            nx, ny = x + dx, y + dy
            if self.grid.is_free(nx, ny) and (nx, ny) not in blocked:
                candidates.append((nx, ny))
        if candidates:
            idx = int(self._rng.integers(len(candidates)))
            self.pos = candidates[idx]
        # else stay put

    def _step_patrol(self, blocked: set) -> None:
        dx, dy = self._patrol_dir
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        if self.grid.is_free(nx, ny) and (nx, ny) not in blocked:
            self.pos = (nx, ny)
        else:
            # Reverse and try once
            self._patrol_dir = (-dx, -dy)
            nx2, ny2 = self.pos[0] - dx, self.pos[1] - dy
            if self.grid.is_free(nx2, ny2) and (nx2, ny2) not in blocked:
                self.pos = (nx2, ny2)
            # else stay put


# ── Factory ───────────────────────────────────────────────────────────────────

def spawn_dynamic_obstacles(
    n: int,
    grid: Grid,
    rng: np.random.Generator,
    exclude: List[Position] | None = None,
    pattern: str = "mixed",
) -> List[DynamicObstacle]:
    """
    Randomly place *n* dynamic obstacles on free cells not in *exclude*.

    pattern: "random_walk" | "patrol" | "mixed" (half/half)
    """
    exclude_set = set(exclude or [])
    free = [
        (x, y)
        for y in range(grid.height)
        for x in range(grid.width)
        if grid.is_free(x, y) and (x, y) not in exclude_set
    ]
    if len(free) < n:
        n = len(free)

    idx = rng.choice(len(free), size=n, replace=False)
    positions = [free[i] for i in idx]

    obstacles: List[DynamicObstacle] = []
    for k, pos in enumerate(positions):
        if pattern == "mixed":
            pat = "patrol" if k % 2 == 0 else "random_walk"
        else:
            pat = pattern
        obstacles.append(DynamicObstacle(pos, grid, pattern=pat, rng=rng))

    return obstacles
