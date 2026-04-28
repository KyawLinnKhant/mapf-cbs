"""Space-time A* — low-level solver for CBS."""

import heapq
from typing import Dict, List, Optional, Set, Tuple

from .grid import Grid, Position

# Vertex constraint: agent cannot be at (x, y) at time t
VertexConstraint = Tuple[int, int, int]  # (x, y, t)
# Edge constraint: agent cannot move from (x1,y1) to (x2,y2) at time t
EdgeConstraint = Tuple[int, int, int, int, int]  # (x1, y1, x2, y2, t)


def heuristic(pos: Position, goal: Position) -> int:
    """Manhattan distance."""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def astar(
    grid: Grid,
    start: Position,
    goal: Position,
    vertex_constraints: Set[VertexConstraint] = None,
    edge_constraints: Set[EdgeConstraint] = None,
    max_t: int = 256,
    max_nodes: int = 0,
) -> Optional[List[Position]]:
    """
    Space-time A* with vertex and edge constraints.

    Returns the shortest path from start to goal that respects all constraints,
    or None if no path exists within max_t steps or max_nodes expansions.

    max_nodes: hard cap on node expansions (0 = unlimited).
    """
    vc = vertex_constraints or set()
    ec = edge_constraints or set()

    start_state = (start[0], start[1], 0)
    start_h = heuristic(start, goal)

    # (f, g, x, y, t) — no path stored in heap
    open_set: List[Tuple] = [(start_h, 0, start[0], start[1], 0)]
    visited: Set[Tuple[int, int, int]] = set()

    # Parent pointers: state → parent state (None for root)
    came_from: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {
        start_state: None
    }
    # Best g seen so far for each state (avoids re-pushing with worse g)
    g_score: Dict[Tuple[int, int, int], int] = {start_state: 0}

    expansions = 0

    while open_set:
        f, g, x, y, t = heapq.heappop(open_set)

        state = (x, y, t)
        if state in visited:
            continue
        visited.add(state)
        expansions += 1

        if max_nodes and expansions > max_nodes:
            return None

        if (x, y) == goal:
            # Reconstruct path via parent pointers — O(path_len), not O(n²)
            path = []
            cur: Optional[Tuple[int, int, int]] = state
            while cur is not None:
                path.append((cur[0], cur[1]))
                cur = came_from.get(cur)
            return path[::-1]

        if t >= max_t:
            continue

        for nx, ny in grid.neighbors(x, y):
            nt = t + 1
            nstate = (nx, ny, nt)

            if nstate in visited:
                continue
            if (nx, ny, nt) in vc:
                continue
            if (x, y, nx, ny, nt) in ec:
                continue
            if (nx, ny, x, y, nt) in ec:
                continue

            ng = g + 1
            if ng < g_score.get(nstate, 10**9):
                g_score[nstate] = ng
                came_from[nstate] = state
                nf = ng + heuristic((nx, ny), goal)
                heapq.heappush(open_set, (nf, ng, nx, ny, nt))

    return None  # no path found
