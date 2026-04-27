"""Conflict-Based Search (CBS) — high-level solver for MAPF."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .astar import EdgeConstraint, VertexConstraint, astar
from .grid import Grid, Position


@dataclass
class Conflict:
    """Detected conflict between two agents."""

    type: str  # "vertex" or "edge"
    agent_a: int
    agent_b: int
    # vertex conflict: (x, y, t); edge conflict: (x1, y1, x2, y2, t)
    location: Tuple


@dataclass
class CTNode:
    """Node in the CBS Constraint Tree."""

    constraints_v: Dict[int, Set[VertexConstraint]] = field(default_factory=dict)
    constraints_e: Dict[int, Set[EdgeConstraint]] = field(default_factory=dict)
    paths: Dict[int, List[Position]] = field(default_factory=dict)
    cost: int = 0

    def __lt__(self, other: "CTNode") -> bool:
        return self.cost < other.cost


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pad_path(path: List[Position], length: int) -> List[Position]:
    """Extend a path by waiting at the goal until *length* timesteps."""
    if not path:
        return path
    return path + [path[-1]] * (length - len(path))


def _find_conflict(paths: Dict[int, List[Position]]) -> Optional[Conflict]:
    """Return the first vertex or edge conflict found, or None."""
    agents = list(paths.keys())
    # pad all paths to the same length
    max_len = max(len(p) for p in paths.values())
    padded = {a: _pad_path(paths[a], max_len) for a in agents}

    for i, a in enumerate(agents):
        for b in agents[i + 1:]:
            pa, pb = padded[a], padded[b]
            for t in range(max_len):
                # vertex conflict
                if pa[t] == pb[t]:
                    return Conflict("vertex", a, b, (pa[t][0], pa[t][1], t))
                # edge conflict (swap)
                if t + 1 < max_len and pa[t] == pb[t + 1] and pb[t] == pa[t + 1]:
                    x1, y1 = pa[t]
                    x2, y2 = pa[t + 1]
                    return Conflict("edge", a, b, (x1, y1, x2, y2, t + 1))
    return None


def _node_cost(paths: Dict[int, List[Position]]) -> int:
    """Sum-of-costs objective."""
    return sum(len(p) - 1 for p in paths.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cbs(
    grid: Grid,
    starts: List[Position],
    goals: List[Position],
    max_t: int = 256,
    max_ct_nodes: int = 0,
    max_astar_nodes: int = 0,
) -> Optional[Dict[int, List[Position]]]:
    """
    Conflict-Based Search for MAPF.

    Parameters
    ----------
    grid           : the environment
    starts         : start position for each agent (index = agent id)
    goals          : goal position  for each agent
    max_t          : maximum timesteps for the low-level A* solver
    max_ct_nodes   : hard cap on CT node expansions (0 = unlimited)
    max_astar_nodes: hard cap on A* node expansions per call (0 = unlimited)

    Returns
    -------
    dict {agent_id: path} or None if no solution within budget.
    """
    n = len(starts)
    assert len(goals) == n, "starts and goals must have the same length"

    # Root node: no constraints, solve each agent independently
    root = CTNode()
    for i in range(n):
        root.constraints_v[i] = set()
        root.constraints_e[i] = set()
        path = astar(grid, starts[i], goals[i], max_t=max_t,
                     max_nodes=max_astar_nodes)
        if path is None:
            return None  # even unconstrained A* fails
        root.paths[i] = path
    root.cost = _node_cost(root.paths)

    open_list: List[CTNode] = [root]
    heapq.heapify(open_list)
    ct_expansions = 0

    while open_list:
        node = heapq.heappop(open_list)
        ct_expansions += 1

        if max_ct_nodes and ct_expansions > max_ct_nodes:
            return None  # CT budget exhausted

        conflict = _find_conflict(node.paths)
        if conflict is None:
            return node.paths  # solution found

        # Branch: add a constraint for each of the two agents
        for affected_agent in (conflict.agent_a, conflict.agent_b):
            child = CTNode(
                constraints_v={a: set(s) for a, s in node.constraints_v.items()},
                constraints_e={a: set(s) for a, s in node.constraints_e.items()},
                paths=dict(node.paths),
            )

            if conflict.type == "vertex":
                x, y, t = conflict.location
                child.constraints_v[affected_agent].add((x, y, t))
            else:  # edge
                x1, y1, x2, y2, t = conflict.location
                if affected_agent == conflict.agent_a:
                    child.constraints_e[affected_agent].add((x1, y1, x2, y2, t))
                else:
                    # reverse direction for agent_b
                    child.constraints_e[affected_agent].add((x2, y2, x1, y1, t))

            # Re-plan for the constrained agent
            new_path = astar(
                grid,
                starts[affected_agent],
                goals[affected_agent],
                vertex_constraints=child.constraints_v[affected_agent],
                edge_constraints=child.constraints_e[affected_agent],
                max_t=max_t,
                max_nodes=max_astar_nodes,
            )
            if new_path is None:
                continue  # this branch is infeasible

            child.paths[affected_agent] = new_path
            child.cost = _node_cost(child.paths)
            heapq.heappush(open_list, child)

    return None  # open list exhausted — no solution
