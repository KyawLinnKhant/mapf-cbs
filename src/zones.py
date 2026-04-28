"""Named warehouse zones per difficulty level."""

from __future__ import annotations
from typing import Dict, List, Tuple

from .grid import Grid, Position

# Zone definitions: name → list of candidate (x,y) cells (in priority order)
# Cells are spread across the grid to avoid clustering all agents in one spot.
ZONE_CANDIDATES: Dict[str, Dict[str, List[Position]]] = {
    "easy": {
        "top_left":     [(1,5),(1,4),(2,5)],
        "top_right":    [(5,5),(5,4),(4,5)],
        "bottom_left":  [(1,1),(1,2),(2,1)],
        "bottom_right": [(5,1),(5,2),(4,1)],
        "center":       [(3,3),(3,2),(2,3)],
    },
    "medium": {
        "loading_bay":  [(9,9),(8,9),(9,8),(8,8)],
        "storage":      [(2,2),(2,3),(3,2),(3,3)],
        "charging":     [(9,1),(8,1),(9,2),(8,2)],
        "dispatch":     [(1,9),(2,9),(1,8),(2,8)],
        "center":       [(5,5),(5,6),(6,5),(4,5)],
    },
    "hard": {
        "loading_bay":  [(13,13),(12,13),(13,12),(12,12),(11,13)],
        "storage_a":    [(2,2),(2,3),(3,2),(3,3),(1,2)],
        "storage_b":    [(13,2),(12,2),(13,3),(12,3),(11,2)],
        "charging":     [(2,13),(2,12),(3,13),(3,12),(1,13)],
        "inspection":   [(7,7),(7,8),(8,7),(6,7),(7,6)],
        "dispatch":     [(7,2),(7,3),(8,2),(6,2),(7,1)],
        "exit":         [(13,7),(12,7),(13,8),(13,6),(11,7)],
    },
    "expert": {
        "loading_bay":   [(18,18),(17,18),(18,17),(17,17),(16,18),(18,16),(16,17)],
        "storage_a":     [(2,2),(2,3),(3,2),(3,3),(1,2),(2,1),(4,2)],
        "storage_b":     [(18,2),(17,2),(18,3),(17,3),(16,2),(18,1),(16,3)],
        "storage_c":     [(2,18),(2,17),(3,18),(3,17),(1,18),(2,16),(4,18)],
        "charging":      [(10,18),(10,17),(11,18),(9,18),(10,16),(11,17),(9,17)],
        "inspection":    [(10,10),(10,11),(11,10),(9,10),(10,9),(11,11),(9,9)],
        "dispatch":      [(18,10),(17,10),(18,11),(18,9),(17,11),(16,10),(17,9)],
        "exit":          [(10,2),(10,3),(11,2),(9,2),(10,1),(11,3),(9,3)],
        "staging":       [(2,10),(2,11),(3,10),(1,10),(2,9),(3,11),(1,11)],
    },
}

ZONE_ALIASES: Dict[str, str] = {
    "load":      "loading_bay",
    "bay":       "loading_bay",
    "charge":    "charging",
    "charger":   "charging",
    "store":     "storage_a",
    "warehouse": "storage_a",
    "middle":    "center",
    "centre":    "center",
    "out":       "exit",
    "door":      "exit",
    "stage":     "staging",
    "hold":      "staging",
    "check":     "inspection",
    "quality":   "inspection",
}


def available_zones(level: str) -> List[str]:
    return list(ZONE_CANDIDATES.get(level, ZONE_CANDIDATES["expert"]).keys())


def resolve_goals(
    zone_assignments: Dict[int, str],
    level: str,
    grid: Grid,
) -> Dict[int, Position]:
    """
    Convert {agent_id: zone_name} → {agent_id: (x, y)} picking free cells.
    Agents assigned to the same zone get distinct nearby cells.
    """
    candidates = ZONE_CANDIDATES.get(level, ZONE_CANDIDATES["expert"])

    # Normalise zone names through aliases
    def normalise(name: str) -> str:
        name = name.lower().strip()
        return ZONE_ALIASES.get(name, name)

    # Group agents by zone
    zone_agents: Dict[str, List[int]] = {}
    for agent_id, zone in zone_assignments.items():
        z = normalise(zone)
        if z not in candidates:
            # fallback: pick closest valid zone name
            z = available_zones(level)[0]
        zone_agents.setdefault(z, []).append(agent_id)

    goals: Dict[int, Position] = {}
    used: set = set()

    for zone, agents in zone_agents.items():
        pool = [p for p in candidates[zone] if grid.is_free(*p) and p not in used]
        # If pool exhausted, expand with nearby free cells
        if len(pool) < len(agents):
            cx = sum(p[0] for p in candidates[zone]) // len(candidates[zone])
            cy = sum(p[1] for p in candidates[zone]) // len(candidates[zone])
            for r in range(1, max(grid.width, grid.height)):
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        p = (cx+dx, cy+dy)
                        if grid.is_free(*p) and p not in used and p not in pool:
                            pool.append(p)
                if len(pool) >= len(agents):
                    break

        for i, agent_id in enumerate(agents):
            pos = pool[i % len(pool)]
            goals[agent_id] = pos
            used.add(pos)

    return goals
