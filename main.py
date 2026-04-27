"""
MAPF-CBS Demo
=============
8 agents navigating a 12x12 grid with obstacles.
Outputs:
  results/demo.gif        — animated GIF
  results/paths.png       — static path overview
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.grid import Grid
from src.cbs import cbs
from src.visualize import animate, plot_paths

# ── Environment ────────────────────────────────────────────────────────────
GRID_W, GRID_H = 12, 12

# fmt: off
OBSTACLE_MAP = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,1,0,0,0,0,0],
    [0,0,0,0,0,1,1,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,0,1,1,0],
    [0,1,1,0,0,0,0,0,0,1,1,0],
    [0,0,0,0,0,1,1,0,0,0,0,0],
    [0,0,0,0,0,1,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.int8)
# fmt: on

grid = Grid.from_array(OBSTACLE_MAP)

# ── Agents ──────────────────────────────────────────────────────────────────
STARTS = [
    (0, 0),  (11, 0),  (0, 11), (11, 11),   # corners
    (0, 5),  (11, 5),  (5, 0),  (5, 11),    # edges (mid)
]
GOALS = [
    (11, 11), (0, 11), (11, 0),  (0, 0),    # swap corners
    (11, 6),  (0, 6),  (6, 11),  (6, 0),    # swap edges
]

# ── Solve ───────────────────────────────────────────────────────────────────
print("Running CBS…")
solution = cbs(grid, STARTS, GOALS, max_t=300)

if solution is None:
    print("No solution found.")
    raise SystemExit(1)

soc = sum(len(p) - 1 for p in solution.values())
makespan = max(len(p) - 1 for p in solution.values())
print(f"Solution found!  agents={len(solution)}  SoC={soc}  makespan={makespan}")

# ── Output ──────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

# Static overview
plot_paths(grid, solution, GOALS,
           title=f"CBS — 8 agents | SoC={soc} | makespan={makespan}",
           save_path="results/paths.png")

# Animated GIF
anim = animate(grid, solution, GOALS,
               interval=250, trail_len=5,
               save_path="results/demo.gif")

plt.show()
print("Done. Check results/demo.gif and results/paths.png")
