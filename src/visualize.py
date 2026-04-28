"""Visualisation for MAPF solutions — animated and static."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

from .grid import Grid, Position

# Colorblind-friendly palette (10 distinct colours)
AGENT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def _pad_path(path: List[Position], length: int) -> List[Position]:
    if not path:
        return path
    return path + [path[-1]] * (length - len(path))


def animate(
    grid: Grid,
    paths: Dict[int, List[Position]],
    goals: List[Position],
    interval: int = 200,
    trail_len: int = 4,
    save_path: Optional[str] = None,
    dynamic_obstacle_paths: Optional[Dict[int, List[Position]]] = None,
) -> FuncAnimation:
    """
    Produce a matplotlib FuncAnimation of agents moving on the grid.

    Parameters
    ----------
    grid      : the environment
    paths     : {agent_id: [(x,y), ...]}
    goals     : goal position for each agent (index = agent_id)
    interval  : milliseconds between frames
    trail_len : number of past positions drawn as fading trail
    save_path : if provided, save a GIF to this path (requires Pillow)
    """
    n = len(paths)
    dyn_paths = dynamic_obstacle_paths or {}
    max_len = max(len(p) for p in paths.values())
    padded = {a: _pad_path(paths[a], max_len) for a in paths}
    dyn_padded = {k: _pad_path(dyn_paths[k], max_len) for k in dyn_paths}

    fig, ax = plt.subplots(figsize=(max(6, grid.width * 0.6), max(6, grid.height * 0.6)))
    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(-0.5, grid.height - 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Grid lines
    for x in range(grid.width + 1):
        ax.axvline(x - 0.5, color="#333", linewidth=0.4)
    for y in range(grid.height + 1):
        ax.axhline(y - 0.5, color="#333", linewidth=0.4)

    # Obstacles
    for ox, oy in grid.obstacles:
        rect = mpatches.FancyBboxPatch(
            (ox - 0.45, oy - 0.45), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor="#555", edgecolor="#777", linewidth=0.8,
        )
        ax.add_patch(rect)

    # Goal markers (stars)
    for i, (gx, gy) in enumerate(goals):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax.plot(gx, gy, marker="*", markersize=14, color=color,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)

    # Dynamic obstacle markers (grey diamonds)
    dyn_markers = []
    for k in dyn_padded:
        x, y = dyn_padded[k][0]
        marker = plt.Polygon(
            [(x, y+0.38), (x+0.38, y), (x, y-0.38), (x-0.38, y)],
            closed=True, facecolor="#888888", edgecolor="#cccccc",
            linewidth=0.8, zorder=4, alpha=0.85,
        )
        ax.add_patch(marker)
        dyn_markers.append(marker)

    # Agent circles (initialised at t=0)
    agent_circles = []
    for i in range(n):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        x, y = padded[i][0]
        circle = plt.Circle((x, y), 0.35, color=color, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=6)
        agent_circles.append(circle)

    # Agent labels (move with circle)
    agent_labels = []
    for i in range(n):
        x, y = padded[i][0]
        lbl = ax.text(x, y, str(i), ha="center", va="center",
                      fontsize=8, fontweight="bold", color="white", zorder=6)
        agent_labels.append(lbl)

    # Trails (line objects, one per agent)
    trail_lines = []
    for i in range(n):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        (line,) = ax.plot([], [], color=color, linewidth=1.5,
                          alpha=0.5, zorder=4)
        trail_lines.append(line)

    # Timestep counter
    time_text = ax.text(
        0.02, 0.97, "t = 0", transform=ax.transAxes,
        color="white", fontsize=10, va="top",
    )

    def _update(frame: int):
        # Move dynamic obstacle diamonds
        for k, marker in enumerate(dyn_markers):
            x, y = dyn_padded[k][frame]
            marker.set_xy([(x, y+0.38), (x+0.38, y), (x, y-0.38), (x-0.38, y)])

        for i in range(n):
            x, y = padded[i][frame]
            agent_circles[i].set_center((x, y))
            agent_labels[i].set_position((x, y))

            # trail
            start = max(0, frame - trail_len)
            xs = [padded[i][t][0] for t in range(start, frame + 1)]
            ys = [padded[i][t][1] for t in range(start, frame + 1)]
            trail_lines[i].set_data(xs, ys)

        time_text.set_text(f"t = {frame}")
        return agent_circles + agent_labels + trail_lines + dyn_markers + [time_text]

    anim = FuncAnimation(
        fig, _update, frames=max_len,
        interval=interval, blit=True, repeat=True,
    )

    if save_path:
        writer = PillowWriter(fps=1000 // interval)
        anim.save(save_path, writer=writer)
        print(f"Saved animation to {save_path}")

    return anim


def plot_paths(
    grid: Grid,
    paths: Dict[int, List[Position]],
    goals: List[Position],
    title: str = "MAPF — CBS Solution",
    save_path: Optional[str] = None,
    dynamic_obstacle_paths: Optional[Dict[int, List[Position]]] = None,
) -> None:
    """Static plot of all agent paths."""
    fig, ax = plt.subplots(figsize=(max(6, grid.width * 0.7), max(6, grid.height * 0.7)))
    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(-0.5, grid.height - 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_title(title, color="white", fontsize=13)

    for x in range(grid.width + 1):
        ax.axvline(x - 0.5, color="#333", linewidth=0.4)
    for y in range(grid.height + 1):
        ax.axhline(y - 0.5, color="#333", linewidth=0.4)

    for ox, oy in grid.obstacles:
        rect = mpatches.FancyBboxPatch(
            (ox - 0.45, oy - 0.45), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor="#555", edgecolor="#777",
        )
        ax.add_patch(rect)

    for i, path in paths.items():
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color=color, linewidth=2, alpha=0.7, zorder=3)
        # start
        ax.plot(xs[0], ys[0], "o", color=color, markersize=10,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.text(xs[0], ys[0], str(i), ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=6)
        # goal
        ax.plot(goals[i][0], goals[i][1], "*", color=color, markersize=14,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)

    # Draw dynamic obstacle start+end positions as grey diamonds
    if dynamic_obstacle_paths:
        for k, dpath in dynamic_obstacle_paths.items():
            if not dpath:
                continue
            # Draw full trajectory as faint grey trail
            xs = [p[0] for p in dpath]
            ys = [p[1] for p in dpath]
            ax.plot(xs, ys, color="#888888", linewidth=1, alpha=0.3, zorder=2)
            # Final position as diamond
            ax.plot(xs[-1], ys[-1], marker="D", markersize=8,
                    color="#888888", markeredgecolor="#cccccc",
                    markeredgewidth=0.6, alpha=0.85, zorder=4)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved static plot to {save_path}")

    plt.tight_layout()
