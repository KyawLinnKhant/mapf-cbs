"""
Language-Conditioned MAPF Demo Showcase
========================================
Runs several preset commands non-interactively, saves GIFs + PNGs.
Uses rule-based parser (no Ollama required).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure project root on path
sys.path.insert(0, os.path.dirname(__file__))

from src.env import MAPFEnv, OBS_DIM, N_ACTIONS, CROP_SIZE
from src.maps import DIFFICULTY_LEVELS, generate_map, sample_positions
from src.mappo import MAPPO
from src.grid import Grid
from src.zones import available_zones, resolve_goals
from src.lang import parse_command, _rule_based_parse
from src.visualize import animate, plot_paths

OUT_DIR    = "results/demo_showcase"
CHECKPOINT = "checkpoints/mappo_best.pt"
LEVEL      = "expert"
DEVICE     = "mps"
SEED       = 42

LLM_MODEL = "qwen:4b"

COMMANDS = [
    "Send agents 0, 1, 2 to the loading bay. Put the rest at charging.",
    "All agents go to inspection.",
    "Agents 0-5 to loading bay, agents 6-11 to storage_a.",
    "Split the team: half to storage_a, half to dispatch.",
]

COLORS = {
    "loading_bay": "#3B82F6",
    "charging":    "#F59E0B",
    "inspection":  "#8B5CF6",
    "storage_a":   "#10B981",
    "storage_b":   "#059669",
    "storage_c":   "#047857",
    "dispatch":    "#EF4444",
    "exit":        "#F97316",
    "staging":     "#6B7280",
}


def run_episode(agent, config, goals, grid, max_steps=200):
    env = MAPFEnv(config, max_steps=max_steps)

    # Use the caller's grid directly so navigation map matches the visualisation.
    rng = np.random.default_rng(SEED)
    env._rng = rng
    env.grid = grid
    env.grid_array = grid.to_array()
    env._padded_grid = np.pad(env.grid_array, CROP_SIZE // 2,
                              mode='constant', constant_values=1)
    starts = sample_positions(grid, config.n_agents, rng)
    env.positions = {i: starts[i] for i in range(config.n_agents)}
    env.goals = goals
    env.t = 0
    env.goals_reached = {i: 0 for i in range(config.n_agents)}
    env.collisions = 0
    env.dynamic_obstacles = []
    env.cbs_paths = None
    env.cbs_available = False

    paths = {i: [env.positions[i]] for i in range(config.n_agents)}
    done = False
    while not done:
        obs = env._all_obs()
        with torch.no_grad():
            actions, _, _ = agent.act(obs)
        prev_goals = dict(env.goals)
        _, _, dones, info = env.step(actions)
        env.goals = prev_goals
        done = dones[0]
        for i in range(config.n_agents):
            paths[i].append(env.positions[i])
    return paths, info


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    config = DIFFICULTY_LEVELS[LEVEL]

    print(f"\n{'='*60}")
    print(f"  Language-Conditioned MAPF Demo Showcase")
    print(f"  Level: {LEVEL}  ({config.width}×{config.height}, {config.n_agents} agents)")
    print(f"  Commands: {len(COMMANDS)}")
    print(f"{'='*60}\n")

    # Load policy
    agent = MAPPO(OBS_DIM, config.n_agents, N_ACTIONS,
                  hidden_dim=128, n_heads=4, n_comm_layers=2, device=DEVICE)
    if os.path.exists(CHECKPOINT):
        agent.load(CHECKPOINT)
        print(f"  Loaded: {CHECKPOINT}")
    else:
        print(f"  WARNING: No checkpoint found — using random policy.")
    agent.set_eval()

    grid, _ = generate_map(config, seed=SEED)
    zones   = available_zones(LEVEL)

    for idx, cmd in enumerate(COMMANDS, 1):
        print(f"\n[{idx}/{len(COMMANDS)}] Command: \"{cmd}\"")
        t0 = time.time()

        # Parse with real LLM (qwen:4b via Ollama)
        assignments = parse_command(cmd, config.n_agents, LEVEL, model=LLM_MODEL)
        goals = resolve_goals(assignments, LEVEL, grid)

        print(f"  Zone assignments:")
        for agent_id, zone in sorted(assignments.items()):
            print(f"    agent {agent_id:2d} → {zone}")

        paths, info = run_episode(agent, config, goals, grid)
        elapsed = time.time() - t0

        makespan = max(len(p) - 1 for p in paths.values())
        print(f"  Done — goals={info['goals_reached']}  "
              f"collisions={info['collisions']}  makespan={makespan}  "
              f"({elapsed:.1f}s)")

        # GIF
        goal_list = [goals[i] for i in range(config.n_agents)]
        gif_path = os.path.join(OUT_DIR, f"demo_{idx:02d}.gif")
        animate(grid, paths, goal_list, interval=180, trail_len=6, save_path=gif_path)

        # PNG with zone color legend
        png_path = os.path.join(OUT_DIR, f"demo_{idx:02d}.png")
        title = f'"{cmd[:55]}"  |  {config.n_agents} agents  |  makespan={makespan}'
        plot_paths(grid, paths, goal_list, title=title, save_path=png_path)
        plt.close("all")

        print(f"  → {gif_path}")
        print(f"  → {png_path}")

    # Summary figure (zone layout with assignment overlay)
    print(f"\n  Generating summary figure...")
    _summary_figure(config, grid, zones, COMMANDS, goal_list)

    print(f"\n{'='*60}")
    print(f"  Done. All outputs in: {OUT_DIR}/")
    print(f"{'='*60}\n")


def _summary_figure(config, grid, zones, commands, example_goals):
    """4-panel figure showing zone layout + 3 example command assignments."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    def draw(ax, assignment_title, assignments=None):
        W = config.width
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, W - 0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Grid
        for i in range(W + 1):
            ax.axhline(i - 0.5, color="#E5E7EB", lw=0.3, zorder=0)
            ax.axvline(i - 0.5, color="#E5E7EB", lw=0.3, zorder=0)

        # Walls (show as dark cells)
        for y in range(W):
            for x in range(W):
                if not grid.is_free(x, y):
                    rect = plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                                         color="#374151", zorder=1)
                    ax.add_patch(rect)

        ax.set_title(assignment_title, fontsize=7.5, fontweight="bold",
                     pad=4, wrap=True)

        if assignments is None:
            return

        from src.zones import ZONE_CANDIDATES
        cands = ZONE_CANDIDATES.get(LEVEL, {})
        for zone, cells in cands.items():
            fc = COLORS.get(zone, "#9CA3AF")
            for (cx, cy) in cells:
                rect = plt.Rectangle((cx - 0.43, cy - 0.43), 0.86, 0.86,
                                     color=fc, alpha=0.3, zorder=2)
                ax.add_patch(rect)

        for agent_id, zone in assignments.items():
            from src.zones import ZONE_CANDIDATES, resolve_goals
            goals = resolve_goals(assignments, LEVEL, grid)
            gx, gy = goals[agent_id]
            fc = COLORS.get(zone, "#9CA3AF")
            circ = plt.Circle((gx, gy), 0.35, color=fc, zorder=5, ec="white", lw=0.8)
            ax.add_patch(circ)
            ax.text(gx, gy, str(agent_id), ha="center", va="center",
                    fontsize=4.5, color="white", fontweight="bold", zorder=6)

    # Panel 1: zone layout legend
    ax = axes[0][0]
    draw(ax, f"Expert Warehouse\n{config.width}×{config.height}, {config.n_agents} agents")
    from src.zones import ZONE_CANDIDATES
    for zone, cells in ZONE_CANDIDATES.get(LEVEL, {}).items():
        fc = COLORS.get(zone, "#9CA3AF")
        cx = sum(c[0] for c in cells) / len(cells)
        cy = sum(c[1] for c in cells) / len(cells)
        for (x, y) in cells:
            rect = plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                                  color=fc, alpha=0.6, zorder=2)
            ax.add_patch(rect)
        ax.text(cx, cy, zone.replace("_", "\n"), ha="center", va="center",
                fontsize=4, fontweight="bold", color="white", zorder=3)

    # Panels 2-4: command assignments
    for panel_idx, (cmd, ax_) in enumerate(zip(COMMANDS[:3],
                                               [axes[0][1], axes[1][0], axes[1][1]])):
        assignments = parse_command(cmd, config.n_agents, LEVEL, model=LLM_MODEL)
        label = f'Command {panel_idx+1}: "{cmd[:45]}..."' if len(cmd) > 45 else f'Command {panel_idx+1}: "{cmd}"'
        draw(ax_, label, assignments)

    # Zone color legend
    handles = [mpatches.Patch(color=COLORS.get(z, "#9CA3AF"), label=z, alpha=0.8)
               for z in available_zones(LEVEL)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.01), title="Warehouse Zones")

    fig.suptitle("Language-Conditioned MAPF: Natural Language → Zone Assignment → Collision-Free Navigation",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "summary.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    main()
