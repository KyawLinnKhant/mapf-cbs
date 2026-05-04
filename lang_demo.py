"""
Language-Conditioned MAPF Demo
===============================
Type a natural language command → agents navigate to those zones → GIF saved.

Usage
-----
  python lang_demo.py                          # expert level, final checkpoint
  python lang_demo.py --level medium
  python lang_demo.py --model llama3.2         # use a different Ollama model

Example commands
----------------
  "Send agents 0, 1, 2 to the loading bay. Put the rest at charging."
  "All agents go to inspection."
  "Split the team: half to storage_a, half to dispatch."
  "Agents 0-5 to loading bay, agents 6-11 to storage."
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.env import MAPFEnv, OBS_DIM, N_ACTIONS, CROP_SIZE
from src.maps import DIFFICULTY_LEVELS, generate_map, sample_positions
from src.mappo import MAPPO
from src.grid import Grid
from src.zones import available_zones, resolve_goals
from src.lang import parse_command
from src.visualize import animate, plot_paths


def run_language_episode(
    agent: MAPPO,
    config,
    goals: dict,
    grid: Grid,
    max_steps: int = 200,
):
    """Run episode with fixed language-assigned goals (no lifelong reassignment)."""
    env = MAPFEnv(config, max_steps=max_steps)

    # Use the caller's grid directly — avoids env.reset() generating a different map,
    # which would make agents appear to walk through walls in the visualisation.
    rng = np.random.default_rng(42)
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
        env.goals = prev_goals  # keep language goals fixed

        done = dones[0]
        for i in range(config.n_agents):
            paths[i].append(env.positions[i])

    return paths, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="checkpoints/mappo_final.pt")
    p.add_argument("--level",         default="expert",
                   choices=["easy", "medium", "hard", "expert"])
    p.add_argument("--model",         default="qwen:4b",
                   help="Ollama model name")
    p.add_argument("--hidden-dim",    type=int, default=128)
    p.add_argument("--n-heads",       type=int, default=4)
    p.add_argument("--n-comm-layers", type=int, default=2)
    p.add_argument("--device",        default="mps")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--out-dir",       default="results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = DIFFICULTY_LEVELS[args.level]

    print(f"\n{'═'*56}")
    print(f"  🤖  Language-Conditioned MAPF  (CBS-Bootstrapped MAPPO)")
    print(f"{'─'*56}")
    print(f"  Level  : {args.level}  ({config.width}×{config.height} grid, {config.n_agents} agents)")
    print(f"  LLM    : {args.model} via Ollama  (local, no cloud API)")
    print(f"  Zones  : {', '.join(available_zones(args.level))}")
    print(f"{'═'*56}\n")
    print("  Try these commands:")
    print('    "Send agents 0, 1, 2 to the loading bay. Put the rest at charging."')
    print('    "All agents go to inspection."')
    print('    "Agents 0-5 to loading bay, agents 6-11 to storage_a."')
    print('    "Split the team: half to storage_a, half to dispatch."')
    print()

    # Load policy
    agent = MAPPO(
        obs_dim       = OBS_DIM,
        n_agents      = config.n_agents,
        n_actions     = N_ACTIONS,
        hidden_dim    = args.hidden_dim,
        n_heads       = args.n_heads,
        n_comm_layers = args.n_comm_layers,
        device        = args.device,
    )
    if os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"No checkpoint found at {args.checkpoint} — using random policy.")
    agent.set_eval()

    # Generate fixed map
    grid, _ = generate_map(config, seed=args.seed)

    run_idx = 0
    print("Type a command (or 'quit' to exit):\n")

    while True:
        try:
            command = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if command.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not command:
            continue

        print(f"\n  {'─'*50}")
        print(f"  ⟳  Parsing via {args.model}...")
        zone_assignments = parse_command(command, config.n_agents, args.level, model=args.model)

        from collections import Counter
        zone_counts = Counter(zone_assignments.values())
        print(f"  ✓  Zone assignments:")
        for zone, count in sorted(zone_counts.items()):
            agents_in_zone = sorted(k for k,v in zone_assignments.items() if v == zone)
            agents_str = ", ".join(str(a) for a in agents_in_zone)
            print(f"       {zone:<18} ← agents [{agents_str}]  ({count} robots)")

        goal_positions = resolve_goals(zone_assignments, args.level, grid)

        print(f"\n  ⟳  Navigating {config.n_agents} agents collision-free...")
        paths, info = run_language_episode(agent, config, goal_positions, grid)

        makespan = max(len(p) - 1 for p in paths.values())
        success = info.get('goals_reached', 0)
        print(f"  ✓  Done — goals={success}  collisions={info['collisions']}  makespan={makespan} steps")

        run_idx += 1
        gif_path = os.path.join(args.out_dir, f"lang_run_{run_idx:02d}.gif")
        png_path = os.path.join(args.out_dir, f"lang_run_{run_idx:02d}.png")

        goal_list = [goal_positions[i] for i in range(config.n_agents)]
        # Subsample to every 2nd frame so GIF saves in ~15s instead of ~30s
        sparse_paths = {i: paths[i][::2] for i in paths}
        animate(grid, sparse_paths, goal_list, interval=300, trail_len=5, save_path=gif_path)
        plot_paths(grid, paths, goal_list,
                   title=f'"{command[:50]}"  |  {config.n_agents} agents  |  makespan={makespan}',
                   save_path=png_path)
        plt.close("all")

        print(f"\n  📁  {gif_path}  (animation)")
        print(f"  📁  {png_path}  (static)")
        print(f"  {'─'*50}\n")

        # Auto-open GIF on macOS so it plays immediately
        os.system(f"open '{gif_path}'")


if __name__ == "__main__":
    main()
