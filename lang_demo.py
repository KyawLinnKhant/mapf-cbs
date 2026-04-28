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

from src.env import MAPFEnv, OBS_DIM, N_ACTIONS
from src.maps import DIFFICULTY_LEVELS, generate_map
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
    # Inject our language goals instead of random ones
    env.reset(seed=42)
    env.goals = goals

    paths = {i: [env.positions[i]] for i in range(config.n_agents)}

    done = False
    while not done:
        obs = env._all_obs()
        with torch.no_grad():
            actions, _, _ = agent.act(obs)

        # Step but suppress lifelong goal reassignment
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
    p.add_argument("--model",         default="qwen2.5:3b",
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

    print(f"\n── Language-Conditioned MAPF ──────────────────────────")
    print(f"   Level:   {args.level}  ({config.width}×{config.height}, {config.n_agents} agents)")
    print(f"   Model:   {args.model}")
    print(f"   Zones:   {', '.join(available_zones(args.level))}")
    print(f"───────────────────────────────────────────────────────\n")

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

        print(f"\n  Parsing: \"{command}\"")
        zone_assignments = parse_command(command, config.n_agents, args.level, model=args.model)

        print(f"  Zone assignments:")
        for agent_id, zone in sorted(zone_assignments.items()):
            print(f"    agent {agent_id:2d} → {zone}")

        goal_positions = resolve_goals(zone_assignments, args.level, grid)

        print(f"\n  Running episode...")
        paths, info = run_language_episode(agent, config, goal_positions, grid)

        makespan = max(len(p) - 1 for p in paths.values())
        print(f"  Done — goals_reached={info['goals_reached']}  "
              f"collisions={info['collisions']}  makespan={makespan}")

        run_idx += 1
        gif_path = os.path.join(args.out_dir, f"lang_run_{run_idx:02d}.gif")
        png_path = os.path.join(args.out_dir, f"lang_run_{run_idx:02d}.png")

        goal_list = [goal_positions[i] for i in range(config.n_agents)]
        animate(grid, paths, goal_list, interval=200, trail_len=5, save_path=gif_path)
        plot_paths(grid, paths, goal_list,
                   title=f'"{command[:50]}"  |  {config.n_agents} agents  |  makespan={makespan}',
                   save_path=png_path)
        plt.close("all")

        print(f"  Saved → {gif_path}")
        print(f"  Saved → {png_path}\n")


if __name__ == "__main__":
    main()
