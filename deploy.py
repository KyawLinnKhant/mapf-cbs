"""
MARL Deployment Visualiser
==========================
Loads the trained MAPPO checkpoint, runs one episode on each difficulty
level, and saves an animated GIF + static path overview for each.

Usage
-----
  python deploy.py                                  # all levels, final checkpoint
  python deploy.py --level expert                   # one level only
  python deploy.py --checkpoint checkpoints/mappo_step1000000.pt
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.env import MAPFEnv, OBS_DIM, N_ACTIONS
from src.mappo import MAPPO
from src.maps import DIFFICULTY_LEVELS, generate_map, sample_positions
from src.visualize import animate, plot_paths


def run_episode(agent: MAPPO, config, seed: int = 42):
    """Run one episode and return recorded paths + grid + goals."""
    env = MAPFEnv(config, max_steps=256)
    obs = env.reset(seed=seed)

    # Record paths: {agent_id: [pos_t0, pos_t1, ...]}
    paths = {i: [env.positions[i]] for i in range(config.n_agents)}
    goals = [env.goals[i] for i in range(config.n_agents)]

    done = False
    while not done:
        with torch.no_grad():
            actions, _, _ = agent.act(obs)
        obs, _, dones, info = env.step(actions)
        done = dones[0]
        for i in range(config.n_agents):
            paths[i].append(env.positions[i])

    return env.grid, paths, goals, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="checkpoints/mappo_final.pt")
    p.add_argument("--level",         default="all",
                   choices=["all", "easy", "medium", "hard", "expert"])
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--hidden-dim",    type=int, default=128)
    p.add_argument("--n-heads",       type=int, default=4)
    p.add_argument("--n-comm-layers", type=int, default=2)
    p.add_argument("--device",        default="mps")
    p.add_argument("--out-dir",       default="results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    levels = (
        ["easy", "medium", "hard", "expert"]
        if args.level == "all"
        else [args.level]
    )

    for level in levels:
        config = DIFFICULTY_LEVELS[level]
        print(f"\n── {level.upper()}  ({config.width}×{config.height}, {config.n_agents} agents) ──")

        agent = MAPPO(
            obs_dim       = OBS_DIM,
            n_agents      = config.n_agents,
            n_actions     = N_ACTIONS,
            hidden_dim    = args.hidden_dim,
            n_heads       = args.n_heads,
            n_comm_layers = args.n_comm_layers,
            device        = args.device,
        )
        agent.load(args.checkpoint)
        agent.set_eval()

        grid, paths, goals, info = run_episode(agent, config, seed=args.seed)

        soc      = sum(len(p) - 1 for p in paths.values())
        makespan = max(len(p) - 1 for p in paths.values())
        print(f"  goals_reached={info['goals_reached']}  "
              f"collisions={info['collisions']}  "
              f"makespan={makespan}  SoC={soc}")

        gif_path  = os.path.join(args.out_dir, f"marl_{level}.gif")
        png_path  = os.path.join(args.out_dir, f"marl_{level}.png")

        animate(grid, paths, goals,
                interval=200, trail_len=5,
                save_path=gif_path)

        plot_paths(grid, paths, goals,
                   title=f"MARL — {level} | {config.n_agents} agents | "
                         f"goals={info['goals_reached']} | makespan={makespan}",
                   save_path=png_path)

        plt.close("all")
        print(f"  Saved → {gif_path}")
        print(f"  Saved → {png_path}")

    print("\nDone. Check results/ for all GIFs and PNGs.")


if __name__ == "__main__":
    main()
