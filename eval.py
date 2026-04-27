"""
MAPF-MARL Evaluation
====================
Compares trained MARL policy vs CBS baseline on held-out unseen maps.

Usage
-----
  python eval.py                                       # auto-finds checkpoints/mappo_final.pt
  python eval.py --checkpoint checkpoints/mappo_step500000.pt --level hard
  python eval.py --level expert --n-episodes 200
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Optional

import numpy as np
import torch

from src.cbs import cbs
from src.env import MAPFEnv, OBS_DIM, N_ACTIONS
from src.mappo import MAPPO
from src.maps import DIFFICULTY_LEVELS, generate_map, sample_positions


# ── Evaluation helpers ────────────────────────────────────────────────────────

def eval_rl(
    agent: MAPPO,
    config,
    n_episodes: int,
    seed_offset: int = 99_999,
) -> Dict[str, float]:
    """Run trained policy on n_episodes of freshly generated maps."""
    env = MAPFEnv(config, max_steps=256)
    agent.set_eval()

    goals_list, coll_list, steps_list, success_list = [], [], [], []

    for ep in range(n_episodes):
        obs  = env.reset(seed=seed_offset + ep)
        done = False

        while not done:
            with torch.no_grad():
                actions, _, _ = agent.act(obs)
            obs, _, dones, info = env.step(actions)
            done = dones[0]

        goals_list.append(info["goals_reached"])
        coll_list.append(info["collisions"])
        steps_list.append(info["makespan"])
        success_list.append(float(info["success"] >= config.n_agents))

    return {
        "success_rate":  float(np.mean(success_list)),
        "goals_reached": float(np.mean(goals_list)),
        "makespan":      float(np.mean(steps_list)),
        "collisions":    float(np.mean(coll_list)),
    }


def eval_cbs(
    config,
    n_episodes: int,
    seed_offset: int = 99_999,
    max_t: int = 256,
) -> Dict[str, float]:
    """Run CBS solver on n_episodes of the same map seeds."""
    rng = np.random.default_rng(seed_offset)
    soc_list, makespan_list, success_list, time_list = [], [], [], []

    for ep in range(n_episodes):
        grid, _ = generate_map(config, seed=seed_offset + ep)
        starts   = sample_positions(grid, config.n_agents, rng)
        goals    = sample_positions(grid, config.n_agents, rng, exclude=starts)

        if starts is None or goals is None:
            success_list.append(0.0)
            continue

        t0       = time.monotonic()
        solution = cbs(grid, starts, goals, max_t=max_t)
        elapsed  = time.monotonic() - t0
        time_list.append(elapsed)

        if solution is None:
            success_list.append(0.0)
            soc_list.append(max_t * config.n_agents)
            makespan_list.append(max_t)
        else:
            soc      = sum(len(p) - 1 for p in solution.values())
            makespan = max(len(p) - 1 for p in solution.values())
            soc_list.append(soc)
            makespan_list.append(makespan)
            success_list.append(1.0)

    return {
        "success_rate":   float(np.mean(success_list)),
        "soc":            float(np.mean(soc_list))      if soc_list      else float("inf"),
        "makespan":       float(np.mean(makespan_list)) if makespan_list else float("inf"),
        "solve_time_ms":  float(np.mean(time_list) * 1000) if time_list else 0.0,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(rl: Dict, cbs_m: Dict, level: str, n: int) -> None:
    W = 62
    print()
    print("=" * W)
    print(f"  Evaluation  —  level: {level}  |  episodes: {n}")
    print("=" * W)
    print(f"  {'Metric':<28} {'MARL (ours)':>12}  {'CBS baseline':>14}")
    print(f"  {'-'*56}")

    def row(label, rl_val, cbs_val, fmt=".3f"):
        rv = f"{rl_val:{fmt}}"  if rl_val  is not None else "   N/A"
        cv = f"{cbs_val:{fmt}}" if cbs_val is not None else "    N/A"
        print(f"  {label:<28} {rv:>12}  {cv:>14}")

    row("Success rate",         rl["success_rate"],  cbs_m["success_rate"])
    row("Avg makespan (steps)", rl["makespan"],       cbs_m["makespan"],  fmt=".1f")
    row("Avg goals reached",    rl["goals_reached"],  None,               fmt=".2f")
    row("Avg collisions",       rl["collisions"],     None,               fmt=".2f")
    row("CBS SoC",              None,                 cbs_m["soc"],       fmt=".1f")
    row("CBS solve time (ms)",  None,                 cbs_m["solve_time_ms"], fmt=".1f")
    print("=" * W)
    # Quick verdict
    if rl["success_rate"] >= cbs_m["success_rate"] * 0.9:
        print("  ✓  MARL matches or nearly matches CBS on success rate.")
    else:
        gap = cbs_m["success_rate"] - rl["success_rate"]
        print(f"  ✗  MARL trails CBS by {gap:.1%} on success rate — train longer.")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="checkpoints/mappo_final.pt")
    p.add_argument("--level",         default="medium",
                   choices=["easy", "medium", "hard", "expert"])
    p.add_argument("--n-episodes",    type=int, default=100)
    p.add_argument("--hidden-dim",    type=int, default=128)
    p.add_argument("--n-heads",       type=int, default=4)
    p.add_argument("--n-comm-layers", type=int, default=2)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--cbs-only",      action="store_true",
                   help="Skip RL evaluation (useful when no checkpoint exists yet)")
    args = parse_args(p)

    config = DIFFICULTY_LEVELS[args.level]

    # CBS baseline (always runs)
    print(f"Running CBS baseline on {args.n_episodes} episodes (level={args.level})…")
    cbs_metrics = eval_cbs(config, args.n_episodes)

    rl_metrics: Optional[Dict] = None
    if not args.cbs_only:
        import os
        if not os.path.exists(args.checkpoint):
            print(f"Checkpoint not found: {args.checkpoint}")
            print("Run  python train.py  first, or use --cbs-only for a baseline-only report.")
            return

        agent = MAPPO(
            obs_dim      = OBS_DIM,
            n_agents     = config.n_agents,
            n_actions    = N_ACTIONS,
            hidden_dim   = args.hidden_dim,
            n_heads      = args.n_heads,
            n_comm_layers= args.n_comm_layers,
            device       = args.device,
        )
        agent.load(args.checkpoint)
        print(f"Loaded: {args.checkpoint}")
        print(f"Running RL policy on {args.n_episodes} episodes…")
        rl_metrics = eval_rl(agent, config, args.n_episodes)

    if rl_metrics is not None:
        print_report(rl_metrics, cbs_metrics, args.level, args.n_episodes)
    else:
        # CBS-only report
        print(f"\nCBS baseline — level: {args.level} | episodes: {args.n_episodes}")
        for k, v in cbs_metrics.items():
            print(f"  {k:<20} {v:.4f}")


def parse_args(p: argparse.ArgumentParser) -> argparse.Namespace:
    return p.parse_args()


if __name__ == "__main__":
    main()
