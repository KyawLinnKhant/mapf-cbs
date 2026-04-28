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
from src.dynamic import spawn_dynamic_obstacles


# ── Evaluation helpers ────────────────────────────────────────────────────────

def eval_rl(
    agent: MAPPO,
    config,
    n_episodes: int,
    seed_offset: int = 99_999,
    n_dynamic_obstacles: int = 0,
    dynamic_pattern: str = "mixed",
) -> Dict[str, float]:
    """Run trained policy on n_episodes of freshly generated maps."""
    env = MAPFEnv(config, max_steps=256,
                  n_dynamic_obstacles=n_dynamic_obstacles,
                  dynamic_pattern=dynamic_pattern)
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
    max_ct_nodes: int = 10_000,
) -> Dict[str, float]:
    """Run CBS solver on n_episodes of the same map seeds."""
    rng = np.random.default_rng(seed_offset)
    soc_list, makespan_list, success_list, time_list = [], [], [], []

    for ep in range(n_episodes):
        if (ep + 1) % 20 == 0:
            print(f"  CBS episode {ep + 1}/{n_episodes}", flush=True)

        grid, _ = generate_map(config, seed=seed_offset + ep)
        starts   = sample_positions(grid, config.n_agents, rng)
        goals    = sample_positions(grid, config.n_agents, rng, exclude=starts)

        if starts is None or goals is None:
            success_list.append(0.0)
            continue

        t0       = time.monotonic()
        solution = cbs(grid, starts, goals, max_t=max_t, max_ct_nodes=max_ct_nodes)
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


# ── Dynamic obstacle comparison ───────────────────────────────────────────────

def eval_cbs_dynamic(
    config,
    n_episodes: int,
    n_dynamic_obstacles: int,
    seed_offset: int = 99_999,
    max_t: int = 256,
    max_ct_nodes: int = 10_000,
    dynamic_pattern: str = "mixed",
) -> Dict[str, float]:
    """
    CBS failure benchmark under dynamic obstacles.

    CBS plans once on the static map (no dynamic obstacles visible).
    The plan is then executed step-by-step while dynamic obstacles move.
    We measure how many of the CBS-planned positions are blocked at execution
    time by a dynamic obstacle — the 'plan invalidation rate'.

    This is the key paper result: CBS degrades catastrophically; MARL adapts.
    """
    rng = np.random.default_rng(seed_offset)
    success_list, collision_list, invalidation_list = [], [], []

    for ep in range(n_episodes):
        if (ep + 1) % 20 == 0:
            print(f"  CBS-dynamic episode {ep + 1}/{n_episodes}", flush=True)

        grid, _ = generate_map(config, seed=seed_offset + ep)
        starts   = sample_positions(grid, config.n_agents, rng)
        goals    = sample_positions(grid, config.n_agents, rng, exclude=starts)

        if starts is None or goals is None:
            success_list.append(0.0)
            collision_list.append(float(max_t))
            invalidation_list.append(1.0)
            continue

        # CBS plans on static map — dynamic obstacles invisible
        solution = cbs(grid, starts, goals, max_t=max_t, max_ct_nodes=max_ct_nodes)

        if solution is None:
            success_list.append(0.0)
            collision_list.append(float(max_t))
            invalidation_list.append(1.0)
            continue

        # Spawn dynamic obstacles (not overlapping starts/goals)
        dyn_rng = np.random.default_rng(seed_offset + ep + 100_000)
        dyn_obs = spawn_dynamic_obstacles(
            n_dynamic_obstacles, grid, dyn_rng,
            exclude=list(starts) + list(goals),
            pattern=dynamic_pattern,
        )

        # Simulate plan execution with moving obstacles
        makespan = max(len(p) - 1 for p in solution.values())
        plan_collisions = 0
        invalidated_steps = 0
        total_steps = 0
        agent_positions = {i: starts[i] for i in range(config.n_agents)}

        for t in range(min(makespan, max_t)):
            # Advance dynamic obstacles
            agent_cells = set(agent_positions.values())
            for obs in dyn_obs:
                obs.step(occupied=agent_cells)
            dyn_cells = {obs.pos for obs in dyn_obs}

            # Try to execute CBS plan step
            for i, path in solution.items():
                if t + 1 < len(path):
                    next_pos = path[t + 1]
                    total_steps += 1
                    if next_pos in dyn_cells:
                        # CBS plan step blocked by dynamic obstacle
                        plan_collisions += 1
                        invalidated_steps += 1
                    else:
                        agent_positions[i] = next_pos

        inv_rate = invalidated_steps / total_steps if total_steps > 0 else 0.0
        # "success" = CBS plan completed with zero plan invalidations
        success_list.append(1.0 if plan_collisions == 0 else 0.0)
        collision_list.append(float(plan_collisions))
        invalidation_list.append(inv_rate)

    return {
        "success_rate":       float(np.mean(success_list)),
        "plan_collisions":    float(np.mean(collision_list)),
        "invalidation_rate":  float(np.mean(invalidation_list)),
    }


def print_dynamic_report(
    rl: Dict,
    cbs_static: Dict,
    cbs_dyn: Dict,
    level: str,
    n: int,
    n_dyn: int,
) -> None:
    W = 70
    print()
    print("=" * W)
    print(f"  Dynamic Obstacle Evaluation — level: {level} | episodes: {n} | obstacles: {n_dyn}")
    print("=" * W)
    print(f"  {'Metric':<32} {'MARL (ours)':>12}  {'CBS static':>10}  {'CBS+dyn':>10}")
    print(f"  {'-'*64}")

    def row(label, rv, sv, dv, fmt=".3f"):
        r = f"{rv:{fmt}}" if rv is not None else "  N/A"
        s = f"{sv:{fmt}}" if sv is not None else "  N/A"
        d = f"{dv:{fmt}}" if dv is not None else "  N/A"
        print(f"  {label:<32} {r:>12}  {s:>10}  {d:>10}")

    row("Success rate",            rl["success_rate"],      cbs_static["success_rate"],  cbs_dyn["success_rate"])
    row("Avg goals reached",       rl["goals_reached"],     None,                        None,          fmt=".2f")
    row("Avg collisions",          rl["collisions"],        None,                        cbs_dyn["plan_collisions"], fmt=".2f")
    row("CBS plan invalidation %", None,                    None,                        cbs_dyn["invalidation_rate"] * 100, fmt=".1f")
    print("=" * W)

    marl_gap = rl["success_rate"] - cbs_dyn["success_rate"]
    if marl_gap >= 0:
        print(f"  ✓  MARL outperforms CBS+dynamic by {marl_gap:.1%} success rate.")
    else:
        print(f"  ~  MARL within {-marl_gap:.1%} of CBS+dynamic.")
    print()


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
    if rl["success_rate"] >= cbs_m["success_rate"] * 0.9:
        print("  ✓  MARL matches or nearly matches CBS on success rate.")
    else:
        gap = cbs_m["success_rate"] - rl["success_rate"]
        print(f"  ✗  MARL trails CBS by {gap:.1%} on success rate — train longer.")
    print()


def save_csv(rl: Dict, cbs_m: Dict, level: str, n: int, path: str) -> None:
    """Append one row per level to a CSV file (creates headers if new)."""
    import csv, os
    fieldnames = [
        "level", "episodes",
        "marl_success", "marl_makespan", "marl_goals", "marl_collisions",
        "cbs_success", "cbs_makespan", "cbs_soc", "cbs_solve_ms",
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({
            "level": level, "episodes": n,
            "marl_success":    round(rl["success_rate"], 4),
            "marl_makespan":   round(rl["makespan"], 2),
            "marl_goals":      round(rl["goals_reached"], 2),
            "marl_collisions": round(rl["collisions"], 2),
            "cbs_success":     round(cbs_m["success_rate"], 4),
            "cbs_makespan":    round(cbs_m["makespan"], 2),
            "cbs_soc":         round(cbs_m["soc"], 2),
            "cbs_solve_ms":    round(cbs_m["solve_time_ms"], 2),
        })
    print(f"  CSV row appended → {path}")


def print_latex_row(rl: Dict, cbs_m: Dict, level: str) -> None:
    """Print a single LaTeX table row for copy-paste into paper."""
    agents = {"easy": 2, "medium": 4, "hard": 8, "expert": 12}.get(level, "?")
    grid   = {"easy": "7×7", "medium": "11×11", "hard": "15×15", "expert": "20×20"}.get(level, "?")
    print(
        f"  {level.capitalize()} ({grid}, {agents}a) & "
        f"{rl['success_rate']:.2f} & {rl['makespan']:.1f} & {rl['goals_reached']:.1f} & {rl['collisions']:.1f} & "
        f"{cbs_m['success_rate']:.2f} & {cbs_m['makespan']:.1f} & {cbs_m['soc']:.1f} & {cbs_m['solve_time_ms']:.1f} \\\\"
    )


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
    p.add_argument("--cbs-only",           action="store_true",
                   help="Skip RL evaluation (useful when no checkpoint exists yet)")
    p.add_argument("--csv",                default=None,
                   help="Append results row to this CSV file (e.g. results/eval.csv)")
    p.add_argument("--latex",              action="store_true",
                   help="Print a LaTeX table row for the paper")
    p.add_argument("--dynamic-obstacles",  type=int, default=0,
                   help="Number of dynamic obstacles (0=disabled). Enables dynamic eval mode.")
    p.add_argument("--dynamic-pattern",    default="mixed",
                   choices=["random_walk", "patrol", "mixed"],
                   help="Movement pattern for dynamic obstacles")
    args = parse_args(p)

    config = DIFFICULTY_LEVELS[args.level]

    n_dyn = args.dynamic_obstacles

    # ── Dynamic obstacle mode ────────────────────────────────────────────────
    if n_dyn > 0:
        print(f"Dynamic obstacle mode: {n_dyn} obstacles ({args.dynamic_pattern})")

        print(f"Running CBS static baseline on {args.n_episodes} episodes…")
        cbs_static = eval_cbs(config, args.n_episodes)

        print(f"Running CBS plan-execution under dynamic obstacles…")
        cbs_dyn = eval_cbs_dynamic(
            config, args.n_episodes, n_dyn,
            dynamic_pattern=args.dynamic_pattern,
        )

        rl_metrics = None
        if not args.cbs_only:
            import os
            if not os.path.exists(args.checkpoint):
                print(f"Checkpoint not found: {args.checkpoint} — skipping RL eval.")
            else:
                agent = MAPPO(
                    obs_dim=OBS_DIM, n_agents=config.n_agents, n_actions=N_ACTIONS,
                    hidden_dim=args.hidden_dim, n_heads=args.n_heads,
                    n_comm_layers=args.n_comm_layers, device=args.device,
                )
                agent.load(args.checkpoint)
                print(f"Running RL policy with {n_dyn} dynamic obstacles…")
                rl_metrics = eval_rl(
                    agent, config, args.n_episodes,
                    n_dynamic_obstacles=n_dyn,
                    dynamic_pattern=args.dynamic_pattern,
                )

        if rl_metrics is not None:
            print_dynamic_report(rl_metrics, cbs_static, cbs_dyn,
                                 args.level, args.n_episodes, n_dyn)
        else:
            print(f"\nCBS static success:  {cbs_static['success_rate']:.3f}")
            print(f"CBS+dynamic success: {cbs_dyn['success_rate']:.3f}")
            print(f"CBS plan invalidation rate: {cbs_dyn['invalidation_rate']:.1%}")
        return

    # ── Standard (static) mode ───────────────────────────────────────────────
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
        if args.csv:
            save_csv(rl_metrics, cbs_metrics, args.level, args.n_episodes, args.csv)
        if args.latex:
            print("\n  LaTeX row:")
            print_latex_row(rl_metrics, cbs_metrics, args.level)
    else:
        # CBS-only report
        print(f"\nCBS baseline — level: {args.level} | episodes: {args.n_episodes}")
        for k, v in cbs_metrics.items():
            print(f"  {k:<20} {v:.4f}")


def parse_args(p: argparse.ArgumentParser) -> argparse.Namespace:
    return p.parse_args()


if __name__ == "__main__":
    main()
