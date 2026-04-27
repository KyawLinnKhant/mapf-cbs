"""
MAPF-MARL Training
==================
Train MAPPO with transformer communication, CBS-bootstrapped curriculum,
and lifelong goals on procedurally generated maps.

Usage
-----
  python train.py                              # defaults (2M steps, starts easy)
  python train.py --total-steps 5000000 --start-level medium
  python train.py --device mps                # Apple Silicon GPU
  python train.py --device cuda               # NVIDIA GPU
"""

import argparse
import torch

from src.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MAPPO on MAPF with CBS curriculum"
    )
    # Volume
    p.add_argument("--total-steps",    type=int,   default=2_000_000)
    p.add_argument("--n-steps",        type=int,   default=256,
                   help="Rollout length per PPO update")
    p.add_argument("--n-epochs",       type=int,   default=4)
    # Architecture
    p.add_argument("--hidden-dim",     type=int,   default=128)
    p.add_argument("--n-heads",        type=int,   default=4)
    p.add_argument("--n-comm-layers",  type=int,   default=2)
    # Optimisation
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--entropy-coef",   type=float, default=0.01)
    # CBS oracle
    p.add_argument("--warmup-steps",   type=int,   default=50_000,
                   help="Phase A length (CBS weight=1.0)")
    p.add_argument("--anneal-end",     type=int,   default=200_000,
                   help="Step at which CBS weight reaches 0")
    p.add_argument("--cbs-bonus",      type=float, default=0.3,
                   help="Extra reward for matching CBS action")
    # Curriculum
    p.add_argument("--start-level",    default="easy",
                   choices=["easy", "medium", "hard", "expert"])
    # Output
    p.add_argument("--save-dir",       default="checkpoints")
    p.add_argument("--log-interval",   type=int,   default=2_000)
    p.add_argument("--save-interval",  type=int,   default=100_000)
    p.add_argument("--device",         default="auto",
                   help="cpu | cuda | mps | auto")
    return p.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    trainer = Trainer(
        total_steps    = args.total_steps,
        n_steps        = args.n_steps,
        n_epochs       = args.n_epochs,
        hidden_dim     = args.hidden_dim,
        n_heads        = args.n_heads,
        n_comm_layers  = args.n_comm_layers,
        lr             = args.lr,
        entropy_coef   = args.entropy_coef,
        warmup_steps   = args.warmup_steps,
        anneal_end     = args.anneal_end,
        cbs_bonus      = args.cbs_bonus,
        start_level    = args.start_level,
        save_dir       = args.save_dir,
        log_interval   = args.log_interval,
        save_interval  = args.save_interval,
        device         = device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
