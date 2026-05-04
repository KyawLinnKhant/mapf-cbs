"""Training loop — MAPPO + curriculum + CBS annealing."""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch

from .curriculum import CBSAnnealer, DifficultyScheduler
from .env import MAPFEnv, OBS_DIM, N_ACTIONS
from .mappo import MAPPO


class Trainer:
    """
    Orchestrates MAPPO training over *total_steps* environment steps.

    Every *n_steps* steps a PPO update is performed (standard on-policy loop).
    Episode boundaries inside a rollout are handled transparently: when an
    episode ends, the env is reset and collection continues until n_steps total
    steps are gathered, then the buffer is flushed.

    CBS shaping
    -----------
    During phases A and B the env's CBS oracle suggests the greedy action for
    each agent. If the agent happens to pick that action it receives an extra
    bonus = cbs_weight × cbs_bonus on top of the environment reward.
    The weight decays to zero by phase C, leaving pure RL reward.
    """

    def __init__(
        self,
        # Volume
        total_steps:    int   = 2_000_000,
        n_steps:        int   = 256,          # rollout length before each update
        n_epochs:       int   = 4,
        # Architecture
        hidden_dim:     int   = 128,
        n_heads:        int   = 4,
        n_comm_layers:  int   = 2,
        # Optimisation
        lr:             float = 3e-4,
        gamma:          float = 0.99,
        gae_lambda:     float = 0.95,
        clip_eps:       float = 0.2,
        entropy_coef:   float = 0.01,
        # CBS oracle
        warmup_steps:   int   = 50_000,
        anneal_end:     int   = 200_000,
        cbs_bonus:      float = 0.3,
        # Curriculum
        start_level:    str   = "easy",
        advance_threshold: float = 0.70,
        env_max_steps:  int   = 256,
        # Output
        save_dir:       str   = "checkpoints",
        log_interval:   int   = 2_000,        # print every N env steps
        save_interval:  int   = 100_000,      # checkpoint every N env steps
        device:         str   = "cpu",
    ):
        self.total_steps   = total_steps
        self.n_steps       = n_steps
        self.n_epochs      = n_epochs
        self.cbs_bonus     = cbs_bonus
        self.log_interval  = log_interval
        self.save_interval = save_interval

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        self.n_dynamic_obstacles = 0
        self.dynamic_pattern     = "mixed"

        # Curriculum + CBS annealing
        self.scheduler    = DifficultyScheduler(start_level=start_level,
                                                advance_threshold=advance_threshold)
        self.annealer     = CBSAnnealer(warmup_steps, anneal_end)
        self.env_max_steps = env_max_steps

        # Environment (starts at the chosen difficulty)
        config   = self.scheduler.current_config
        self.env = MAPFEnv(config, max_steps=env_max_steps,
                           n_dynamic_obstacles=self.n_dynamic_obstacles,
                           dynamic_pattern=self.dynamic_pattern)

        # MAPPO agent (critic scales to any N via mean-pool)
        self.agent = MAPPO(
            obs_dim       = OBS_DIM,
            n_agents      = config.n_agents,
            n_actions     = N_ACTIONS,
            hidden_dim    = hidden_dim,
            n_heads       = n_heads,
            n_comm_layers = n_comm_layers,
            lr            = lr,
            gamma         = gamma,
            gae_lambda    = gae_lambda,
            clip_eps      = clip_eps,
            entropy_coef  = entropy_coef,
            device        = device,
        )

        self._global_step   = 0
        self._episode       = 0
        self._ep_rewards:   list[float] = []
        self._ep_goals:     list[int]   = []
        self._ep_collisions:list[int]   = []

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> None:
        device = next(self.agent.actor.parameters()).device
        print(
            f"MAPPO training | device={device} | "
            f"start_level={self.scheduler.current_level} | "
            f"total_steps={self.total_steps:,}"
        )
        print(
            f"CBS annealing  | phase A end={self.annealer.warmup_steps:,} | "
            f"phase B end={self.annealer.anneal_end:,}"
        )
        print("-" * 72)

        obs            = self.env.reset()
        ep_reward      = np.zeros(self.env.n_agents)
        last_log_step  = 0
        last_save_step = 0
        t0             = time.monotonic()

        self.agent.set_train()

        while self._global_step < self.total_steps:
            # ── Collect n_steps ───────────────────────────────────────────────
            for _ in range(self.n_steps):
                actions, log_probs, value = self.agent.act(obs)
                cbs_weight                = self.annealer.step()

                next_obs, rewards, dones, info = self.env.step(actions)

                # CBS shaping bonus
                if cbs_weight > 0 and self.env.cbs_available:
                    for i in range(self.env.n_agents):
                        cbs_a = self.env.get_cbs_action(i)
                        if cbs_a is not None and actions[i] == cbs_a:
                            rewards[i] += cbs_weight * self.cbs_bonus

                self.agent.store(obs, actions, log_probs, rewards, value, dones[0])
                ep_reward += np.array([rewards[i] for i in range(self.env.n_agents)])
                obs        = next_obs
                self._global_step += 1

                if dones[0]:
                    # Episode ended — record stats, check curriculum
                    self._ep_rewards.append(ep_reward.mean())
                    self._ep_goals.append(info["goals_reached"])
                    self._ep_collisions.append(info["collisions"])
                    ep_reward = np.zeros(self.env.n_agents)
                    self._episode += 1

                    success_rate = min(
                        1.0,
                        info["success"] / max(1, self.env.n_agents),
                    )
                    new_level = self.scheduler.record(success_rate)
                    if new_level is not None:
                        new_cfg  = self.scheduler.current_config
                        self.agent.buffer.clear()          # flush stale transitions
                        self.env = MAPFEnv(new_cfg, max_steps=self.env_max_steps,
                                           n_dynamic_obstacles=self.n_dynamic_obstacles,
                                           dynamic_pattern=self.dynamic_pattern)
                        self.agent.n_agents = new_cfg.n_agents
                        ep_reward = np.zeros(new_cfg.n_agents)
                        print(
                            f"\n  [Curriculum ↑] → {new_level} "
                            f"(step {self._global_step:,}, ep {self._episode})\n"
                        )

                    obs = self.env.reset()

            # ── PPO update ────────────────────────────────────────────────────
            _, _, last_val = self.agent.act(obs)
            losses         = self.agent.update(last_val, self.n_epochs)

            # ── Logging ───────────────────────────────────────────────────────
            if self._global_step - last_log_step >= self.log_interval:
                last_log_step = self._global_step
                w = min(100, len(self._ep_rewards))
                recent_r = np.mean(self._ep_rewards[-w:]) if self._ep_rewards else 0.0
                recent_g = np.mean(self._ep_goals[-w:])   if self._ep_goals   else 0.0
                recent_c = np.mean(self._ep_collisions[-w:]) if self._ep_collisions else 0.0
                elapsed  = time.monotonic() - t0
                sps      = self._global_step / max(elapsed, 1)
                print(
                    f"step={self._global_step:>8,} | "
                    f"ep={self._episode:>6,} | "
                    f"lvl={self.scheduler.current_level:<6} | "
                    f"ph={self.annealer.phase} cbs={self.annealer.weight:.2f} | "
                    f"rew={recent_r:+.3f} | "
                    f"goals={recent_g:.1f} | "
                    f"coll={recent_c:.1f} | "
                    f"pol={losses['policy_loss']:.4f} | "
                    f"val={losses['value_loss']:.4f} | "
                    f"ent={losses['entropy']:.3f} | "
                    f"{sps:.0f} sps"
                )

            # ── Checkpoint ────────────────────────────────────────────────────
            if self._global_step - last_save_step >= self.save_interval:
                last_save_step = self._global_step
                path = os.path.join(self.save_dir, f"mappo_step{self._global_step}.pt")
                self.agent.save(path)
                print(f"  [Saved] {path}")

        self.agent.save(os.path.join(self.save_dir, "mappo_final.pt"))
        print(f"\nTraining complete ({self._global_step:,} steps, {self._episode} episodes).")
