"""MAPPO: Multi-Agent PPO with centralized critic and transformer communication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .comm import CommunicationModule


# ── Actor (shared weights — one policy for all agents) ────────────────────────

class SharedActor(nn.Module):
    """Decentralized policy: per-agent comm embedding → action logits."""

    def __init__(self, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., hidden_dim] → [..., n_actions]"""
        return self.net(x)


# ── Centralized critic (mean-pool over comm embeddings → scales to any N) ────

class CentralizedCritic(nn.Module):
    """
    Global value function for CTDE.

    Takes the set of all agents' comm-enhanced embeddings,
    mean-pools them into a single global state vector, then maps to a scalar.
    This design is permutation-invariant and N-agnostic.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, enhanced_all: torch.Tensor) -> torch.Tensor:
        """
        enhanced_all: [B, N, hidden_dim]  or  [N, hidden_dim]
        → mean pool over N → MLP → [B, 1]
        """
        squeeze = enhanced_all.dim() == 2
        if squeeze:
            enhanced_all = enhanced_all.unsqueeze(0)     # [1, N, D]
        global_state = enhanced_all.mean(dim=1)           # [B, D]
        out = self.net(global_state)                      # [B, 1]
        return out.squeeze(0) if squeeze else out         # [1] or [B, 1]


# ── Rollout buffer ────────────────────────────────────────────────────────────

@dataclass
class Transition:
    obs_all:   np.ndarray   # [N, obs_dim]
    actions:   np.ndarray   # [N]  int64
    log_probs: np.ndarray   # [N]  float32
    rewards:   np.ndarray   # [N]  float32
    value:     float        # centralized value (scalar)
    done:      bool


class RolloutBuffer:
    def __init__(self):
        self._buf: List[Transition] = []

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def clear(self) -> None:
        self._buf = []

    def __len__(self) -> int:
        return len(self._buf)

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generalised Advantage Estimation.

        Returns
        -------
        advantages : [T, N]  per-agent advantages
        returns    : [T]     per-timestep discounted returns (for critic target)
        """
        T = len(self._buf)
        N = self._buf[0].rewards.shape[0]
        advantages = np.zeros((T, N), dtype=np.float32)
        gae = np.zeros(N, dtype=np.float32)
        next_val = last_value

        for t in reversed(range(T)):
            tr = self._buf[t]
            mask = 0.0 if tr.done else 1.0
            delta = tr.rewards + gamma * next_val * mask - tr.value
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
            next_val = tr.value

        values = np.array([tr.value for tr in self._buf], dtype=np.float32)
        # Returns: use mean advantage across agents + shared value baseline
        returns = advantages.mean(axis=1) + values
        return advantages, returns


# ── MAPPO ─────────────────────────────────────────────────────────────────────

class MAPPO:
    """
    Multi-Agent PPO (Centralised Training, Decentralised Execution).

    Architecture
    ------------
    comm   : CommunicationModule — shared encoder + transformer attention
    actor  : SharedActor         — decentralised policy (one per agent, shared weights)
    critic : CentralizedCritic   — mean-pool global value (N-agnostic)

    The full pipeline per step:
      obs_all [N, D] → comm → enhanced [N, H]
                            → actor[i]   → action_i        (execution)
                            → critic     → V(global state) (training only)
    """

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        n_actions: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_comm_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.comm   = CommunicationModule(obs_dim, hidden_dim, n_heads, n_comm_layers)
        self.actor  = SharedActor(hidden_dim, n_actions)
        self.critic = CentralizedCritic(hidden_dim)

        for net in (self.comm, self.actor, self.critic):
            net.to(self.device)

        params = (
            list(self.comm.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(
        self, obs_dict: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, int], Dict[int, float], float]:
        """
        Returns (actions, log_probs, value).
        Operates in eval mode — no gradient, no buffer side-effects.
        """
        obs_np = np.stack([obs_dict[i] for i in range(self.n_agents)])  # [N, D]
        obs_t  = torch.FloatTensor(obs_np).to(self.device)

        enhanced = self.comm(obs_t)                    # [N, H]
        logits   = self.actor(enhanced)                # [N, A]
        dist     = Categorical(logits=logits)
        acts     = dist.sample()                       # [N]
        lps      = dist.log_prob(acts)                 # [N]
        value    = self.critic(enhanced).item()        # scalar

        actions   = {i: acts[i].item()  for i in range(self.n_agents)}
        log_probs = {i: lps[i].item()   for i in range(self.n_agents)}
        return actions, log_probs, value

    # ── Buffer ────────────────────────────────────────────────────────────────

    def store(
        self,
        obs_dict:  Dict[int, np.ndarray],
        actions:   Dict[int, int],
        log_probs: Dict[int, float],
        rewards:   Dict[int, float],
        value:     float,
        done:      bool,
    ) -> None:
        n = self.n_agents
        self.buffer.push(Transition(
            obs_all   = np.stack([obs_dict[i]  for i in range(n)]),
            actions   = np.array([actions[i]   for i in range(n)], dtype=np.int64),
            log_probs = np.array([log_probs[i] for i in range(n)], dtype=np.float32),
            rewards   = np.array([rewards[i]   for i in range(n)], dtype=np.float32),
            value     = value,
            done      = done,
        ))

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(self, last_value: float, n_epochs: int = 4) -> Dict[str, float]:
        """Run PPO update on the current buffer. Clears buffer afterwards."""
        advantages, returns = self.buffer.compute_gae(
            last_value, self.gamma, self.gae_lambda
        )

        # Normalise advantages across all agents and timesteps
        adv_flat = advantages.flatten()
        advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # Tensors: obs_all [T, N, D], actions [T, N], log_probs [T, N]
        T = len(self.buffer)
        obs_all    = torch.FloatTensor(np.stack([tr.obs_all   for tr in self.buffer._buf])).to(self.device)
        old_acts   = torch.LongTensor (np.stack([tr.actions   for tr in self.buffer._buf])).to(self.device)
        old_lps    = torch.FloatTensor(np.stack([tr.log_probs for tr in self.buffer._buf])).to(self.device)
        adv_t      = torch.FloatTensor(advantages).to(self.device)    # [T, N]
        returns_t  = torch.FloatTensor(returns).to(self.device)       # [T]

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(n_epochs):
            # Batch all T timesteps through comm in one call: [T, N, D] → [T, N, H]
            enhanced = self.comm(obs_all)                           # [T, N, H]

            logits   = self.actor(enhanced)                        # [T, N, A]
            dist     = Categorical(logits=logits)
            new_lps  = dist.log_prob(old_acts)                     # [T, N]
            entropy  = dist.entropy().mean()

            ratio    = torch.exp(new_lps - old_lps)               # [T, N]
            surr1    = ratio * adv_t
            surr2    = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t
            pol_loss = -torch.min(surr1, surr2).mean()

            values   = self.critic(enhanced).squeeze(-1)           # [T]
            val_loss = F.mse_loss(values, returns_t)

            loss = pol_loss + self.value_loss_coef * val_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.comm.parameters())
                + list(self.actor.parameters())
                + list(self.critic.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

            stats["policy_loss"] += pol_loss.item()
            stats["value_loss"]  += val_loss.item()
            stats["entropy"]     += entropy.item()

        self.buffer.clear()
        return {k: v / n_epochs for k, v in stats.items()}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "comm":      self.comm.state_dict(),
            "actor":     self.actor.state_dict(),
            "critic":    self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.comm.load_state_dict(ckpt["comm"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def set_train(self) -> None:
        for net in (self.comm, self.actor, self.critic):
            net.train()

    def set_eval(self) -> None:
        for net in (self.comm, self.actor, self.critic):
            net.eval()
