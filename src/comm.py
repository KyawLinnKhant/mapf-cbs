"""Multi-agent transformer communication module."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentEncoder(nn.Module):
    """Maps a single agent's raw observation to a latent embedding."""

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: [..., obs_dim] → [..., hidden_dim]"""
        return self.net(obs)


class MultiAgentAttention(nn.Module):
    """
    Multi-head self-attention over N agent embeddings.

    Each agent attends to every other agent's embedding to build a
    global context vector. Supports variable N via a padding mask.

    mask: bool tensor [B, N], True = padding slot (ignored in attention).
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x   : [B, N, D]
        mask: [B, N] bool — True marks padding (ignored in softmax)
        → [B, N, D]
        """
        B, N, D = x.shape
        H, Hd = self.n_heads, self.head_dim

        q = self.q(x).view(B, N, H, Hd).transpose(1, 2)  # [B, H, N, Hd]
        k = self.k(x).view(B, N, H, Hd).transpose(1, 2)
        v = self.v(x).view(B, N, H, Hd).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, N, N]

        if mask is not None:
            # [B, N] → [B, 1, 1, N] so padding keys are masked for all queries/heads
            attn = attn.masked_fill(mask[:, None, None, :], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        # Replace NaN rows (all-masked) with 0
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)              # [B, H, N, Hd]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out(out)

        return self.norm(x + out)               # residual + LayerNorm


class CommunicationModule(nn.Module):
    """
    Full agent communication stack:
      raw obs → encode → n_layers of self-attention → per-agent context embedding.

    All agents share the same encoder weights (permutation equivariant).
    The output dimension equals hidden_dim regardless of how many agents N there are,
    which lets the downstream critic / actor handle any group size.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = AgentEncoder(obs_dim, hidden_dim)
        self.layers = nn.ModuleList([
            MultiAgentAttention(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        obs_all: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        obs_all: [N, obs_dim]  (single step, no batch)  OR
                 [B, N, obs_dim]  (batched)
        mask   : [N] or [B, N] bool — True = padding

        Returns same leading shape + hidden_dim as last dim.
        """
        squeeze = obs_all.dim() == 2
        if squeeze:
            obs_all = obs_all.unsqueeze(0)          # [1, N, obs_dim]
            if mask is not None:
                mask = mask.unsqueeze(0)            # [1, N]

        x = self.encoder(obs_all)                   # [B, N, hidden_dim]
        for layer in self.layers:
            x = layer(x, mask)

        return x.squeeze(0) if squeeze else x       # back to [N, D] or [B, N, D]
