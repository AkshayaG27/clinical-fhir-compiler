"""
Rotary Position Embeddings (RoPE) Implementation from First Principles.
Applies 2D Givens rotation matrices to preserve relative distance:
<R_m q, R_n k> = g(q, k, m - n).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) module.
    Precomputes sinusoidal frequencies theta_i = base^(-2i / d_k).
    """

    def __init__(self, dim: int, max_seq_len: int = 128, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"Dimension {dim} must be even for RoPE"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute theta frequencies: theta_i = base^(-2i / d) for i in [0, d/2)
        half_dim = dim // 2
        freqs = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        self.register_buffer("freqs", freqs, persistent=False)

        # Precompute cos and sin tables for [0 .. max_seq_len)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(t, freqs)  # shape: (max_seq_len, half_dim)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, start_pos: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin slices matching x's sequence length and start_pos.
        x: tensor of shape (B, H, L, d_k) or (B, L, H, d_k)
        """
        seq_len = x.shape[-2]
        end_pos = start_pos + seq_len

        if end_pos > self.max_seq_len:
            # Dynamically extend cache if sequence exceeds initial max_seq_len
            t = torch.arange(end_pos, dtype=torch.float32, device=x.device)
            angles = torch.outer(t, self.freqs.to(x.device))
            cos = angles.cos()[start_pos:end_pos]
            sin = angles.sin()[start_pos:end_pos]
        else:
            cos = self.cos_cached[start_pos:end_pos].to(x.device)
            sin = self.sin_cached[start_pos:end_pos].to(x.device)

        return cos, sin


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Applies rotary position embeddings to tensor x of shape (B, H, L, d_k).
    cos, sin: shape (L, d_k // 2)
    """
    # Split into two halves along feature dimension
    d = x.shape[-1]
    half_d = d // 2
    x1 = x[..., :half_d]
    x2 = x[..., half_d:]

    # Reshape cos and sin to broadcast with (B, H, L, half_d)
    # cos/sin shape: (1, 1, L, half_d)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Givens 2D rotation:
    # [cos  -sin] [x1] = [x1 * cos - x2 * sin]
    # [sin   cos] [x2] = [x1 * sin + x2 * cos]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos

    return torch.cat([rx1, rx2], dim=-1)
