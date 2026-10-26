"""
Multi-Head Causal Self-Attention with RoPE and Scaled Dot-Product.
Supports full-sequence training mode and single-token KV-cache inference mode.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from src.rope import RotaryEmbedding, apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    Scaling factor 1 / sqrt(d_k) normalizes dot-product variance to 1.0.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_seq_len: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len

        # Linear projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # RoPE module for head_dim
        self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)

        # Upper triangular causal mask: 0 on and below diagonal, -inf above
        mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        Args:
            x: Input tensor (B, L, dim)
            start_pos: Current position index for RoPE and KV-cache
            kv_cache: Optional tuple of (cached_K, cached_V) each of shape (B, H, max_seq_len, head_dim)
        Returns:
            output: (B, L, dim)
            new_kv_cache: Updated (cached_K, cached_V) if kv_cache was provided
        """
        B, L, _ = x.shape

        # Linear projections: (B, L, dim) -> (B, L, H, head_dim) -> (B, H, L, head_dim)
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(q, start_pos=start_pos)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        if kv_cache is not None:
            # Inference step with pre-allocated KV-cache
            cached_k, cached_v = kv_cache
            # Inplace update at slice [start_pos : start_pos + L]
            cached_k[:, :, start_pos : start_pos + L, :] = k
            cached_v[:, :, start_pos : start_pos + L, :] = v

            # Attend to all valid tokens up to start_pos + L
            keys_to_attend = cached_k[:, :, : start_pos + L, :]
            values_to_attend = cached_v[:, :, : start_pos + L, :]

            # Scaled Dot-Product: (B, H, L, head_dim) @ (B, H, head_dim, total_len) -> (B, H, L, total_len)
            scores = torch.matmul(q, keys_to_attend.transpose(-2, -1)) / math.sqrt(self.head_dim)
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)

            # Output: (B, H, L, total_len) @ (B, H, total_len, head_dim) -> (B, H, L, head_dim)
            attn_out = torch.matmul(weights, values_to_attend)
            new_cache = (cached_k, cached_v)
        else:
            # Parallel training mode with causal masking
            # Scaled Dot-Product: (B, H, L, head_dim) @ (B, H, head_dim, L) -> (B, H, L, L)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Add upper-triangular causal mask
            mask_slice = self.causal_mask[:L, :L].unsqueeze(0).unsqueeze(0)
            scores = scores + mask_slice

            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)

            attn_out = torch.matmul(weights, v)
            new_cache = None

        # Reshape back: (B, H, L, head_dim) -> (B, L, H, head_dim) -> (B, L, dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.dim)
        return self.out_proj(attn_out), new_cache
