"""
Unit tests for Causal Multi-Head Attention.
Verifies upper-triangular causal masking and KV-cache exact parity.
"""

import torch
import pytest
from src.attention import CausalSelfAttention


def test_attention_shape():
    dim, n_heads, max_seq_len = 128, 4, 64
    attn = CausalSelfAttention(dim=dim, n_heads=n_heads, max_seq_len=max_seq_len)
    x = torch.randn(2, 16, dim)
    out, _ = attn(x)
    assert out.shape == (2, 16, dim)


def test_causal_mask_prevents_future_leakage():
    """
    Perturbing future token at t=10 must have ZERO impact on output at t=5.
    """
    dim, n_heads, max_seq_len = 64, 2, 32
    attn = CausalSelfAttention(dim=dim, n_heads=n_heads, max_seq_len=max_seq_len)
    attn.eval()

    x1 = torch.randn(1, 16, dim)
    x2 = x1.clone()
    # Perturb future token at index 12
    x2[0, 12, :] = x2[0, 12, :] + 50.0

    with torch.no_grad():
        out1, _ = attn(x1)
        out2, _ = attn(x2)

    # Outputs at position 0..11 must be identical
    diff_past = (out1[0, :12, :] - out2[0, :12, :]).abs().max().item()
    assert diff_past < 1e-5, f"Future perturbation leaked into past tokens: diff={diff_past}"

    # Outputs at position 12 must be different
    diff_future = (out1[0, 12:, :] - out2[0, 12:, :]).abs().max().item()
    assert diff_future > 1e-3, "Perturbation should affect token 12"


def test_kv_cache_numerical_exactness():
    """
    Step-by-step autoregressive generation with KV-cache must match
    the full-sequence causal forward pass bit-for-bit (up to float precision).
    """
    dim, n_heads, max_seq_len = 64, 4, 32
    attn = CausalSelfAttention(dim=dim, n_heads=n_heads, max_seq_len=max_seq_len)
    attn.eval()

    x = torch.randn(1, 10, dim)

    # Path 1: Full-sequence causal forward pass
    with torch.no_grad():
        full_out, _ = attn(x)

    # Path 2: Step-by-step with KV-cache
    k_cache = torch.zeros(1, n_heads, max_seq_len, dim // n_heads)
    v_cache = torch.zeros(1, n_heads, max_seq_len, dim // n_heads)
    step_outputs = []

    with torch.no_grad():
        for t in range(10):
            x_t = x[:, t : t + 1, :]
            out_t, (k_cache, v_cache) = attn(x_t, start_pos=t, kv_cache=(k_cache, v_cache))
            step_outputs.append(out_t)

    cached_out = torch.cat(step_outputs, dim=1)

    max_diff = (full_out - cached_out).abs().max().item()
    assert max_diff < 1e-4, f"KV-cache diverged from full-sequence pass: max diff = {max_diff}"
