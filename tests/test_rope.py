"""
Unit tests for Rotary Position Embeddings (RoPE).
Mathematically validates relative position invariance:
<R_m q, R_n k> = g(q, k, m - n).
"""

import torch
import pytest
from src.rope import RotaryEmbedding, apply_rotary_pos_emb


def test_rope_output_shape():
    batch, heads, seq_len, dim = 2, 4, 16, 32
    x = torch.randn(batch, heads, seq_len, dim)
    rope = RotaryEmbedding(dim=dim, max_seq_len=64)
    cos, sin = rope(x, start_pos=0)
    rx = apply_rotary_pos_emb(x, cos, sin)
    assert rx.shape == x.shape


def test_rope_norm_preservation():
    """A rotation matrix preserves the L2 norm of the vector."""
    dim = 32
    x = torch.randn(1, 1, 10, dim)
    rope = RotaryEmbedding(dim=dim, max_seq_len=64)
    cos, sin = rope(x, start_pos=0)
    rx = apply_rotary_pos_emb(x, cos, sin)

    norm_before = torch.norm(x, p=2, dim=-1)
    norm_after = torch.norm(rx, p=2, dim=-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-5), "RoPE must preserve vector L2 norms"


def test_rope_relative_position_invariance():
    """
    Mathematical Proof Verification:
    Inner product <R_m q, R_n k> depends strictly on (m - n).
    For delta = 3, compare (m=5, n=2) vs (m=10, n=7).
    """
    dim = 32
    rope = RotaryEmbedding(dim=dim, max_seq_len=64)

    # Constant query and key vectors across positions
    q_vec = torch.randn(1, 1, 1, dim)
    k_vec = torch.randn(1, 1, 1, dim)

    # Case A: m=5, n=2 -> delta = 3
    cos_5, sin_5 = rope(q_vec, start_pos=5)
    cos_2, sin_2 = rope(k_vec, start_pos=2)
    q_rot_5 = apply_rotary_pos_emb(q_vec, cos_5, sin_5)
    k_rot_2 = apply_rotary_pos_emb(k_vec, cos_2, sin_2)
    dot_product_A = (q_rot_5 * k_rot_2).sum()

    # Case B: m=10, n=7 -> delta = 3
    cos_10, sin_10 = rope(q_vec, start_pos=10)
    cos_7, sin_7 = rope(k_vec, start_pos=7)
    q_rot_10 = apply_rotary_pos_emb(q_vec, cos_10, sin_10)
    k_rot_7 = apply_rotary_pos_emb(k_vec, cos_7, sin_7)
    dot_product_B = (q_rot_10 * k_rot_7).sum()

    # Case C: m=15, n=12 -> delta = 3
    cos_15, sin_15 = rope(q_vec, start_pos=15)
    cos_12, sin_12 = rope(k_vec, start_pos=12)
    q_rot_15 = apply_rotary_pos_emb(q_vec, cos_15, sin_15)
    k_rot_12 = apply_rotary_pos_emb(k_vec, cos_12, sin_12)
    dot_product_C = (q_rot_15 * k_rot_12).sum()

    # Assert all three inner products are identical within 1e-5 numerical tolerance
    assert torch.allclose(dot_product_A, dot_product_B, atol=1e-5), (
        f"Mismatch: {dot_product_A.item()} vs {dot_product_B.item()}"
    )
    assert torch.allclose(dot_product_B, dot_product_C, atol=1e-5), (
        f"Mismatch: {dot_product_B.item()} vs {dot_product_C.item()}"
    )
