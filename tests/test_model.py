"""
Unit tests for full Clinical Transformer Decoder.
"""

import torch
import pytest
from src.transformer import ClinicalTransformerDecoder


def test_model_instantiation_and_parameter_count():
    model = ClinicalTransformerDecoder(
        vocab_size=120,
        dim=128,
        n_layers=4,
        n_heads=4,
        max_seq_len=64,
    )
    param_count = model.count_parameters()
    assert 400_000 < param_count < 1_200_000, f"Unexpected parameter count: {param_count}"


def test_model_forward_pass():
    vocab_size = 100
    model = ClinicalTransformerDecoder(vocab_size=vocab_size, dim=64, n_layers=2, n_heads=2, max_seq_len=32)
    x = torch.randint(0, vocab_size, (2, 16))
    logits, cache = model(x)
    assert logits.shape == (2, 16, vocab_size)
    assert cache is None


def test_model_cached_generation_parity():
    """
    Greedy decoding with KV-cache must produce the EXACT same token sequence
    as greedy decoding without cache.
    """
    vocab_size = 80
    model = ClinicalTransformerDecoder(vocab_size=vocab_size, dim=64, n_layers=2, n_heads=2, max_seq_len=32)
    model.eval()

    prompt = torch.tensor([[1, 10, 25, 40]])

    tokens_cached = model.generate(prompt, max_new_tokens=8, temperature=0.0, use_cache=True)
    tokens_non_cached = model.generate(prompt, max_new_tokens=8, temperature=0.0, use_cache=False)

    assert tokens_cached == tokens_non_cached, f"Cached: {tokens_cached} vs Non-cached: {tokens_non_cached}"
