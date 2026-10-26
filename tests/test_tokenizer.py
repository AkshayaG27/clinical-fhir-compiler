"""
Unit tests for ClinicalTokenizer.
"""

from pathlib import Path
import pytest
from src.tokenizer import ClinicalTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
VOCAB_PATH = BASE_DIR / "data" / "tokenizer" / "vocab.json"


@pytest.fixture
def tokenizer():
    tok = ClinicalTokenizer()
    tok.load(VOCAB_PATH)
    return tok


def test_special_tokens(tokenizer):
    assert tokenizer.token2id["<pad>"] == 0
    assert tokenizer.token2id["<bos>"] == 1
    assert tokenizer.token2id["<eos>"] == 2
    assert tokenizer.token2id["<sep>"] == 3
    assert tokenizer.token2id["<unk>"] == 4


def test_ir_roundtrip(tokenizer):
    original_ir = "MED|paracetamol|650mg|BD|pc|3d"
    tokens = tokenizer.tokenize(original_ir)
    assert "MED" in tokens
    assert "|" in tokens
    assert "paracetamol" in tokens
    assert "650mg" in tokens
    assert "BD" in tokens
    assert "pc" in tokens
    assert "3d" in tokens

    encoded = tokenizer.encode(original_ir)
    decoded = tokenizer.decode(encoded)
    assert decoded.replace(" ", "") == original_ir.replace(" ", "")


def test_prepare_training_sample(tokenizer):
    shorthand = "Tab PCM 650mg BD"
    ir = "MED|paracetamol|650mg|BD|pc|3d"
    inp, tgt = tokenizer.prepare_training_sample(shorthand, ir, max_length=32)

    assert len(inp) == 32
    assert len(tgt) == 32
    assert inp[0] == tokenizer.BOS_TOKEN_ID
    # First few targets must be -100 (masked out prompt)
    assert tgt[0] == -100
    assert tgt[1] == -100
    # There should be non-masked tokens for the target IR
    non_masked = [t for t in tgt if t != -100]
    assert len(non_masked) > 0
