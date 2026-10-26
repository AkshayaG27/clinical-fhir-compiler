"""
Unit tests for In-Engine DFA Logit Masking.
Verifies strictly regular state transitions and NLEM drug restrictions.
"""

from pathlib import Path
import pytest
import torch
from src.tokenizer import ClinicalTokenizer
from src.dfa_mask import ClinicalDFAMaskEngine, DFAState

BASE_DIR = Path(__file__).resolve().parent.parent
VOCAB_PATH = BASE_DIR / "data" / "tokenizer" / "vocab.json"


@pytest.fixture
def dfa_engine():
    tokenizer = ClinicalTokenizer(VOCAB_PATH)
    return ClinicalDFAMaskEngine(tokenizer)


def test_dfa_initial_state_allows_only_record_types(dfa_engine):
    prompt_tokens = [dfa_engine.tokenizer.BOS_TOKEN_ID, 15, 20, dfa_engine.sep_id]
    mask = dfa_engine.build_logit_mask(prompt_tokens)

    # Only MED and COND should have 0.0, rest -inf
    med_id = dfa_engine.tokenizer.token2id["MED"]
    cond_id = dfa_engine.tokenizer.token2id["COND"]

    assert mask[med_id].item() == 0.0
    assert mask[cond_id].item() == 0.0

    # Arbitrary other token must be -inf
    random_id = dfa_engine.tokenizer.token2id["paracetamol"]
    assert mask[random_id].item() == float("-inf")


def test_dfa_drug_state_restricts_strictly_to_nlem(dfa_engine):
    # After MED | -> must be NLEM drug
    seq = [
        dfa_engine.tokenizer.BOS_TOKEN_ID,
        15,
        dfa_engine.sep_id,
        dfa_engine.tokenizer.token2id["MED"],
        dfa_engine.pipe_id,
    ]
    mask = dfa_engine.build_logit_mask(seq)

    # Paracetamol is an NLEM drug
    pcm_id = dfa_engine.tokenizer.token2id["paracetamol"]
    assert mask[pcm_id].item() == 0.0

    # Amoxicillin is an NLEM drug
    amox_id = dfa_engine.tokenizer.token2id["amoxicillin"]
    assert mask[amox_id].item() == 0.0

    # Non-drug tokens (like dose '650mg' or freq 'BD' or delimiter '|') must be -inf
    dose_id = dfa_engine.tokenizer.token2id["650mg"]
    freq_id = dfa_engine.tokenizer.token2id["BD"]
    pipe_id = dfa_engine.pipe_id

    assert mask[dose_id].item() == float("-inf")
    assert mask[freq_id].item() == float("-inf")
    assert mask[pipe_id].item() == float("-inf")


def test_dfa_record_end_allows_only_eos_or_newline(dfa_engine):
    # Complete MED record: MED|paracetamol|650mg|BD|pc|3d
    toks = [
        dfa_engine.tokenizer.BOS_TOKEN_ID,
        15,
        dfa_engine.sep_id,
        dfa_engine.tokenizer.token2id["MED"],
        dfa_engine.pipe_id,
        dfa_engine.tokenizer.token2id["paracetamol"],
        dfa_engine.pipe_id,
        dfa_engine.tokenizer.token2id["650mg"],
        dfa_engine.pipe_id,
        dfa_engine.tokenizer.token2id["BD"],
        dfa_engine.pipe_id,
        dfa_engine.tokenizer.token2id["pc"],
        dfa_engine.pipe_id,
        dfa_engine.tokenizer.token2id["3d"],
    ]
    mask = dfa_engine.build_logit_mask(toks)

    eos_id = dfa_engine.eos_id
    newline_id = dfa_engine.newline_id

    assert mask[eos_id].item() == 0.0
    assert mask[newline_id].item() == 0.0

    # All other tokens must be -inf
    med_id = dfa_engine.tokenizer.token2id["MED"]
    assert mask[med_id].item() == float("-inf")
