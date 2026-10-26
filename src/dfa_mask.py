"""
In-Engine Deterministic Finite Automaton (DFA) Logit Masking Engine.
Enforces strictly regular grammar transitions and constrains drug names
strictly to the Indian National List of Essential Medicines (NLEM 2022).
"""

import json
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Set, Optional

import torch
from src.tokenizer import ClinicalTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
VOCAB_PATH = BASE_DIR / "data" / "tokenizer" / "vocab.json"


class DFAState(Enum):
    START = auto()
    STATE_RECORD_TYPE = auto()
    STATE_MED_DRUG = auto()
    STATE_MED_DOSE = auto()
    STATE_MED_FREQ = auto()
    STATE_MED_FOOD = auto()
    STATE_MED_DUR = auto()
    STATE_COND_NAME = auto()
    STATE_COND_SEV = auto()
    STATE_COND_DUR = auto()
    STATE_RECORD_END = auto()


class ClinicalDFAMaskEngine:
    """
    Finite State Machine that tracks generated DSL tokens and emits
    a logit mask (-inf for invalid transitions) at each decoding step.
    """

    def __init__(self, tokenizer: ClinicalTokenizer, formulary_path: Optional[Path] = None):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        # Load formulary, abbreviations, conditions
        form_file = Path(formulary_path) if formulary_path is not None else (RAW_DATA_DIR / "nlem_formulary.json")
        with open(form_file, "r", encoding="utf-8") as f:
            formulary = json.load(f)
        with open(RAW_DATA_DIR / "abbreviations.json", "r", encoding="utf-8") as f:
            abbreviations = json.load(f)
        with open(RAW_DATA_DIR / "conditions.json", "r", encoding="utf-8") as f:
            conditions = json.load(f)

        # Precompute valid token IDs for each entity category
        self.pipe_id = tokenizer.token2id.get("|", -1)
        self.newline_id = tokenizer.token2id.get("\n", -1)
        self.eos_id = tokenizer.EOS_TOKEN_ID
        self.sep_id = tokenizer.SEP_TOKEN_ID

        # Record types
        self.med_type_id = tokenizer.token2id.get("MED", -1)
        self.cond_type_id = tokenizer.token2id.get("COND", -1)

        # Valid 50 NLEM Drugs
        self.valid_drug_ids = {
            tokenizer.token2id[d]
            for d in formulary.keys()
            if d in tokenizer.token2id
        }

        # Valid Frequencies
        self.valid_freq_ids = {
            tokenizer.token2id[freq]
            for freq in abbreviations["frequencies"].keys()
            if freq in tokenizer.token2id
        }

        # Valid Food relations
        self.valid_food_ids = {
            tokenizer.token2id[f]
            for f in abbreviations["food_relations"].keys()
            if f in tokenizer.token2id
        }

        # Valid Condition names
        self.valid_cond_ids = {
            tokenizer.token2id[c["name"]]
            for c in conditions.values()
            if c["name"] in tokenizer.token2id
        }

        # Valid Severities
        self.valid_sev_ids = {
            tokenizer.token2id[s]
            for s in ["mild", "moderate", "severe", "acute", "chronic"]
            if s in tokenizer.token2id
        }

        # Dosages & Durations
        self.valid_dose_ids = {
            idx for tok, idx in tokenizer.token2id.items()
            if any(tok.endswith(u) for u in ["mg", "mcg", "g", "ml", "iu"]) or tok.isdigit()
        }
        self.valid_dur_ids = {
            idx for tok, idx in tokenizer.token2id.items()
            if tok.endswith("d") or tok.endswith("w") or tok.endswith("m") or tok.isdigit()
        }

    def get_current_state(self, generated_target_tokens: List[int]) -> DFAState:
        """
        Infers the current DFA state from the sequence of generated target tokens
        (tokens generated after the <sep> token).
        """
        if not generated_target_tokens:
            return DFAState.START

        # Split tokens into lines by newline
        lines: List[List[int]] = [[]]
        for t in generated_target_tokens:
            if t == self.newline_id:
                lines.append([])
            else:
                lines[-1].append(t)

        current_line = lines[-1]
        if not current_line:
            # Just saw a newline -> expecting next record type
            return DFAState.STATE_RECORD_TYPE

        first_tok = current_line[0]

        if first_tok == self.med_type_id:
            # MED record structure:
            # MED | drug | dose | freq | food | dur
            # Count how many pipe '|' symbols have appeared so far
            pipe_count = current_line.count(self.pipe_id)
            last_tok = current_line[-1]

            if pipe_count == 0:
                # Saw "MED", next MUST be pipe '|'
                return DFAState.STATE_RECORD_TYPE
            elif pipe_count == 1:
                # Saw "MED |" -> expecting drug name
                return DFAState.STATE_MED_DRUG if last_tok == self.pipe_id else DFAState.STATE_RECORD_TYPE
            elif pipe_count == 2:
                # Saw "MED | drug |" -> expecting dose
                return DFAState.STATE_MED_DOSE if last_tok == self.pipe_id else DFAState.STATE_RECORD_TYPE
            elif pipe_count == 3:
                # Saw "MED | drug | dose |" -> expecting freq
                return DFAState.STATE_MED_FREQ if last_tok == self.pipe_id else DFAState.STATE_RECORD_TYPE
            elif pipe_count == 4:
                # Saw "MED | drug | dose | freq |" -> expecting food
                return DFAState.STATE_MED_FOOD if last_tok == self.pipe_id else DFAState.STATE_RECORD_TYPE
            elif pipe_count == 5:
                # Saw "MED | drug | dose | freq | food |"
                # If last token is pipe, expecting duration. If last token is duration, record is complete!
                return DFAState.STATE_MED_DUR if last_tok == self.pipe_id else DFAState.STATE_RECORD_END
            else:
                return DFAState.STATE_RECORD_END

        elif first_tok == self.cond_type_id:
            # COND record structure:
            # COND | cond_name | severity | dur (3 pipes total)
            pipe_count = current_line.count(self.pipe_id)
            last_tok = current_line[-1]

            if pipe_count == 0:
                return DFAState.STATE_RECORD_TYPE
            elif pipe_count == 1:
                return DFAState.STATE_COND_NAME if last_tok == self.pipe_id else DFAState.STATE_RECORD_TYPE
            elif pipe_count == 2:
                return DFAState.STATE_COND_SEV if last_tok == self.pipe_id else DFAState.STATE_RECORD_TYPE
            elif pipe_count == 3:
                return DFAState.STATE_COND_DUR if last_tok == self.pipe_id else DFAState.STATE_RECORD_END
            else:
                return DFAState.STATE_RECORD_END

        return DFAState.STATE_RECORD_TYPE

    def get_allowed_tokens(self, generated_tokens: List[int]) -> Set[int]:
        """Returns set of allowed token IDs for the next step."""
        # Find where <sep> token is located
        if self.sep_id in generated_tokens:
            sep_idx = generated_tokens.index(self.sep_id)
            target_tokens = generated_tokens[sep_idx + 1 :]
        else:
            target_tokens = []

        if not target_tokens:
            # First token of generation MUST be MED or COND
            return {self.med_type_id, self.cond_type_id}

        state = self.get_current_state(target_tokens)
        last_tok = target_tokens[-1]

        if state == DFAState.START or state == DFAState.STATE_RECORD_TYPE:
            if last_tok in [self.med_type_id, self.cond_type_id]:
                # Immediately after MED or COND, must emit pipe '|'
                return {self.pipe_id}
            elif last_tok in self.valid_drug_ids or last_tok in self.valid_cond_ids:
                return {self.pipe_id}
            elif last_tok in self.valid_dose_ids or last_tok in self.valid_sev_ids:
                return {self.pipe_id}
            elif last_tok in self.valid_freq_ids:
                return {self.pipe_id}
            elif last_tok in self.valid_food_ids:
                return {self.pipe_id}
            elif last_tok == self.newline_id:
                # Next line must start with MED or COND
                return {self.med_type_id, self.cond_type_id}
            else:
                return {self.med_type_id, self.cond_type_id, self.pipe_id}

        elif state == DFAState.STATE_MED_DRUG:
            return self.valid_drug_ids

        elif state == DFAState.STATE_MED_DOSE:
            return self.valid_dose_ids

        elif state == DFAState.STATE_MED_FREQ:
            return self.valid_freq_ids

        elif state == DFAState.STATE_MED_FOOD:
            return self.valid_food_ids

        elif state == DFAState.STATE_MED_DUR:
            return self.valid_dur_ids

        elif state == DFAState.STATE_COND_NAME:
            return self.valid_cond_ids

        elif state == DFAState.STATE_COND_SEV:
            return self.valid_sev_ids

        elif state == DFAState.STATE_COND_DUR:
            return self.valid_dur_ids

        elif state == DFAState.STATE_RECORD_END:
            # Can terminate with EOS or start new line with \n
            return {self.eos_id, self.newline_id}

        return {self.eos_id}

    def build_logit_mask(self, full_sequence: List[int]) -> torch.Tensor:
        """
        Creates a (vocab_size,) tensor mask.
        Allowed tokens have 0.0, illegal tokens have -inf.
        """
        allowed_ids = self.get_allowed_tokens(full_sequence)
        mask = torch.full((self.vocab_size,), float("-inf"))
        for tid in allowed_ids:
            if 0 <= tid < self.vocab_size:
                mask[tid] = 0.0
        return mask
