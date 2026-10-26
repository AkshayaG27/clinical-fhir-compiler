"""
Unit tests for Synthetic Clinical Shorthand Dataset and EBNF Grammar Conformance.
"""

import json
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SYNTHETIC_DATA_DIR = BASE_DIR / "data" / "synthetic"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


@pytest.fixture(scope="module")
def formulary_data():
    with open(RAW_DATA_DIR / "nlem_formulary.json", "r", encoding="utf-8") as f:
        formulary = json.load(f)
    with open(RAW_DATA_DIR / "abbreviations.json", "r", encoding="utf-8") as f:
        abbreviations = json.load(f)
    with open(RAW_DATA_DIR / "conditions.json", "r", encoding="utf-8") as f:
        conditions = json.load(f)
    return formulary, abbreviations, conditions


def test_dataset_splits_exist_and_counts():
    for split_name, expected_count in [("train", 8000), ("val", 1000), ("test", 1000)]:
        file_path = SYNTHETIC_DATA_DIR / f"{split_name}.json"
        assert file_path.exists(), f"Split file missing: {file_path}"
        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        assert len(records) == expected_count, f"{split_name} count mismatch: {len(records)} vs {expected_count}"


def test_ebnf_grammar_conformance(formulary_data):
    formulary, abbreviations, conditions = formulary_data
    valid_drugs = set(formulary.keys())
    valid_freqs = set(abbreviations["frequencies"].keys())
    valid_foods = set(abbreviations["food_relations"].keys())
    valid_conditions = {c["name"] for c in conditions.values()}
    valid_severities = {"mild", "moderate", "severe", "acute", "chronic"}

    for split_name in ["train", "val", "test"]:
        file_path = SYNTHETIC_DATA_DIR / f"{split_name}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for rec in records:
            assert "shorthand" in rec and len(rec["shorthand"]) > 0
            assert "target_ir" in rec and len(rec["target_ir"]) > 0

            ir_lines = rec["target_ir"].split("\n")
            assert len(ir_lines) in [1, 2], f"Invalid number of IR lines: {len(ir_lines)} in {rec['id']}"

            for line in ir_lines:
                parts = line.split("|")
                record_type = parts[0]
                assert record_type in ["MED", "COND"], f"Invalid record type: {record_type}"

                if record_type == "MED":
                    assert len(parts) == 6, f"MED record must have 6 fields, got {len(parts)} in {line}"
                    _, drug, dose, freq, food, dur = parts
                    assert drug in valid_drugs, f"Unknown drug '{drug}' in {line}"
                    assert freq in valid_freqs, f"Invalid freq '{freq}' in {line}"
                    assert food in valid_foods, f"Invalid food '{food}' in {line}"
                    assert dur.endswith("d") or dur.endswith("w")

                elif record_type == "COND":
                    assert len(parts) == 4, f"COND record must have 4 fields, got {len(parts)} in {line}"
                    _, cond_name, severity, dur = parts
                    assert cond_name in valid_conditions, f"Unknown condition '{cond_name}' in {line}"
                    assert severity in valid_severities, f"Invalid severity '{severity}' in {line}"
                    assert dur.endswith("d") or dur.endswith("w")
