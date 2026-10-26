"""
Clinical Shorthand Dataset Generator
Synthesizes paired examples: Doctor Shorthand <-> Compact Intermediate Representation (IR / DSL)
Adheres to Chapter 13 Spec-Driven Development and NLEM 2022 Formulary.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
SYNTHETIC_DATA_DIR = BASE_DIR / "data" / "synthetic"


def load_formulary() -> Tuple[Dict, Dict, Dict]:
    with open(RAW_DATA_DIR / "nlem_formulary.json", "r", encoding="utf-8") as f:
        formulary = json.load(f)
    with open(RAW_DATA_DIR / "abbreviations.json", "r", encoding="utf-8") as f:
        abbreviations = json.load(f)
    with open(RAW_DATA_DIR / "conditions.json", "r", encoding="utf-8") as f:
        conditions = json.load(f)
    return formulary, abbreviations, conditions


FORMULATION_PREFIXES = [
    "Tab", "Tab.", "tab", "Cap", "Cap.", "cap", "T.", "C.", "Syp", "Syp.", "", ""
]

DURATION_SHORTHAND_TEMPLATES = [
    "x {val}d", "x {val} days", "for {val} days", "{val}/7", "x {val} day", "{val}d"
]

DURATION_WEEKS_TEMPLATES = [
    "x {val}w", "x {val} weeks", "for {val} wks", "{val}/52", "{val}w"
]

CONDITION_PREFIXES = [
    "c/o", "C/O", "k/c/o", "K/C/O", "complains of", "known case of", "diag:", "dx:"
]


def format_dose(raw_dose: str) -> str:
    """Standardizes dosage string to e.g. 500mg, 10mg."""
    m = re.match(r"^([0-9.]+)\s*([a-zA-Z]+)$", raw_dose)
    if m:
        val, unit = m.groups()
        return f"{val}{unit.lower()}"
    return raw_dose


def generate_single_medication(formulary: Dict, abbreviations: Dict) -> Tuple[str, str]:
    """Generates a single paired (shorthand, IR) medication record."""
    drug_key = random.choice(list(formulary.keys()))
    drug_info = formulary[drug_key]

    # Pick generic or alias name
    if random.random() < 0.65 and drug_info["aliases"]:
        drug_name_used = random.choice(drug_info["aliases"])
    else:
        drug_name_used = drug_info["generic"]

    prefix = random.choice(FORMULATION_PREFIXES)

    # Pick dose
    dose_str = random.choice(drug_info["default_doses"])
    canonical_dose = format_dose(dose_str)
    # Shorthand might have space: "650 mg" or "650mg"
    shorthand_dose = canonical_dose if random.random() < 0.5 else canonical_dose.replace("mg", " mg")

    # Pick frequency
    freq_key = random.choice(list(abbreviations["frequencies"].keys()))
    freq_info = abbreviations["frequencies"][freq_key]
    shorthand_freq = random.choice(freq_info["aliases"])

    # Pick food relation
    food_key = random.choice(list(abbreviations["food_relations"].keys()))
    food_info = abbreviations["food_relations"][food_key]
    shorthand_food = random.choice(food_info["aliases"])

    # Pick duration
    dur_val = random.choice([1, 2, 3, 5, 7, 10, 14])
    dur_unit = "d"
    canonical_dur = f"{dur_val}{dur_unit}"
    shorthand_dur = random.choice(DURATION_SHORTHAND_TEMPLATES).format(val=dur_val)

    # Assemble Shorthand string
    parts = []
    if prefix:
        parts.append(f"{prefix} {drug_name_used}")
    else:
        parts.append(drug_name_used)

    parts.append(shorthand_dose)
    parts.append(shorthand_freq)
    if shorthand_food:
        parts.append(shorthand_food)
    parts.append(shorthand_dur)

    shorthand_str = " ".join(parts)
    ir_str = f"MED|{drug_key}|{canonical_dose}|{freq_key}|{food_key}|{canonical_dur}"
    return shorthand_str, ir_str


def generate_single_condition(conditions: Dict) -> Tuple[str, str]:
    """Generates a single paired (shorthand, IR) condition record."""
    cond_key = random.choice(list(conditions.keys()))
    cond_info = conditions[cond_key]

    c_prefix = random.choice(CONDITION_PREFIXES)
    cond_alias = random.choice(cond_info["aliases"])

    dur_val = random.choice([1, 2, 3, 5, 7])
    canonical_dur = f"{dur_val}d"
    shorthand_dur = random.choice(DURATION_SHORTHAND_TEMPLATES).format(val=dur_val)

    severity = cond_info["default_severity"]
    canonical_cond_name = cond_info["name"]

    shorthand_str = f"{c_prefix} {cond_alias} {shorthand_dur}"
    ir_str = f"COND|{canonical_cond_name}|{severity}|{canonical_dur}"
    return shorthand_str, ir_str


def generate_paired_sample(formulary: Dict, abbreviations: Dict, conditions: Dict) -> Tuple[str, str]:
    """
    Generates a realistic clinical encounter sample.
    Can be:
      - 1 medication
      - 1 condition + 1 medication
      - 2 medications
    """
    sample_type = random.choices(["med_only", "cond_and_med", "two_meds"], weights=[0.45, 0.40, 0.15])[0]

    if sample_type == "med_only":
        return generate_single_medication(formulary, abbreviations)

    elif sample_type == "cond_and_med":
        c_short, c_ir = generate_single_condition(conditions)
        m_short, m_ir = generate_single_medication(formulary, abbreviations)
        # Separator in shorthand can be comma or newline or semicolon
        sep = random.choice([", ", "; ", " | ", "\n"])
        return f"{c_short}{sep}{m_short}", f"{c_ir}\n{m_ir}"

    else:  # two_meds
        m1_short, m1_ir = generate_single_medication(formulary, abbreviations)
        m2_short, m2_ir = generate_single_medication(formulary, abbreviations)
        sep = random.choice([", ", "; ", " + ", "\n"])
        return f"{m1_short}{sep}{m2_short}", f"{m1_ir}\n{m2_ir}"


def build_dataset(total_samples: int = 10000, seed: int = 42) -> Dict[str, List[Dict]]:
    random.seed(seed)
    formulary, abbreviations, conditions = load_formulary()

    samples = []
    seen_shorthands = set()

    for idx in range(total_samples):
        shorthand, ir = generate_paired_sample(formulary, abbreviations, conditions)
        # Add basic jitter
        if random.random() < 0.2:
            shorthand = shorthand.lower()
        samples.append({
            "id": f"rec_{idx:05d}",
            "shorthand": shorthand,
            "target_ir": ir
        })

    # Shuffle and split 80% train, 10% val, 10% test
    random.shuffle(samples)
    n_train = int(total_samples * 0.8)
    n_val = int(total_samples * 0.1)

    train_data = samples[:n_train]
    val_data = samples[n_train:n_train + n_val]
    test_data = samples[n_train + n_val:]

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }


def save_dataset(dataset_splits: Dict[str, List[Dict]]):
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, records in dataset_splits.items():
        out_path = SYNTHETIC_DATA_DIR / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"Saved {len(records)} records to {out_path}")


if __name__ == "__main__":
    splits = build_dataset(total_samples=10000, seed=42)
    save_dataset(splits)
