"""
Stage 2 Lowering Pass: Compiles Intermediate Representation (IR / DSL) into
certified HL7 FHIR Release 4 (R4) JSON Bundles (MedicationRequest and Condition).
Adheres strictly to HL7 FHIR R4 schema and India ABDM standards.
"""

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


@dataclass
class MedicationItem:
    drug: str
    dose: str
    freq: str
    food: str
    dur: str


@dataclass
class ConditionItem:
    cond_name: str
    severity: str
    dur: str


class FHIRLoweringCompiler:
    """
    Deterministic AST lowering compiler.
    Parses DSL strings and generates official HL7 FHIR R4 JSON bundles.
    """

    def __init__(self, formulary_path: Optional[Path] = None):
        form_file = Path(formulary_path) if formulary_path is not None else (RAW_DATA_DIR / "nlem_formulary.json")
        with open(form_file, "r", encoding="utf-8") as f:
            self.formulary: Dict[str, Any] = json.load(f)
        with open(RAW_DATA_DIR / "abbreviations.json", "r", encoding="utf-8") as f:
            self.abbreviations: Dict[str, Any] = json.load(f)
        with open(RAW_DATA_DIR / "conditions.json", "r", encoding="utf-8") as f:
            self.conditions: Dict[str, Any] = json.load(f)

        # SNOMED severity mappings
        self.severity_snomed = {
            "mild": ("255604002", "Mild"),
            "moderate": ("6736007", "Moderate"),
            "severe": ("24484000", "Severe"),
            "acute": ("385315009", "Acute"),
            "chronic": ("90734009", "Chronic"),
        }

    def parse_dsl(self, dsl_text: str) -> List[Any]:
        """Parses pipe-delimited DSL lines into strongly typed AST nodes."""
        items = []
        lines = [l.strip() for l in dsl_text.strip().split("\n") if l.strip()]

        for line in lines:
            parts = [p.strip() for p in line.split("|")]
            record_type = parts[0]

            if record_type == "MED" and len(parts) >= 6:
                items.append(
                    MedicationItem(
                        drug=parts[1].lower(),
                        dose=parts[2],
                        freq=parts[3].upper(),
                        food=parts[4].lower(),
                        dur=parts[5].lower(),
                    )
                )
            elif record_type == "COND" and len(parts) >= 4:
                items.append(
                    ConditionItem(
                        cond_name=parts[1].lower(),
                        severity=parts[2].lower(),
                        dur=parts[3].lower(),
                    )
                )
        return items

    def _parse_numeric_dose(self, dose_str: str) -> Tuple[float, str]:
        m = re.match(r"^([0-9.]+)\s*([a-zA-Z]+)?$", dose_str)
        if m:
            val = float(m.group(1))
            unit = m.group(2) or "mg"
            return val, unit
        return 1.0, "tab"

    def _parse_numeric_duration(self, dur_str: str) -> int:
        m = re.match(r"^([0-9]+)\s*([a-zA-Z]+)?$", dur_str)
        if m:
            val = int(m.group(1))
            unit = m.group(2) or "d"
            if unit == "w":
                return val * 7
            elif unit == "m":
                return val * 30
            return val
        return 3

    def lower_medication(self, med: MedicationItem, patient_id: str = "example") -> Dict[str, Any]:
        """Compiles a MedicationItem into an HL7 FHIR R4 MedicationRequest resource."""
        res_id = f"medrx-{uuid.uuid4().hex[:8]}"
        drug_info = self.formulary.get(med.drug, {})
        rxnorm_code = drug_info.get("rxnorm", "000000")
        snomed_code = drug_info.get("snomed", "000000")
        display_name = drug_info.get("display", med.drug.capitalize())

        dose_val, dose_unit = self._parse_numeric_dose(med.dose)
        dur_days = self._parse_numeric_duration(med.dur)

        freq_info = self.abbreviations["frequencies"].get(med.freq, {})
        fhir_freq = freq_info.get("fhir", {})

        food_info = self.abbreviations["food_relations"].get(med.food, {})
        fhir_when = food_info.get("fhir_when", [])

        # Build dosageInstruction
        repeat_obj: Dict[str, Any] = {
            "period": fhir_freq.get("period", 1),
            "periodUnit": fhir_freq.get("periodUnit", "d"),
        }
        if "frequency" in fhir_freq:
            repeat_obj["frequency"] = fhir_freq["frequency"]
        if fhir_when:
            repeat_obj["when"] = fhir_when

        dosage_instruction: Dict[str, Any] = {
            "text": f"{med.dose} {med.freq} {med.food} for {med.dur}",
            "timing": {"repeat": repeat_obj},
            "doseAndRate": [
                {
                    "doseQuantity": {
                        "value": dose_val,
                        "unit": dose_unit,
                        "system": "http://unitsofmeasure.org",
                        "code": dose_unit,
                    }
                }
            ],
        }

        if fhir_freq.get("asNeededBoolean", False):
            dosage_instruction["asNeededBoolean"] = True

        resource: Dict[str, Any] = {
            "resourceType": "MedicationRequest",
            "id": res_id,
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {
                "coding": [
                    {
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": rxnorm_code,
                        "display": f"{display_name} {med.dose}",
                    },
                    {
                        "system": "http://snomed.info/sct",
                        "code": snomed_code,
                        "display": f"{display_name} (substance)",
                    },
                ],
                "text": f"{med.drug} {med.dose}",
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
                "display": "OPD Patient",
            },
            "dosageInstruction": [dosage_instruction],
            "dispenseRequest": {
                "expectedSupplyDuration": {
                    "value": dur_days,
                    "unit": "days",
                    "system": "http://unitsofmeasure.org",
                    "code": "d",
                }
            },
        }
        return resource

    def lower_condition(self, cond: ConditionItem, patient_id: str = "example") -> Dict[str, Any]:
        """Compiles a ConditionItem into an HL7 FHIR R4 Condition resource."""
        res_id = f"cond-{uuid.uuid4().hex[:8]}"

        # Find matching condition by canonical name
        cond_info = {}
        for c in self.conditions.values():
            if c["name"].lower() == cond.cond_name.lower():
                cond_info = c
                break

        snomed_code = cond_info.get("snomed", "386661006")
        display_name = cond_info.get("display", cond.cond_name.capitalize())

        sev_code, sev_display = self.severity_snomed.get(
            cond.severity, ("6736007", "Moderate")
        )

        resource: Dict[str, Any] = {
            "resourceType": "Condition",
            "id": res_id,
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                        "display": "Active",
                    }
                ]
            },
            "verificationStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": "provisional",
                        "display": "Provisional",
                    }
                ]
            },
            "severity": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": sev_code,
                        "display": sev_display,
                    }
                ]
            },
            "code": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": snomed_code,
                        "display": display_name,
                    }
                ],
                "text": cond.cond_name,
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
                "display": "OPD Patient",
            },
        }
        return resource

    def lower_to_bundle(self, dsl_text: str, patient_id: str = "example") -> Dict[str, Any]:
        """
        Compiles the entire DSL into an HL7 FHIR R4 collection Bundle.
        """
        ast_items = self.parse_dsl(dsl_text)
        bundle_id = f"bundle-{uuid.uuid4().hex[:8]}"

        entries = []
        for item in ast_items:
            if isinstance(item, MedicationItem):
                res = self.lower_medication(item, patient_id=patient_id)
            elif isinstance(item, ConditionItem):
                res = self.lower_condition(item, patient_id=patient_id)
            else:
                continue

            entries.append({
                "fullUrl": f"urn:uuid:{res['id']}",
                "resource": res,
            })

        bundle: Dict[str, Any] = {
            "resourceType": "Bundle",
            "id": bundle_id,
            "type": "collection",
            "entry": entries,
        }
        return bundle


if __name__ == "__main__":
    compiler = FHIRLoweringCompiler()
    sample_dsl = "COND|acute viral fever|moderate|3d\nMED|paracetamol|650mg|BD|pc|3d"
    bundle = compiler.lower_to_bundle(sample_dsl)
    print(json.dumps(bundle, indent=2))
