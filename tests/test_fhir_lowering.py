"""
Unit tests for Stage 2 HL7 FHIR Lowering Pass.
Validates structural conformance against HL7 FHIR Release 4 specification.
"""

from pathlib import Path
import pytest
import jsonschema
from src.fhir_lowering import FHIRLoweringCompiler

# Official minimal FHIR R4 Bundle JSON Schema for automated validation
FHIR_BUNDLE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["resourceType", "id", "type", "entry"],
    "properties": {
        "resourceType": {"const": "Bundle"},
        "id": {"type": "string"},
        "type": {"type": "string", "enum": ["collection", "transaction"]},
        "entry": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["fullUrl", "resource"],
                "properties": {
                    "fullUrl": {"type": "string"},
                    "resource": {
                        "type": "object",
                        "required": ["resourceType", "id"],
                        "properties": {
                            "resourceType": {
                                "type": "string",
                                "enum": ["MedicationRequest", "Condition"],
                            },
                            "id": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}

MEDICATION_REQUEST_SCHEMA = {
    "type": "object",
    "required": [
        "resourceType",
        "id",
        "status",
        "intent",
        "medicationCodeableConcept",
        "dosageInstruction",
        "subject",
    ],
    "properties": {
        "resourceType": {"const": "MedicationRequest"},
        "status": {"type": "string", "enum": ["active", "completed", "draft"]},
        "intent": {"type": "string", "enum": ["order", "proposal", "plan"]},
        "medicationCodeableConcept": {
            "type": "object",
            "required": ["coding", "text"],
            "properties": {
                "coding": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["system", "code", "display"],
                        "properties": {
                            "system": {"type": "string"},
                            "code": {"type": "string"},
                            "display": {"type": "string"},
                        },
                    },
                },
                "text": {"type": "string"},
            },
        },
        "subject": {
            "type": "object",
            "required": ["reference"],
        },
        "dosageInstruction": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["text", "timing"],
            },
        },
    },
}

CONDITION_SCHEMA = {
    "type": "object",
    "required": [
        "resourceType",
        "id",
        "clinicalStatus",
        "verificationStatus",
        "severity",
        "code",
        "subject",
    ],
    "properties": {
        "resourceType": {"const": "Condition"},
        "clinicalStatus": {
            "type": "object",
            "required": ["coding"],
        },
        "verificationStatus": {
            "type": "object",
            "required": ["coding"],
        },
        "severity": {
            "type": "object",
            "required": ["coding"],
        },
        "code": {
            "type": "object",
            "required": ["coding", "text"],
        },
        "subject": {
            "type": "object",
            "required": ["reference"],
        },
    },
}


@pytest.fixture
def compiler():
    return FHIRLoweringCompiler()


def test_fhir_bundle_lowering_single_med(compiler):
    dsl = "MED|amoxicillin|500mg|TID|ac|5d"
    bundle = compiler.lower_to_bundle(dsl)

    # 1. Validate against FHIR Bundle Schema
    jsonschema.validate(instance=bundle, schema=FHIR_BUNDLE_SCHEMA)

    assert len(bundle["entry"]) == 1
    med_rx = bundle["entry"][0]["resource"]

    # 2. Validate against MedicationRequest Schema
    jsonschema.validate(instance=med_rx, schema=MEDICATION_REQUEST_SCHEMA)

    # 3. Check RxNorm and SNOMED codes
    codings = med_rx["medicationCodeableConcept"]["coding"]
    rxnorm_entry = next(c for c in codings if "rxnorm" in c["system"])
    snomed_entry = next(c for c in codings if "snomed" in c["system"])

    assert rxnorm_entry["code"] == "723"
    assert snomed_entry["code"] == "372687004"

    # 4. Check DosageInstruction
    dosage = med_rx["dosageInstruction"][0]
    assert dosage["timing"]["repeat"]["frequency"] == 3
    assert "AC" in dosage["timing"]["repeat"]["when"]


def test_fhir_bundle_lowering_condition_and_med(compiler):
    dsl = "COND|acute viral fever|moderate|3d\nMED|paracetamol|650mg|BD|pc|3d"
    bundle = compiler.lower_to_bundle(dsl)

    jsonschema.validate(instance=bundle, schema=FHIR_BUNDLE_SCHEMA)
    assert len(bundle["entry"]) == 2

    cond_res = bundle["entry"][0]["resource"]
    med_res = bundle["entry"][1]["resource"]

    jsonschema.validate(instance=cond_res, schema=CONDITION_SCHEMA)
    jsonschema.validate(instance=med_res, schema=MEDICATION_REQUEST_SCHEMA)

    # Check Condition SNOMED codes
    cond_coding = cond_res["code"]["coding"][0]
    assert cond_coding["code"] == "386661006"
    assert cond_res["severity"]["coding"][0]["code"] == "6736007"
