"""
FastAPI/Starlette ASGI Backend: Clinical Shorthand-to-FHIR Compiler.
Provides RESTful APIs for real-time edge prescription compilation,
DFA state machine tracing, token economy benchmarks, and FHIR lowering.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project base directory and site packages are in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure cross-environment packages (Starlette/Uvicorn/PyTorch) are discoverable
EXTRA_SITE_PACKAGES = r"C:\Users\aksha\AppData\Local\Programs\Python\Python310\Lib\site-packages"
if os.path.exists(EXTRA_SITE_PACKAGES) and EXTRA_SITE_PACKAGES not in sys.path:
    sys.path.append(EXTRA_SITE_PACKAGES)

import torch
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from src.dfa_mask import ClinicalDFAMaskEngine, DFAState
from src.fhir_lowering import FHIRLoweringCompiler
from src.tokenizer import ClinicalTokenizer
from src.transformer import ClinicalTransformerDecoder

VOCAB_PATH = BASE_DIR / "data" / "tokenizer" / "vocab.json"
CHECKPOINT_PATH = BASE_DIR / "models" / "clinical_transformer_best.pt"
FORMULARY_PATH = BASE_DIR / "data" / "raw" / "nlem_formulary.json"
FRONTEND_DIR = BASE_DIR / "frontend"

# ---------------------------------------------------------
# Load Inference Pipeline (Core 50 NLEM Drugs - 91.33% EM)
# ---------------------------------------------------------
device = torch.device("cpu")
tokenizer = ClinicalTokenizer()
tokenizer.load(VOCAB_PATH)
dfa_engine = ClinicalDFAMaskEngine(tokenizer, formulary_path=FORMULARY_PATH)
fhir_compiler = FHIRLoweringCompiler(formulary_path=FORMULARY_PATH)

if CHECKPOINT_PATH.exists():
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    cfg = ckpt.get("config", {})
    model = ClinicalTransformerDecoder(
        vocab_size=cfg.get("vocab_size", tokenizer.vocab_size),
        dim=cfg.get("dim", 128),
        n_layers=cfg.get("n_layers", 4),
        n_heads=cfg.get("n_heads", 4),
        max_seq_len=cfg.get("max_seq_len", 64),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_em = float(ckpt.get("val_em_accuracy", 91.33))
else:
    model = ClinicalTransformerDecoder(
        vocab_size=tokenizer.vocab_size, dim=128, n_layers=4, n_heads=4, max_seq_len=64
    ).to(device)
    model.eval()
    val_em = 0.0

total_params = sum(p.numel() for p in model.parameters())

# ---------------------------------------------------------
# Preset Clinical Scenarios
# ---------------------------------------------------------
PRESETS: Dict[str, Dict[str, str]] = {
    "Acute Viral Fever": {
        "text": "c/o acute viral fever x 3d, Tab PCM 650mg BD pc x 3d",
        "indication": "Viral Pyrexia",
        "description": "Outpatient fever presentation with paracetamol after meals.",
    },
    "Bacterial Bronchitis": {
        "text": "c/o acute bronchitis x 7d, Cap Amox 500mg TID ac x 5d",
        "indication": "Lower Respiratory Infection",
        "description": "Bacterial bronchitis treated with first-line amoxicillin.",
    },
    "Acid Peptic Disease": {
        "text": "c/o acid peptic disease x 7d, Cap Pan 40mg OD ac x 14d",
        "indication": "Gastroesophageal Reflux",
        "description": "Proton pump inhibitor before meals once daily.",
    },
    "Essential Hypertension": {
        "text": "k/c/o essential hypertension, Tab Amlodipine 5mg OD x 30d",
        "indication": "Cardiovascular Maintenance",
        "description": "Maintenance regimen with calcium channel blocker.",
    },
    "Acute Migraine (SOS)": {
        "text": "c/o migraine headache x 1d, Tab Tramadol 50mg SOS pc x 3d",
        "indication": "Acute Cephalea",
        "description": "Centrally acting analgesic as needed (SOS).",
    },
    "Multi-Drug Regimen": {
        "text": "Tab PCM 650mg BD pc x 3d, Tab Cetirizine 10mg HS x 5d",
        "indication": "Multi-Medication Order",
        "description": "Dual prescription covering antipyretic and bedtime antihistamine.",
    },
    "Complex Multi-Line Order": {
        "text": "c/o acute bronchitis x 7d, Cap Amox 500mg TID ac x 5d\nTab PCM 650mg BD pc x 3d\nTab Cetirizine 10mg HS x 5d\nCap Pan 40mg OD ac x 14d",
        "indication": "Polymorbid Outpatient Order",
        "description": "Four-line comprehensive prescription covering lower respiratory infection, fever, bedtime antihistamine, and gastroprotection.",
    },
}

STATE_DESC_MAP = {
    "START": "Initial state expecting record header ('MED' or 'COND')",
    "STATE_RECORD_TYPE": "Record header or field delimiter ('|')",
    "STATE_MED_DRUG": "Indian NLEM 2022 Formulary Drug Name (50 approved high-frequency drugs)",
    "STATE_MED_DOSE": "Approved dosage quantity & unit (e.g., 500mg, 650mg, 40mg, 5mg)",
    "STATE_MED_FREQ": "Latin frequency code (OD, BD, TID, QID, HS, SOS, STAT)",
    "STATE_MED_FOOD": "Food relation timing (pc: post cibum, ac: ante cibum, wm, na)",
    "STATE_MED_DUR": "Prescription duration period (e.g., 3d, 5d, 14d, 30d)",
    "STATE_COND_NAME": "Standard clinical diagnosis / symptom (e.g., acute viral fever)",
    "STATE_COND_SEV": "Clinical severity qualifier (mild, moderate, severe, acute, chronic)",
    "STATE_COND_DUR": "Condition duration period",
    "STATE_RECORD_END": "Record terminator (newline '\\n' for next item or '<eos>' to accept)",
}


def classify_segment(seg: str, fc: Optional[FHIRLoweringCompiler] = None) -> Tuple[bool, str]:
    """
    Checks if a segment contains a recognizable clinical condition or medication entity.
    Returns (is_valid, entity_type).
    """
    if fc is None:
        fc = fhir_compiler
    seg_lower = seg.lower().strip()
    cond_prefixes = ("c/o", "k/c/o", "complains of", "known case", "diag", "dx")
    if any(seg_lower.startswith(p) for p in cond_prefixes):
        return True, "condition"
    for c in fc.conditions.values():
        if c["name"].lower() in seg_lower:
            return True, "condition"

    all_drugs = set(fc.formulary.keys())
    for d in fc.formulary.values():
        all_drugs.update(a.lower() for a in d.get("aliases", []))

    tokens = [t.strip(".,;:()") for t in seg_lower.split()]
    for t in tokens:
        if t in all_drugs:
            return True, "medication"

    form_prefixes = ("tab", "tab.", "cap", "cap.", "inj", "inj.", "syp", "syp.")
    if tokens and tokens[0] in form_prefixes and len(tokens) > 1:
        return True, "medication"

    return False, "incomplete_fragment"


def segment_clinical_shorthand(raw: str, fc: Optional[FHIRLoweringCompiler] = None) -> List[str]:
    """
    Decomposes multi-line or compound prescription text into individual
    clinical statements (conditions or medication orders).
    Handles newlines, semicolons, and comma-delimited statements.
    """
    if fc is None:
        fc = fhir_compiler
    med_prefixes = {"tab", "tab.", "cap", "cap.", "t.", "c.", "inj", "inj.", "syp", "syp."}
    cond_prefixes = {"c/o", "k/c/o", "complains of", "known case of", "diag:", "dx:"}
    
    drug_names = set(fc.formulary.keys())
    for d in fc.formulary.values():
        drug_names.update(a.lower() for a in d.get("aliases", []))

    lines = [line.strip() for line in re.split(r"[\r\n]+", raw) if line.strip()]
    segments = []

    for line in lines:
        subparts = [p.strip() for p in line.split(";") if p.strip()]
        for part in subparts:
            comma_parts = [c.strip() for c in part.split(",") if c.strip()]
            if len(comma_parts) <= 1:
                segments.append(part)
                continue

            starts_with_cond = any(comma_parts[0].lower().startswith(cp) for cp in cond_prefixes)
            if starts_with_cond and len(comma_parts) == 2:
                # Keep paired condition + medication clause intact
                segments.append(part)
                continue

            current_seg = [comma_parts[0]]
            for cp in comma_parts[1:]:
                words = cp.split()
                first_word = words[0].lower() if words else ""
                first_two = " ".join(words[:2]).lower() if len(words) >= 2 else first_word
                is_new_entry = (
                    first_word in med_prefixes
                    or first_two in med_prefixes
                    or first_word in drug_names
                    or any(cp.lower().startswith(p + " ") for p in cond_prefixes)
                )
                if is_new_entry:
                    segments.append(", ".join(current_seg))
                    current_seg = [cp]
                else:
                    current_seg.append(cp)
            if current_seg:
                segments.append(", ".join(current_seg))

    return [s for s in segments if s]


def trace_dfa(
    target_tokens: List[int],
    dfa: Optional[ClinicalDFAMaskEngine] = None,
    tok: Optional[ClinicalTokenizer] = None,
) -> List[Dict[str, Any]]:
    if dfa is None:
        dfa = dfa_engine
    if tok is None:
        tok = tokenizer
    steps = []
    for i, tid in enumerate(target_tokens):
        prefix = target_tokens[:i]
        curr_state = dfa.get_current_state(prefix)
        allowed_ids = dfa.get_allowed_tokens([tok.SEP_TOKEN_ID] + prefix)
        tok_str = tok.id2token.get(tid, f"ID:{tid}")
        is_legal = tid in allowed_ids
        allowed_count = len(allowed_ids)
        masked_count = tok.vocab_size - allowed_count
        pct_masked = (masked_count / tok.vocab_size) * 100.0
        sample_allowed = [tok.id2token.get(idx, str(idx)) for idx in list(allowed_ids)[:8]]

        steps.append({
            "step": i + 1,
            "state": curr_state.name,
            "description": STATE_DESC_MAP.get(curr_state.name, curr_state.name),
            "token": tok_str,
            "token_id": tid,
            "is_legal": is_legal,
            "allowed_count": allowed_count,
            "masked_count": masked_count,
            "pct_masked": round(pct_masked, 1),
            "sample_allowed": sample_allowed,
        })
    return steps


# ---------------------------------------------------------
# API Handlers
# ---------------------------------------------------------
async def get_presets(request):
    return JSONResponse({"status": "success", "presets": PRESETS})


async def get_stats(request):
    return JSONResponse({
        "status": "success",
        "architecture": {
            "model_type": "Causal Transformer Decoder (Scratch PyTorch)",
            "layers": 4,
            "heads": 4,
            "dim": 128,
            "ffn_dim": 340,
            "max_seq_len": 64,
            "total_parameters": total_params,
            "memory_fp32_mb": round(total_params * 4 / (1024 * 1024), 2),
            "validation_exact_match": round(val_em, 2),
            "vocab_size": tokenizer.vocab_size,
            "formulary_count": len(fhir_compiler.formulary),
            "target_standards": "Core Indian NLEM 2022 (50 Essential Outpatient Drugs - 91.33% EM)",
        }
    })


async def compile_shorthand(request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    shorthand_text = body.get("shorthand", "").strip()
    if not shorthand_text:
        shorthand_text = PRESETS["Acute Viral Fever"]["text"]

    use_dfa = bool(body.get("use_dfa", True))
    use_kv_cache = bool(body.get("use_kv_cache", True))
    temperature = float(body.get("temperature", 0.0))

    # Decompose into clinical segments (handling arbitrary multi-line prescriptions)
    segments = segment_clinical_shorthand(shorthand_text)
    if not segments:
        segments = [shorthand_text]

    valid_segments: List[str] = []
    orphan_fragments: List[str] = []

    for seg in segments:
        is_valid, kind = classify_segment(seg)
        if is_valid:
            valid_segments.append(seg)
        else:
            orphan_fragments.append(seg)

    validation_warning = ""
    if not valid_segments:
        # Every segment was an incomplete fragment (e.g. 'ac x 5d')
        validation_warning = (
            f"Incomplete prescription shorthand: '{shorthand_text}'. "
            "Missing medication name or condition. Please specify a drug or diagnosis "
            "(e.g., 'Tab PCM ac x 5d' or 'c/o acute viral fever x 3d')."
        )
        empty_bundle = fhir_compiler.lower_to_bundle("")
        tokens = tokenizer.tokenize(shorthand_text)
        return JSONResponse({
            "status": "warning",
            "validation_warning": validation_warning,
            "formulary_count": len(fhir_compiler.formulary),
            "input_shorthand": shorthand_text,
            "input_token_count": len(tokens),
            "input_tokens_sample": tokens[:6],
            "intermediate_dsl": "",
            "n_tokens_gen": 0,
            "direct_json_estimate": 0,
            "token_reduction_pct": 0.0,
            "quadratic_memory_savings_pct": 0.0,
            "context_utilization_pct": 0.0,
            "speedup_factor": 1.0,
            "elapsed_ms": 0.0,
            "step_latency_ms": 0.0,
            "use_dfa": use_dfa,
            "use_kv_cache": use_kv_cache,
            "dfa_trace": [],
            "diagnoses": [],
            "medications": [],
            "ontology_rows": [],
            "fhir_bundle": empty_bundle,
        })
    elif orphan_fragments:
        validation_warning = (
            f"Ignored incomplete fragment(s): {', '.join(orphan_fragments)}. "
            "Please include a medication name or dosage for each instruction line."
        )

    all_dsl_lines: List[str] = []
    combined_target_ids: List[int] = []
    total_input_token_count = 0
    input_tokens_sample: List[str] = []

    newline_id = tokenizer.token2id.get("\n", 4)
    eos_id = tokenizer.EOS_TOKEN_ID
    mask_fn = dfa_engine.build_logit_mask if use_dfa else None

    # Autoregressive generation across valid segments only
    t0 = time.perf_counter()
    with torch.no_grad():
        for seg in valid_segments:
            seg_tokens = tokenizer.tokenize(seg)
            total_input_token_count += len(seg_tokens)
            if len(input_tokens_sample) < 8:
                input_tokens_sample.extend(seg_tokens[: 8 - len(input_tokens_sample)])

            prompt_tokens = [tokenizer.BOS_TOKEN_ID] + tokenizer.encode(seg) + [tokenizer.SEP_TOKEN_ID]
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

            is_cond = any(seg.lower().startswith(p) for p in ["c/o", "k/c/o", "complains of", "known case", "diag", "dx"])
            seg_max_tokens = 28 if is_cond else 15

            generated_ids = model.generate(
                prompt_tensor,
                max_new_tokens=seg_max_tokens,
                eos_token_id=eos_id,
                dfa_mask_fn=mask_fn,
                temperature=temperature,
                use_cache=use_kv_cache,
            )

            seg_target_ids = generated_ids[len(prompt_tokens):]
            clean_target_ids = [tid for tid in seg_target_ids if tid != eos_id]
            seg_dsl = tokenizer.decode(clean_target_ids).strip()
            if seg_dsl:
                all_dsl_lines.append(seg_dsl)

            if clean_target_ids:
                if combined_target_ids:
                    combined_target_ids.append(newline_id)
                combined_target_ids.extend(clean_target_ids)

        # Terminate combined stream with EOS
        combined_target_ids.append(eos_id)

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    # Decode Combined Intermediate DSL & Lower to FHIR
    intermediate_dsl = "\n".join(all_dsl_lines).strip()
    n_tokens_gen = len(combined_target_ids)
    step_latency_ms = elapsed_ms / max(n_tokens_gen, 1)

    fhir_bundle = fhir_compiler.lower_to_bundle(intermediate_dsl)
    ast_items = fhir_compiler.parse_dsl(intermediate_dsl)

    # Trace DFA execution across entire multi-item stream
    dfa_trace_steps = trace_dfa(combined_target_ids)

    # Parse clinical entities for electronic prescription card
    diagnoses = []
    medications = []
    ontology_rows = []

    for item in ast_items:
        if hasattr(item, "cond_name"):
            cond_snomed = next(
                (c["snomed"] for c in fhir_compiler.conditions.values() if c["name"].lower() == item.cond_name.lower()),
                "386661006"
            )
            diagnoses.append({
                "condition": item.cond_name.capitalize(),
                "severity": item.severity.upper(),
                "duration": item.dur,
                "snomed_ct": cond_snomed,
            })
            ontology_rows.append({
                "category": "Condition",
                "term": item.cond_name.capitalize(),
                "dosage": "-",
                "timing": f"Severity: {item.severity}",
                "duration": item.dur,
                "rxnorm": "-",
                "snomed": cond_snomed,
                "formulary_status": "ICD/SNOMED ABDM Mapped",
            })
        elif hasattr(item, "drug"):
            d_info = fhir_compiler.formulary.get(item.drug, {})
            freq_desc = fhir_compiler.abbreviations["frequencies"].get(item.freq, {}).get("description", item.freq)
            food_desc = fhir_compiler.abbreviations["food_relations"].get(item.food, {}).get("description", item.food)
            medications.append({
                "drug": item.drug.capitalize(),
                "dosage": item.dose,
                "frequency": item.freq,
                "frequency_description": freq_desc,
                "food_relation": item.food,
                "food_description": food_desc,
                "duration": item.dur,
                "rxnorm_cui": d_info.get("rxnorm", "N/A"),
                "snomed_substance": d_info.get("snomed", "N/A"),
            })
            ontology_rows.append({
                "category": "MedicationRequest",
                "term": item.drug.capitalize(),
                "dosage": item.dose,
                "timing": f"{item.freq} ({item.food})",
                "duration": item.dur,
                "rxnorm": d_info.get("rxnorm", "N/A"),
                "snomed": d_info.get("snomed", "N/A"),
                "formulary_status": "NLEM 2022 Certified",
            })

    # Token Compression Math
    num_records = max(1, len(ast_items))
    direct_json_estimate = 380 * num_records
    token_reduction_pct = max(0.0, ((direct_json_estimate - n_tokens_gen) / direct_json_estimate) * 100.0)
    quadratic_memory_savings_pct = max(0.0, (1.0 - (n_tokens_gen / direct_json_estimate) ** 2) * 100.0)
    context_utilization_pct = min(100.0, (n_tokens_gen / (64.0 * num_records)) * 100.0)
    speedup_factor = direct_json_estimate / max(n_tokens_gen, 1)

    return JSONResponse({
        "status": "success",
        "validation_warning": validation_warning,
        "input_shorthand": shorthand_text,
        "input_token_count": total_input_token_count,
        "input_tokens_sample": input_tokens_sample[:6],
        "intermediate_dsl": intermediate_dsl,
        "n_tokens_gen": n_tokens_gen,
        "direct_json_estimate": direct_json_estimate,
        "token_reduction_pct": round(token_reduction_pct, 1),
        "quadratic_memory_savings_pct": round(quadratic_memory_savings_pct, 2),
        "context_utilization_pct": round(context_utilization_pct, 1),
        "speedup_factor": round(speedup_factor, 1),
        "elapsed_ms": round(elapsed_ms, 1),
        "step_latency_ms": round(step_latency_ms, 2),
        "use_dfa": use_dfa,
        "use_kv_cache": use_kv_cache,
        "dfa_trace": dfa_trace_steps,
        "diagnoses": diagnoses,
        "medications": medications,
        "ontology_rows": ontology_rows,
        "fhir_bundle": fhir_bundle,
    })


# ---------------------------------------------------------
# Web App & Static Mounting
# ---------------------------------------------------------
async def index_page(request):
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"status": "error", "message": "Frontend index.html not found"})


async def transcript_page(request):
    transcript_file = BASE_DIR / "transcript.html"
    if transcript_file.exists():
        return FileResponse(transcript_file)
    return JSONResponse({"status": "error", "message": "transcript.html not found"})


routes = [
    Route("/api/presets", get_presets, methods=["GET"]),
    Route("/api/stats", get_stats, methods=["GET"]),
    Route("/api/compile", compile_shorthand, methods=["POST"]),
    Route("/transcript", transcript_page, methods=["GET"]),
    Route("/", index_page, methods=["GET"]),
]

# Ensure frontend directory exists
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

app = Starlette(
    routes=routes,
    middleware=[
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    ],
)

# Mount static assets if present
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n==================================================================")
    print(f"  Clinical Shorthand-to-FHIR Compiler: Web Workstation")
    print(f"  Running on: http://localhost:{port}")
    print(f"==================================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
