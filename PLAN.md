# PLAN.md: Step-by-Step Implementation Plan for Project 2

**Project:** Edge Clinical Shorthand-to-FHIR Compiler  
**Spec Reference:** [SPEC.md](file:///C:/Users/aksha/DSAI_Projects/Project2_Clinical_Shorthand_FHIR_Compiler/SPEC.md)  
**Target Environment:** Conda environment `umc301` (`conda run -n umc301 python ...`)  
**Status:** In Progress (Sprint 2)

---

## Phase 1: Specifications, Domain Data Generation & Formulary Curation

- [x] **Task 1.1: Project Environment & Dependency Specification (`requirements.txt`)**
  - Verify and specify required packages: `torch`, `numpy`, `pandas`, `jsonschema`, `streamlit`, `pytest`.
  - Ensure compatibility with `umc301` conda environment.

- [x] **Task 1.2: Clinical Formulary & Medical Abbreviations Setup (`data/raw/`)**
  - Build curated Indian National List of Essential Medicines (NLEM 2022) table with 50 primary-care drugs (`data/raw/nlem_formulary.json`), including RxNorm and SNOMED-CT codes.
  - Compile Latin medical abbreviations (`OD`, `BD`, `TID`, `QID`, `HS`, `SOS`, `STAT`, `pc`, `ac`, `wm`) and dosage patterns (`mg`, `mcg`, `g`, `ml`, `IU`).
  - Compile primary-care clinical conditions/diagnoses with SNOMED-CT mappings (`fever`, `hypertension`, `type 2 diabetes`, `urinary tract infection`, `gastroenteritis`, etc.).

- [x] **Task 1.3: Synthetic Clinical Shorthand Generator (`src/dataset_generator.py`)**
  - Synthesize $N = 10,000$ paired examples of realistic doctor shorthand notes $\leftrightarrow$ ground-truth compact Intermediate Representation (IR / DSL).
  - Include noise variations: casing jitter (`Tab PCM`, `tab pcm`), Latin abbreviation variations (`BD` vs `bid`), spacing variations, combined diagnosis and multi-drug prescriptions.
  - Split dataset into train (80%), validation (10%), and test (10%) splits with fixed random seed ($42$).
  - Write `tests/test_dataset.py` to verify grammar conformity of all generated DSL target strings.

---

## Phase 2: Scratch PyTorch Transformer Architecture (From First Principles)

- [x] **Task 2.1: Rotary Position Embeddings Module (`src/rope.py`)**
  - Implement 2D Givens rotation matrix computation: $\theta_i = 10000^{-2i/d_k}$.
  - Apply complex rotation to query and key head vectors.
  - Write `tests/test_rope.py` to mathematically verify relative position dot-product invariance: $\langle R_m q, R_n k \rangle = g(m-n)$.

- [x] **Task 2.2: RMSNorm & SwiGLU Activation Modules (`src/layers.py`)**
  - Implement `RMSNorm` with learnable scaling parameter $\gamma$ and $\epsilon = 10^{-6}$.
  - Implement `SwiGLU` MLP layer: $(x W_{\text{gate}} \cdot \text{SiLU}(x W_{\text{gate}})) \otimes (x W_{\text{up}}) W_{\text{down}}$.
  - Write unit tests verifying numerical stability and tensor dimensions.

- [x] **Task 2.3: Multi-Head Causal Self-Attention (`src/attention.py`)**
  - Implement custom causal self-attention with scaling factor $\frac{1}{\sqrt{d_k}}$.
  - Implement upper-triangular causal attention mask ($-\infty$ above diagonal).
  - Write `tests/test_attention.py` asserting that future token positions receive strictly zero attention weight.

- [x] **Task 2.4: Full Decoder Transformer Architecture (`src/transformer.py`)**
  - Assemble `TransformerBlock` and `ClinicalTransformerDecoder`:
    - Token embedding table ($|V| \times d_{\text{model}}$).
    - $N_{\text{layers}} = 4$ Pre-LN decoder blocks with RMSNorm, RoPE causal attention, and SwiGLU.
    - Final RMSNorm and linear projection head to vocabulary logits.
  - Support forward mode for parallel training (teacher forcing) and step mode for autoregressive decoding.

---

## Phase 3: Tokenizer, Training Pipeline & Convergence Verification

- [x] **Task 3.1: Clinical Domain Tokenizer (`src/tokenizer.py`)**
  - Implement domain character/subword tokenizer tuned for Latin shorthand and pipe-delimited IR.
  - Include special tokens: `<pad>`, `<bos>`, `<eos>`, `<sep>`, `<unk>`, `|`.
  - Save vocabulary mappings and test bidirectional encode/decode roundtrips.

- [x] **Task 3.2: Training Loop & Loss Engine (`src/train.py`)**
  - Implement causal cross-entropy loss with padding token masking.
  - Optimizer: AdamW ($\beta_1 = 0.9, \beta_2 = 0.98, \text{weight\_decay} = 0.01$).
  - Cosine learning rate schedule with linear warmup.
  - Gradient clipping ($\|\nabla\|_2 \le 1.0$).
  - Track train loss, validation loss, and string-level Exact Match (EM) accuracy per epoch (up to 15 epochs with patience $k = 3$ and target EM $\ge 98.0\%$).
  - Save best model checkpoints to `models/clinical_transformer_best.pt`.

---

## Phase 4: Circular KV-Cache & In-Engine DFA Logit-Masking

- [x] **Task 4.1: Contiguous Circular KV-Cache (`src/kv_cache.py` / `src/attention.py`)**
  - Implement pre-allocated tensor cache for key and value representations ($B \times H \times L_{\max} \times d_k$).
  - Implement single-step forward pass updating only slice index $t$.
  - Write `tests/test_attention.py` and `tests/test_model.py` verifying exact numerical parity between KV-cached autoregressive decoding and naive full-context recomputation ($\|y_{\text{cached}} - y_{\text{naive}}\|_{\infty} < 10^{-4}$).

- [x] **Task 4.2: In-Engine DFA Logit Masking Engine (`src/dfa_mask.py`)**
  - Implement 11-state Deterministic Finite Automaton tracking current parsing position (`STATE_RECORD_TYPE`, `STATE_MED_DRUG`, `STATE_MED_DOSE`, etc.).
  - Build state transition matrix mapping valid token IDs for each state.
  - Inject $-\infty$ logit mask at each decoding step to enforce zero syntactic violations.
  - Constrain drug names strictly to the 50 validated NLEM formulary entities.
  - Write `tests/test_dfa_mask.py` asserting that invalid transitions are rejected with $100\%$ precision.

---

## Phase 5: Stage 2 Deterministic Lowering Pass & HL7 FHIR Validation

- [x] **Task 5.1: AST Lowering Compiler (`src/fhir_lowering.py`)**
  - Ingest validated IR string and parse into AST data structures (`Prescription`, `MedicationItem`, `ConditionItem`).
  - Map Latin frequencies (`BD`, `TID`, `QID`, `HS`, `SOS`, `pc`, `ac`) to FHIR `dosageInstruction` structures.
  - Map NLEM drug names to official RxNorm and SNOMED-CT coding systems.
  - Generate full compliant HL7 FHIR R4 `Bundle` JSON containing `MedicationRequest` and `Condition` entries.

- [x] **Task 5.2: Programmatic FHIR Schema Validation Suite (`tests/test_fhir_lowering.py`)**
  - Load official HL7 FHIR R4 JSON schemas (`Bundle`, `MedicationRequest`, `Condition`).
  - Validate lowered JSON bundles programmatically using `jsonschema`.
  - Assert 100% schema compliance across all test cases.

---

## Phase 6: Benchmarking, Streamlit Clinical App & Viva Walkthrough

- [x] **Task 6.1: Latency & Memory Profiling Suite (`src/benchmark.py`)**
  - Measure per-token step latency on consumer CPU: verify $< 20\text{ ms}$ per step.
  - Benchmark KV-cached decoding vs. non-cached decoding across varying sequence lengths ($L \in \{8, 16, 24, 32, 48, 64\}$) to empirically demonstrate $O(L)$ vs. $O(L^2)$ scaling.
  - Save performance charts to `reports/kv_cache_latency_profile.png`.

- [x] **Task 6.2: Interactive Streamlit Clinical Prescriber (`app.py`)**
  - Real-time clinical transcription input with sample doctor shorthand presets.
  - Visual display of Stage 1 generated Intermediate Representation (with step latency).
  - Visual DFA state progression visualizer showing token transitions.
  - Interactive HL7 FHIR R4 JSON tree viewer with one-click download.
  - Schema validity badge and RxNorm / SNOMED-CT code inspector.

- [x] **Task 6.3: Comprehensive Viva Defense Walkthrough (`WALKTHROUGH.md`)**
  - Whiteboard derivations: Scaled dot-product variance $\frac{QK^T}{\sqrt{d_k}}$, RoPE Givens rotation matrix, $O(L)$ KV-cache step complexity.
  - Architectural justification: Two-stage compiler vs end-to-end LLM generation.
  - Active-recall viva defense simulation with 10 challenging oral exam questions and model answers.
