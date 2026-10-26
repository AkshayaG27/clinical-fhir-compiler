# WALKTHROUGH.md: Viva Defense & First-Principles Guide
## Project 2: Edge Clinical Shorthand-to-FHIR Compiler

**Course:** Applied Data Science and Artificial Intelligence (UMC301)  
**Institution:** Indian Institute of Science (IISc), Computational and Data Sciences (CDS)  
**Target:** A+ Grade (Flawless First-Principles Whiteboard Defense)

---

## 1. Executive Summary & Architecture Blueprint

Project 2 implements an edge-efficient compiler that ingests terse Latin outpatient prescription shorthand (e.g., `Tab PCM 650mg BD pc x 3d, c/o acute fever x 2d`) and outputs fully compliant **HL7 FHIR Release 4 (R4)** JSON bundles (`MedicationRequest` and `Condition`).

```mermaid
flowchart LR
    Shorthand["Clinician Shorthand<br>'Tab PCM 650mg BD pc x 3d'"] --> ScratchTransformer["Stage 1: Scratch PyTorch Transformer<br>(Causal MHA + RoPE + Circular KV-Cache)"]
    ScratchTransformer --> DFAMask["In-Engine DFA Grammar Mask<br>(Enforces NLEM 2022 Formulary)"]
    DFAMask --> CompactIR["Compact Intermediate Representation (IR)<br>MED|paracetamol|650mg|BD|pc|3d<br>(15-25 tokens, L <= 64)"]
    CompactIR --> LoweringPass["Stage 2: Deterministic Python Lowering Pass<br>(AST Schema Generator)"]
    LoweringPass --> HAPIFHIR["Certified HL7 FHIR R4 Bundle<br>(Validated via jsonschema / FHIR Schema)"]
```

### Why the Two-Stage Architecture?
1. **The Sequence Bloat Problem:** Generating complete HL7 FHIR JSON directly requires **350–500+ tokens** per prescription. On edge hardware with limited memory, generating 500 tokens per prescription creates massive latency ($>15\text{s}$).
2. **The Grammar Complexity Problem:** JSON is a **Context-Free Grammar (CFG)** with recursive braces and brackets, requiring a pushdown stack automaton for constrained generation.
3. **The Solution:** Our Stage 1 model generates a compact pipe-delimited **Intermediate Representation (IR)** spanning only **15–25 tokens** ($L \le 64$). Because the IR is strictly linear, it forms a **regular grammar** governed by an 11-state **Deterministic Finite Automaton (DFA)**. Stage 2 deterministically lowers the verified IR to certified HL7 FHIR JSON with $100\%$ schema pass rate.

---

## 2. First-Principles Mathematical Derivations (Whiteboard Checklist)

### 2.1 Derivation 1: Scaled Dot-Product Attention Variance Normalization
**Question:** *Why do we divide $Q K^T$ by $\sqrt{d_k}$ in scaled dot-product attention?*

**Proof:**
Let $q, k \in \mathbb{R}^{d_k}$ be independent query and key vectors whose components are independent and identically distributed with zero mean and unit variance:
$$\mathbb{E}[q_i] = 0, \quad \text{Var}(q_i) = 1, \quad \mathbb{E}[k_i] = 0, \quad \text{Var}(k_i) = 1$$
The dot product is $S = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$.
1. **Expectation:**
   $$\mathbb{E}[S] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$$
2. **Variance:**
   $$\text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] - 0 = 1 \times 1 = 1$$
   By independence of components:
   $$\text{Var}(S) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$
3. **Consequence of Unscaled Dot Products:**
   When $d_k$ is large (e.g., $d_k = 32$ or $64$), the variance of the dot product is $d_k$, meaning standard deviation is $\sigma = \sqrt{d_k}$. Large magnitude inputs push the $\text{softmax}(z)$ function into its saturated regions where the gradient $\frac{\partial \text{softmax}(z)_i}{\partial z_j} \approx 0$, causing **gradient vanishing**.
4. **Rescaling:**
   Dividing by $\sqrt{d_k}$ rescales the variance back to $1.0$:
   $$\text{Var}\left( \frac{q \cdot k}{\sqrt{d_k}} \right) = \frac{1}{d_k} \text{Var}(q \cdot k) = \frac{d_k}{d_k} = 1.0$$
   This keeps softmax inputs in the active linear gradient regime.

---

### 2.2 Derivation 2: Rotary Position Embeddings (RoPE) Relative Distance Invariance
**Question:** *Prove that RoPE preserves relative distance $\langle R_m q, R_n k \rangle = g(q, k, m - n)$.*

**Proof:**
For a 2D subspace of query $q$ and key $k$, RoPE defines the Givens rotation matrix for position $m$:
$$R_{\theta, m} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$$
where $\theta_i = 10000^{-2i/d}$.
The inner product between rotated vectors $\tilde{q}_m = R_{\theta, m} q$ and $\tilde{k}_n = R_{\theta, n} k$ is:
$$\langle \tilde{q}_m, \tilde{k}_n \rangle = \tilde{q}_m^T \tilde{k}_n = q^T R_{\theta, m}^T R_{\theta, n} k$$
Multiplying the rotation matrices:
$$R_{\theta, m}^T R_{\theta, n} = \begin{pmatrix} \cos(m\theta) & \sin(m\theta) \\ -\sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} \cos(n\theta) & -\sin(n\theta) \\ \sin(n\theta) & \cos(n\theta) \end{pmatrix}$$
Using standard trigonometric angle-subtraction identities:
$$\cos(m\theta)\cos(n\theta) + \sin(m\theta)\sin(n\theta) = \cos((m - n)\theta)$$
$$-\cos(m\theta)\sin(n\theta) + \sin(m\theta)\cos(n\theta) = \sin((n - m)\theta) = -\sin((m - n)\theta)$$
Thus:
$$R_{\theta, m}^T R_{\theta, n} = \begin{pmatrix} \cos((m - n)\theta) & -\sin((m - n)\theta) \\ \sin((m - n)\theta) & \cos((m - n)\theta) \end{pmatrix} = R_{\theta, m - n}$$
Therefore:
$$\langle \tilde{q}_m, \tilde{k}_n \rangle = q^T R_{\theta, m - n} k = g(q, k, m - n)$$
**Conclusion:** The attention score depends strictly on the relative displacement $(m - n)$, guaranteeing shift invariance with zero learned parameters.

---

### 2.3 Derivation 3: Circular KV-Cache Step Complexity $\mathcal{O}(L)$ vs $\mathcal{O}(L^2)$
**Question:** *What is the computational complexity of autoregressive generation with vs without KV-cache?*

**Derivation:**
Suppose we generate a sequence of length $L$.
* **Without KV-Cache (Naive Recomputation):**
  At step $t$, the model processes all $t$ previous tokens from scratch.
  Computing self-attention across $t$ tokens requires $t \times t = t^2$ operations.
  Total operations across the entire generation:
  $$\sum_{t=1}^L \mathcal{O}(t \cdot d_{\text{model}}) = \mathcal{O}\left( \frac{L(L+1)}{2} d_{\text{model}} \right) = \mathbf{\mathcal{O}(L^2 d_{\text{model}})}$$
* **With Circular KV-Cache:**
  Keys and values from steps $1 \dots t-1$ are stored in contiguous memory buffers: $K_{\text{cache}}, V_{\text{cache}} \in \mathbb{R}^{B \times H \times L_{\max} \times d_k}$.
  At step $t$, only the single new token $x_t$ is projected into query $q_t$, key $k_t$, and value $v_t$.
  $k_t$ and $v_t$ are written inplace to the cache at slice $t$.
  Query $q_t$ ($1 \times d_k$) attends to the cached keys $K[:, :, :t+1, :]$ ($t \times d_k$).
  This requires strictly $t$ vector dot products:
  $$\text{Per-Step Complexity} = \mathbf{\mathcal{O}(t \cdot d_k) = \mathcal{O}(L \cdot d)}$$
  Per-token latency remains strictly bounded ($\sim 3.8\text{ ms}$ on CPU).

---

## 3. Empirical Results & Performance Benchmarks

### 3.1 Model Convergence
* **Architecture:** 4 Layers, 4 Attention Heads, $d_{\text{model}} = 128$, $d_{\text{ffn}} = 340$, SwiGLU, RMSNorm.
* **Parameter Count:** 851,712 parameters ($\sim 3.25\text{ MB}$ FP32 memory).
* **Training Protocol:** Up to 15 epochs (cosine learning rate annealing, early-stopping patience $k = 3$, target EM $\ge 98.0\%$). Initial 8-epoch baseline completed in 6.32 minutes on consumer CPU.
* **Loss Progression:**
  * Epoch 1: Train Loss 2.7404 | Val Loss 0.9603 | Val EM: 0.00%
  * Epoch 4: Train Loss 0.1855 | Val Loss 0.0769 | Val EM: 58.00%
  * Epoch 6: Train Loss 0.0090 | Val Loss 0.0067 | Val EM: 88.67%
  * Epoch 8: Train Loss 0.0016 | Val Loss 0.0032 | Val EM: **91.33%**
* **With DFA Logit Masking:** **100.0% Syntactically Valid IR tuples**.

### 3.2 Step Latency Profile (Consumer CPU)
Measured via `src/benchmark.py`:
* Sequence Length $L = 8$: **3.77 ms** (Speedup: $1.99\times$)
* Sequence Length $L = 16$: **3.79 ms** (Speedup: $1.31\times$)
* Sequence Length $L = 32$: **4.27 ms** (Speedup: $2.06\times$)
* Sequence Length $L = 64$: **4.81 ms** (Speedup: $1.29\times$)
* **All steps comfortably exceed the $< 20\text{ ms}$ sub-second edge target!**

---

## 4. Viva Defense: 10 Active-Recall Interview Questions & Answers

### Q1: Why did you choose a Decoder-Only architecture instead of an Encoder-Decoder (T5)?
**Answer:** In an Encoder-Decoder, parameter capacity is split between an encoder and a decoder, and inference requires managing two distinct KV-caches (encoder cross-attention and decoder self-attention). A Decoder-Only architecture unifies prompt and completion into a single causal sequence, utilizing 100% of parameter capacity during autoregressive generation, requiring only a single circular buffer, and allowing seamless in-engine DFA logit masking.

### Q2: Why is causal attention strictly mandatory for a decoder-only model?
**Answer:** A decoder-only model factorizes joint probability via the chain rule: $P(X) = \prod_{t=1}^T P(x_t \mid x_{<t})$. Causal attention masks future tokens ($M_{i, j} = -\infty$ for $j > i$) to prevent information leakage from future tokens during parallel teacher-forcing training.

### Q3: What is the difference between Teacher Forcing and free autoregressive generation?
**Answer:** In free generation, each step receives the model's own predicted token. In Teacher Forcing, the input at step $t$ is always the true ground-truth token from the dataset, regardless of what the model predicted. This prevents cascading error accumulation and enables parallel matrix multiplication across all 64 sequence tokens during training.

### Q4: Why did you set prompt loss targets to `-100` during training?
**Answer:** `-100` is PyTorch's `ignore_index` for `CrossEntropyLoss`. Our model's task is strictly to compile shorthand into the IR, not to predict doctor writing styles. Masking the prompt prevents wasting model capacity and gradient updates on prompt tokens.

### Q5: How does Rotary Position Embedding (RoPE) differ from absolute positional embeddings?
**Answer:** Absolute embeddings (GPT-2) add static position vectors $x_m + p_m$, which fail to generalize if a word pair appears at a novel offset. RoPE rotates query and key vectors in 2D planes using Givens rotation matrices. The resulting inner product depends purely on relative distance $(m - n)$, ensuring that the attention bond between a drug and its dose is identical anywhere in the prescription.

### Q6: Why did you use RMSNorm instead of standard LayerNorm?
**Answer:** LayerNorm centers activations by subtracting the mean and dividing by standard deviation (requiring two memory passes). Zhang & Sennrich (2019) demonstrated that mean-centering provides no training stability benefit. RMSNorm normalizes strictly by root-mean-square in a single memory pass, reducing memory bandwidth by $\sim 7\%$ and speeding up CPU inference.

### Q7: Why SwiGLU instead of standard ReLU / GELU?
**Answer:** Standard MLPs apply a static non-linear activation. SwiGLU uses a bilinear multiplicative gate: $(\text{SiLU}(x W_{\text{gate}}) \otimes x W_{\text{up}}) W_{\text{down}}$. The gating branch acts as an adaptive dimmer switch, dynamically filtering and routing features.

### Q8: Why did you choose context length $L = 64$?
**Answer:** Empirical token distribution shows that 100% of OPD clinical encounters (including polypharmacy and diagnoses) span $\le 54$ tokens. $L = 64$ is the minimal power-of-two that avoids truncation while allowing the entire $64 \times 64$ attention matrix ($16\text{ KB}$) to fit inside the CPU's L1 cache, preventing RAM bandwidth bottlenecks.

### Q9: What happens if an unapproved drug is entered into the model?
**Answer:** The in-engine DFA logit mask monitors the decoding state. When in `STATE_MED_DRUG`, all token logits outside the 50 validated NLEM formulary entities are set to $-\infty$, mathematically guaranteeing that hallucinated chemical names have zero probability of being generated.

### Q10: How are the lowered resources verified against healthcare standards?
**Answer:** Stage 2 parses the validated DSL into an official HL7 FHIR R4 collection `Bundle` containing `MedicationRequest` and `Condition` resources. Each drug is mapped to its RxNorm CUI and SNOMED-CT substance ID, and conditions are mapped to SNOMED-CT disorder concepts (mandated by ABDM). The generated JSON passes programmatic `jsonschema` validation against official HL7 FHIR schemas with a 100% pass rate.
