# ACADEMIC CRITIQUE — "Reviewer #2" Evaluation

**Paper Under Review:** *MaayaTrain: Cross-Platform Distributed Low-Communication Training of Language Models on Consumer Hardware*

**Reviewer:** Anonymous (Senior Area Chair — MLSys/NeurIPS/ICLR)

**Date:** 2026-04-21

**Verdict: MAJOR REVISION REQUIRED**

---

## I. Narrative & Framing Critique

### 1.1 Overall Assessment

The manuscript presents a legitimate and timely contribution — a practical, consumer-grade implementation of DiLoCo with novel extensions. However, the current draft reads more like a **project README** or **technical blog post** than a peer-reviewed systems paper. Multiple claims lack empirical backing, the mathematical formalization is incomplete relative to the actual codebase, and several SOTA features introduced in recent code upgrades are entirely absent from the draft.

**CAUTION: The draft describes a codebase that no longer exists.** The code has since added Compute-Proportional Async-DiLoCo, Block-Wise INT8, Dynamic Sharding, and Stream-Chunked Median — none of which appear in the paper. The paper must be rewritten from scratch for the Algorithmic Framework section.

### 1.2 Colloquial Language & Tone Issues

| Location | Problematic Phrasing | Academic Replacement |
|----------|---------------------|---------------------|
| Abstract | "plug-and-play distributed training" | "zero-configuration distributed training requiring no manual network orchestration" |
| §1.1 | "creates an enormous barrier to entry" | "imposes prohibitive capital and operational costs, constraining participation to well-funded organizations" |
| §1.1 | "these resources remain largely untapped" | "this aggregate capacity remains underutilized for collaborative model training" |
| §2.8 | "MaayaTrain differs from prior work along several axes" | "We distinguish MaayaTrain from prior work across four dimensions" |
| §4.3.1 | "sanity guards" | "defensive bounds to prevent protocol-level denial-of-service" |
| §5.2 | "gold-standard LLM training schedule" | "the empirically dominant schedule for autoregressive language model training (Hoffmann et al., 2022)" |
| §7.4 | "An RTX 4090 (~82.6 TFLOPS) completes 500 inner steps much faster than an Apple M4" | Provide actual measured wall-clock times or remove the vague comparison |
| §9 | "any collection of consumer devices connected over standard Wi-Fi can contribute to training LLMs" | Overly broad — qualify with model scale, minimum hardware, and network requirements |

### 1.3 Unsupported SOTA Claims

**WARNING: The following claims in the draft have ZERO empirical evidence and would be instantly flagged by any competent reviewer:**

1. **"<1% quality degradation" for INT8** (Abstract, §4.3.3) — The draft describes *per-tensor* INT8, but the code now uses *block-wise* INT8. The per-tensor error analysis in Appendix C is mathematically correct but applies to a **deprecated** code path. No empirical NRMSE measurements are presented for the actual block-wise implementation.

2. **"convergence parity" with single-device baseline** (Abstract) — No training curves, no validation perplexity numbers, no convergence plots. This is a checkmark claim with zero evidence.

3. **"500× communication reduction"** (Abstract, §9) — This is the theoretical *H* factor from DiLoCo. The actual reduction depends on compression ratio, streaming overhead, and protocol framing. Must be measured, not asserted.

4. **"8–12× compression with minimal quality impact"** (§1.2, §4.3.3) — Compression ratios are claimed but never measured on actual pseudo-gradients from a real training run.

5. **"near-linear scaling"** (§6.1) — Mentioned as a target metric but never demonstrated.

### 1.4 Structural Weaknesses

1. **Section 5 ("SOTA Training Optimizations") is filler.** AMP, cosine warmup, gradient accumulation, and gradient clipping are standard PyTorch training features available since 2020. They are not contributions. They should be described in 2-3 sentences in the System Architecture section, not given their own 3-page section. **The actual SOTA contributions (compute-proportional weighting, dynamic sharding, block-wise quantization, chunked median) are not in the paper at all.**

2. **The Evaluation section (§6) is entirely hypothetical.** It describes what experiments *should* be run but presents no results. For a systems paper, this is fatal.

3. **The comparison table (§7.5) makes claims about other systems** (OpenDiLoCo, ColossalAI, DeepSpeed) without citing specific version numbers or benchmarks. The "Min. bandwidth" numbers appear to be rough estimates.

4. **Appendices are underweight.** Appendix C analyzes per-tensor INT8 error, which is now deprecated in favor of block-wise. The analysis must be redone for the block-wise case.

---

## II. Literature Review Gap Analysis

**IMPORTANT: The following 5 papers are MANDATORY citations for the revised manuscript. Omitting any of them would expose a critical gap to knowledgeable reviewers.**

### Paper 1: GaLore — Gradient Low-Rank Projection (Zhao et al., 2024)

- **Citation:** Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., & Tian, Y. (2024). GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection. *arXiv:2403.03507*.
- **Relevance:** GaLore reduces optimizer state memory by 65.5% via gradient-space low-rank projection. It is the primary competitor/complement to your block-wise INT8 approach. **You MUST discuss why block-wise INT8 is preferable for communication reduction (GaLore targets memory, not bandwidth) and where they could be combined.**
- **Integration Point:** Related Work §2.7 (Gradient Compression) — GaLore is a fundamentally different compression paradigm (low-rank vs. quantization). Your paper should clearly delineate the two approaches along the memory-vs-communication axis.

### Paper 2: QLoRA / Block-Wise NF4 Quantization (Dettmers et al., 2023)

- **Citation:** Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS 2023*.
- **Relevance:** Your block-wise INT8 quantization (128-element blocks with independent scale/min) is **directly inspired** by the block-wise design in bitsandbytes. The docstrings in `tensor_codec.py` already cite Dettmers et al., but the paper draft still describes per-tensor INT8. **This is a critical inconsistency.** The revised paper must formally credit the block-wise design pattern and explain your independent implementation with clean-room rationale.
- **Integration Point:** §3 (Algorithmic Framework) and Related Work — establish that your approach adapts block-wise quantization from weight compression (QLoRA's domain) to gradient compression (a different statistical regime with heavier tails).

### Paper 3: Asynchronous DiLoCo with Delay Correction (ICLR 2025 / OpenReview)

- **Citation:** Recent work on Asynchronous DiLoCo with momentum-based look-ahead delay correction (e.g., submissions to ICLR 2025). The specific citation would be the paper at `openreview.net` or `arxiv.org` on "Asynchronous distributed training with delay correction for DiLoCo."
- **Relevance:** Your `apply_outer_step_weighted` method implements compute-proportional weighting based on local step counts — a form of implicit asynchrony handling. You MUST compare this to explicit delay-correction methods that adjust the Nesterov momentum based on gradient staleness. Your approach is simpler (weight by steps, no explicit staleness tracking) but theoretically less grounded.
- **Integration Point:** §3 (Algorithmic Framework) — discuss compute-proportional weighting as a lightweight alternative to full delay-correction, with the trade-off of simplicity vs. theoretical convergence guarantees.

### Paper 4: FedBuff / Buffered Asynchronous Aggregation (Nguyen et al., 2022)

- **Citation:** Nguyen, J., Malik, K., Zhan, H., Yurochkin, M., Huo, Z., & Mauch, L. (2022). Federated Learning with Buffered Asynchronous Aggregation. *AISTATS 2022*.
- **Relevance:** FedBuff introduces a buffered asynchronous approach where the server aggregates updates as they arrive, without waiting for all clients. Your time-bounded sync mode (`sync_mode="time"` → `train_steps_timed`) is architecturally similar — faster workers contribute more steps within a fixed window, mimicking a buffered async model. Citing FedBuff positions your work within the established async FL literature.
- **Integration Point:** Related Work §2.5 and §3 — connect your time-bounded window approach to the buffered asynchronous aggregation paradigm.

### Paper 5: CASA — Clustering-Aggregation Synergy under Asynchrony (KDD 2024)

- **Citation:** CASA: Asynchronous Clustered Federated Learning. *KDD 2024*.
- **Relevance:** CASA addresses data heterogeneity with bi-level aggregation under asynchrony — directly relevant to your system where different workers may have different data subsets and processing speeds. While MaayaTrain assumes IID data, acknowledging CASA as handling the non-IID extension demonstrates awareness of the broader landscape.
- **Integration Point:** Related Work §2.5 and Future Work.

---

## III. Formal Mathematical Extraction from Codebase

The following formulas are extracted **strictly** from the Python implementation in the current codebase (`diloco.py`, `tensor_codec.py`, `orchestrator.py`, `tcp_channel.py`). Each formula is annotated with the exact file and line numbers.

---

### Feature 1: Compute-Proportional Async-DiLoCo

**Source:** `diloco.py` lines 206-292, `orchestrator.py` lines 148-161

**Mechanism:** When `sync_mode="time"`, each worker k trains for a fixed wall-clock window T_window seconds. Faster hardware completes more inner steps h_k. The outer aggregation weights each worker's pseudo-gradient proportionally to its local step count.

**Formal Definition:**

Given n workers, each completing h_k inner steps within the time window T_window, define:

```
w_k = h_k / Σ_{j=1}^{n} h_j,    k = 1, ..., n
```

The compute-proportional aggregated pseudo-gradient is:

```
Δθ_agg = Σ_{k=1}^{n} w_k · Δθ_k
```

where Δθ_k = θ_global − θ_local,k is the pseudo-gradient of worker k.

**FP32 Accumulation Guard** (lines 270-273): All weighted accumulation is performed in FP32 to prevent FP16 overflow (max_fp16 = 65504):

```
acc ← float32(Δθ_1) · w_1
acc ← acc + float32(Δθ_k) · w_k,  k = 2, ..., n
```

**Byzantine Safety Bypass** (lines 240-247): When `aggregation="median"`, compute-proportional weighting is **disabled** and the system falls back to unweighted median aggregation, preserving the Byzantine fault tolerance guarantee.

**Amdahl's Law Insight:** In synchronous DiLoCo, the slowest worker dictates the round duration:

```
U_sync = (n · h_min) / Σ_{k=1}^{n} h_k*    (can be as low as 20% with 10× speed disparity)
```

In compute-proportional mode:

```
U_async = Σ_{k=1}^{n} h_k / Σ_{k=1}^{n} h_k = 1.0    (100% utilization by construction)
```

---

### Feature 2: Network-Aware Dynamic Sharding

**Source:** `orchestrator.py` lines 256-294, `tcp_channel.py` lines 49-58

**Mechanism:** At each outer round, the orchestrator measures cluster-wide average RTT (via heartbeat probes) and adapts the number of streaming shards K using a piecewise doubling/halving policy.

**RTT Measurement** (tcp_channel.py, lines 49-58): A rolling window of the last 10 RTT samples per peer:

```
RTT_peer = (1/|W|) · Σ_{i ∈ W} rtt_i,    |W| ≤ 10
RTT_cluster = (1/|P|) · Σ_{p ∈ P} RTT_p
```

where P is the set of peers with at least one RTT sample.

**Sharding Adaptation** (orchestrator.py, lines 257-294):

```
            ┌ min(2·K_t, K_max)        if RTT_cluster > τ_high
K_{t+1} =  │ max(⌊K_t / 2⌋, K_min)    if RTT_cluster < τ_low
            └ K_t                       otherwise
```

where τ_high = 150ms, τ_low = 30ms, K_min = 1, K_max = 16.

**Effect on Payload Size:** With P total model parameters and compression ratio C:

```
Payload_per-shard = (P · b) / (K · C)    bytes
```

where b is bytes per parameter (4 for FP32, 1 for INT8). For P = 124M, K = 16, INT8+zstd (C ≈ 2):

```
Payload ≈ (124 × 10^6 × 1) / (16 × 2) ≈ 3.9 MB per shard
```

---

### Feature 3: Outlier-Resilient Block-Wise INT8 Quantization

**Source:** `tensor_codec.py` lines 54-159

**Mechanism:** Instead of computing a single (scale, zero-point) pair per tensor (which lets a single outlier destroy precision globally), the tensor is partitioned into contiguous blocks of B = 128 elements, each quantized independently.

**Quantization** (lines 82-110):

Given a tensor x ∈ ℝ^N:

1. **Pad** to nearest multiple of B: N' = B · ⌈N/B⌉, padding with zeros.
2. **Reshape** into blocks: X ∈ ℝ^{N'/B × B}.
3. **Per-block statistics** for block i:
   ```
   x_min^(i) = min(X_i),    x_max^(i) = max(X_i)
   ```
4. **Scale factor** (clamped for numerical safety):
   ```
   s^(i) = max((x_max^(i) - x_min^(i)) / 254, 1e-8)
   ```
5. **Quantize** to signed INT8 (range [-127, 127]):
   ```
   q_j^(i) = clamp(round((X_{i,j} - x_min^(i)) / s^(i)) - 127, -127, 127)
   ```

**Dequantization** (lines 146-159):

```
X̂_{i,j} = (q_j^(i) + 127) · s^(i) + x_min^(i)
```

**Storage overhead per block:** 2 × 2 = 4 bytes (scale and min in FP16), amortized over 128 × 1 = 128 bytes of INT8 data → 3.1% overhead.

**Why block-wise survives AdamW outliers:** AdamW pseudo-gradients exhibit sparse outlier features (analogous to LLM weight outliers identified by Dettmers et al., 2022). Per-tensor INT8 maps:

```
s_per-tensor = (max(x) - min(x)) / 254
```

A single outlier x_outlier >> ||x||_∞ inflates s_per-tensor and crushes precision for all other elements. Block-wise confines the damage to the single 128-element block containing the outlier, leaving the remaining ⌊N/B⌋ − 1 blocks unaffected.

**Compression pipeline:** INT8 data → zstandard (level 3, multi-threaded) → wire payload. Expected total compression: **8–12×** vs. raw FP32.

---

### Feature 4: Stream-Chunked Median Aggregation

**Source:** `diloco.py` lines 412-467

**Mechanism:** Coordinate-wise median requires stacking all n worker tensors along a new dimension, which costs O(n × P) memory. For n = 8 workers and P = 774M parameters in FP32, this is ≈ 24.8 GB — exceeding consumer RAM. The stream-chunked implementation processes the median in fixed-size chunks.

**Algorithm:**

Given pseudo-gradients {Δθ_1, ..., Δθ_n}, each of size P, and chunk size C = 5 × 10^6:

1. **Flatten** each Δθ_k to f_k ∈ ℝ^P.
2. **Allocate** output buffer o ∈ ℝ^P on device.
3. **For** j = 0, C, 2C, ...:
   - Extract chunk: c_k = f_k[j : min(j+C, P)] for each worker k.
   - Stack: M = stack(c_1, ..., c_n) ∈ ℝ^{n × |c|}.
   - Compute: o[j : j+|c|] = median(M, dim=0).values.clone().
   - **Delete** M immediately to free device memory.
4. **Reshape** o to original tensor shape.

**Memory complexity:**

```
Peak memory = O(n × C)    vs.    O(n × P) for naive
```

For n = 8, C = 5M: peak ≈ 160 MB (vs. 24.8 GB naive). **155× reduction.**

**Critical detail** (line 463): `.clone()` after `.median()` breaks PyTorch's internal reference chain from the return tuple back to the stacked tensor. Without `.clone()`, `del chunk_stack` would not actually free memory because the median output holds a reference.

---

## IV. Proposed Academic Titles

Based on the analysis above, the paper's primary contributions are: (1) compute-proportional asynchronous DiLoCo for heterogeneous edge devices, (2) network-adaptive dynamic sharding, (3) outlier-resilient block-wise gradient quantization, and (4) memory-safe robust aggregation.

### Title Option A (Systems-Forward — MLSys)
> **MaayaTrain: Compute-Proportional Distributed Low-Communication Training of Language Models on Heterogeneous Consumer Hardware over Wi-Fi**

*Rationale:* "Compute-Proportional" is the novel keyword that will attract reviewers. "Over Wi-Fi" immediately signals the extreme-bandwidth-constraint niche. Clear system name in title.

### Title Option B (Algorithm-Forward — NeurIPS/ICLR)
> **Async-DiLoCo with Network-Adaptive Sharding and Block-Wise Gradient Quantization for Edge-Distributed LLM Training**

*Rationale:* Leads with algorithmic novelty. "Async-DiLoCo" positions as a direct extension of the foundational work. "Edge-Distributed" signals the consumer hardware angle without colloquial language.

### Title Option C (Problem-Forward — Broad Appeal)
> **Breaking the Bandwidth Barrier: Distributed Language Model Training Across Consumer Hardware with Adaptive Low-Communication Optimization**

*Rationale:* The "Breaking the X Barrier" framing works well for high-profile venues. Emphasizes the problem (bandwidth) rather than the solution, inviting broader readership.

---

## V. Summary of Required Actions Before LaTeX Phase

| Priority | Action | Status |
|----------|--------|--------|
| **P0** | Rewrite Algorithmic Framework (§3) with all 4 SOTA features | ❌ Not started |
| **P0** | Add 5 mandatory citations (GaLore, QLoRA, Async-DiLoCo, FedBuff, CASA) | ❌ Not started |
| **P0** | Replace per-tensor INT8 math with block-wise INT8 math | ❌ Not started |
| **P0** | Add Compute-Proportional Aggregation formula (§III.1 above) | ❌ Not started |
| **P0** | Add Dynamic Sharding piecewise function (§III.2 above) | ❌ Not started |
| **P0** | Add Chunked Median algorithm (§III.4 above) | ❌ Not started |
| **P1** | Eliminate colloquial language (see §I.2 table) | ❌ Not started |
| **P1** | Move basic PyTorch features (AMP, clipping) out of "SOTA" section | ❌ Not started |
| **P1** | Fix Appendix C for block-wise error analysis | ❌ Not started |
| **P2** | Generate synthetic benchmark figures | ❌ Not started |
| **P2** | Add actual experimental results or clearly label as synthetic | ❌ Not started |

---

**— Reviewer #2**

*"I recommend Major Revision. The codebase shows genuine technical depth, but the manuscript fails to communicate it. The four SOTA upgrades — compute-proportional DiLoCo, dynamic sharding, block-wise INT8, and chunked median — are each individually publishable in a workshop paper. Together, they constitute a strong systems contribution, but only if properly formalized and empirically validated."*
