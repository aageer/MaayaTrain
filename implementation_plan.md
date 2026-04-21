# MaayaTrain — SOTA Improvements Plan

**Author:** Akhil Ageer  
**Date:** 2026-04-20  
**CRD Compliance:** All improvements independently designed from public research papers.

---

## Research Findings

Based on SOTA research in distributed ML training (2024-2025):

| Source | Technique | Impact |
|--------|-----------|--------|
| Google DeepMind (arXiv:2501.18512) | **Streaming DiLoCo** — sync parameter subsets in sequence | 100× less peak bandwidth |
| Google DeepMind (2025) | **4-bit quantization** of outer gradients | 8× less transfer per sync |
| LLM training best practices | **Cosine warmup LR schedule** | Better convergence, avoids early divergence |
| PyTorch AMP (2025) | **Mixed precision (bfloat16/float16)** | 2× faster training, 50% less memory |
| OpenDiLoCo (PrimeIntellect) | **Gradient accumulation** | Train larger effective batches on small GPUs |
| SPARTA (OpenReview 2024) | **Robust aggregation (median)** | Byzantine fault tolerance |
| HuggingFace ecosystem | **Tokenizer + datasets** | Production-quality data pipeline |

---

## Proposed Changes

### 1. Mixed Precision Training (AMP)
**File:** `maayatrain/training/loop.py`  
- Wrap forward pass in `torch.amp.autocast`
- Use `GradScaler` for FP16 (skip for bfloat16)
- Auto-detect bfloat16 support (Ampere+ GPUs, Apple M-series)
- ~2× speedup with 50% memory reduction

### 2. Cosine Warmup Learning Rate Schedule
**File:** `maayatrain/training/lr_schedule.py` [NEW]  
- Linear warmup for first 10% of steps
- Cosine decay to `min_lr` (10% of peak)
- Per-inner-loop scheduling (resets each DiLoCo round)
- Config: `warmup_steps`, `min_lr_ratio` in `[training]`

### 3. Gradient Accumulation
**File:** `maayatrain/training/loop.py`  
- Accumulate gradients over N micro-batches before stepping
- Simulate larger batch sizes on memory-constrained devices
- Config: `gradient_accumulation_steps` in `[training]`

### 4. Streaming Sync (from Streaming DiLoCo paper)
**File:** `maayatrain/training/diloco.py`  
- Split parameters into K shards (by layer groups)
- Sync one shard at a time during inner loop
- Overlap communication with computation
- Config: `streaming_shards` in `[diloco]`

### 5. 4-bit Quantized Gradient Compression
**File:** `maayatrain/comms/tensor_codec.py`  
- Add INT8 quantization option (4-bit needs custom kernels)
- Scale-based quantization: normalize → quantize → pack
- ~4× smaller than FP16, ~8× smaller than FP32
- Config: `compress_int8` in `[diloco]`

### 6. Robust Aggregation (Byzantine Tolerance)
**File:** `maayatrain/training/diloco.py`  
- Add coordinate-wise median aggregation option
- Tolerates up to 1/3 faulty/malicious workers
- Config: `aggregation = "mean" | "median"` in `[diloco]`

### 7. HuggingFace Tokenizer Integration
**File:** `maayatrain/training/tokenizer.py` [NEW]  
- BPE tokenizer training from raw text
- Vocabulary persistence (save/load)
- Replaces character-level tokenization for real training

### 8. Auto-Batch Sizing
**File:** `maayatrain/training/loop.py`  
- Binary search for max batch size that fits in VRAM
- Prevents OOM crashes on heterogeneous hardware

---

## Verification Plan

- All existing 27 tests still pass
- New tests for each feature
- Verify mixed precision + DiLoCo integration
