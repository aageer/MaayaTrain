"""Base training loop for MaayaTrain.

Device-agnostic: all tensors flow through ``device`` passed at init,
so the same loop works on CUDA, MPS, XPU, or CPU.

SOTA features (independently designed from public research):
- Mixed precision (AMP) with auto-detection of bfloat16 support
- Gradient accumulation for effective large batch training
- Cosine warmup learning rate schedule
- Gradient clipping with configurable norm
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .lr_schedule import build_cosine_warmup_scheduler

logger = logging.getLogger("maayatrain.training")


@dataclass
class StepMetrics:
    """Metrics emitted after each training step."""

    step: int
    loss: float
    tokens_per_sec: float
    lr: float
    elapsed_sec: float
    gpu_mem_gb: float = 0.0  # peak GPU memory allocated


class SimpleTextDataset:
    """Minimal character-level text dataset for bootstrapping.

    Reads a text file, tokenises with a simple byte-pair-free approach
    (character-level or space-level), and yields fixed-length sequences.
    For production, replace with HuggingFace ``datasets`` integration.
    """

    def __init__(self, path: str, seq_length: int = 512, device: str = "cpu") -> None:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
        # Build a character-level vocabulary
        chars = sorted(set(raw))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in raw], dtype=torch.long)
        self.seq_length = seq_length
        self.device = device
        logger.info(
            "Loaded dataset: %d chars, vocab_size=%d, seq_length=%d",
            len(self.data),
            self.vocab_size,
            seq_length,
        )

    def random_batch(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Return a random (input, target) pair of shape (batch_size, seq_length)."""
        max_start = len(self.data) - self.seq_length - 1
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.data[s : s + self.seq_length] for s in starts])
        y = torch.stack([self.data[s + 1 : s + self.seq_length + 1] for s in starts])
        return x.to(self.device), y.to(self.device)

    def sequential_batches(
        self, batch_size: int, *, shuffle: bool = True
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        """Yield non-overlapping batches covering the entire dataset once."""
        total_tokens = len(self.data) - 1
        chunk = self.seq_length
        n_chunks = total_tokens // chunk

        indices = list(range(n_chunks))
        if shuffle:
            import random

            random.shuffle(indices)

        batch_x, batch_y = [], []
        for idx in indices:
            start = idx * chunk
            x = self.data[start : start + chunk]
            y = self.data[start + 1 : start + chunk + 1]
            batch_x.append(x)
            batch_y.append(y)
            if len(batch_x) == batch_size:
                yield (
                    torch.stack(batch_x).to(self.device),
                    torch.stack(batch_y).to(self.device),
                )
                batch_x, batch_y = [], []


# ---------------------------------------------------------------------------
# Mixed Precision helpers
# ---------------------------------------------------------------------------


def _detect_amp_dtype(device: torch.device, preference: str = "auto") -> Optional[torch.dtype]:
    """Determine the best AMP dtype for the current hardware.

    Returns None if AMP should be disabled (CPU or unsupported).
    """
    if preference == "off":
        return None

    if preference == "bf16":
        return torch.bfloat16
    if preference == "fp16":
        return torch.float16

    # Auto-detect
    if device.type == "cuda":
        # Ampere+ (compute capability ≥8.0) → bfloat16, else float16
        props = torch.cuda.get_device_properties(device)
        if props.major >= 8:
            return torch.bfloat16
        return torch.float16

    if device.type == "mps":
        # Apple Silicon supports float16 natively
        return torch.float16

    # CPU — AMP with bfloat16 is supported but marginal benefit
    return None


def _get_peak_memory_gb(device: torch.device) -> float:
    """Return peak GPU memory in GB (0.0 for CPU)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**3)
    return 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: SimpleTextDataset,
    *,
    num_steps: int,
    batch_size: int,
    device: torch.device,
    start_step: int = 0,
    log_every: int = 10,
    on_step: Optional[Callable[[StepMetrics], None]] = None,
    # SOTA features
    mixed_precision: str = "auto",
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
) -> List[StepMetrics]:
    """Run *num_steps* of training and return per-step metrics.

    Parameters
    ----------
    model : nn.Module
        The model to train (should already be on *device*).
    optimizer : torch.optim.Optimizer
        The inner optimizer (e.g. AdamW).
    dataset : SimpleTextDataset
        Data source.
    num_steps : int
        How many gradient steps to perform (one inner loop of DiLoCo).
    batch_size : int
        Micro-batch size per step.
    device : torch.device
        Compute device.
    start_step : int
        Global step counter offset (for logging).
    log_every : int
        Print metrics every N steps.
    on_step : callable | None
        Optional callback invoked after each step with StepMetrics.
    mixed_precision : str
        AMP mode: "auto", "fp16", "bf16", or "off".
    gradient_accumulation_steps : int
        Number of micro-batches to accumulate before stepping.
    warmup_steps : int
        Linear warmup steps (0 = auto 10% of num_steps).
    min_lr_ratio : float
        Minimum LR as ratio of peak for cosine decay.
    max_grad_norm : float
        Gradient clipping max norm.

    Returns
    -------
    list[StepMetrics]
        Metrics for every step executed.
    """
    model.train()
    all_metrics: List[StepMetrics] = []
    loss_fn = nn.CrossEntropyLoss()
    accum = gradient_accumulation_steps

    # --- AMP setup ---
    amp_dtype = _detect_amp_dtype(device, mixed_precision)
    use_amp = amp_dtype is not None
    # GradScaler is only needed for float16 (bfloat16 has full dynamic range)
    use_scaler = use_amp and amp_dtype == torch.float16

    if use_amp:
        logger.info("Mixed precision: %s (AMP enabled)", amp_dtype)
    else:
        logger.info("Mixed precision: off (full FP32)")

    scaler = torch.amp.GradScaler(device.type) if use_scaler else None

    # --- LR schedule setup ---
    effective_warmup = warmup_steps if warmup_steps > 0 else max(1, num_steps // 10)
    scheduler = build_cosine_warmup_scheduler(
        optimizer, effective_warmup, num_steps, min_lr_ratio
    )
    logger.info(
        "LR schedule: linear warmup (%d steps) → cosine decay (min %.1f%%)",
        effective_warmup,
        min_lr_ratio * 100,
    )

    # Reset CUDA peak memory counter
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    running_loss = 0.0

    for local_step in range(num_steps):
        step = start_step + local_step
        step_loss = 0.0

        # --- Gradient accumulation loop ---
        for micro_step in range(accum):
            x, y = dataset.random_batch(batch_size)

            # Forward pass (with or without AMP)
            if use_amp:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(x)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / accum  # Normalize for accumulation
            else:
                logits = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / accum

            # Backward pass
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_loss += loss.item()

        # --- Optimizer step (after all micro-batches) ---
        if use_scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)  # More memory-efficient
        scheduler.step()

        # --- Metrics ---
        elapsed = time.time() - t0
        effective_batch = batch_size * accum
        tokens_processed = (local_step + 1) * effective_batch * dataset.seq_length
        tps = tokens_processed / max(elapsed, 1e-9)
        lr = scheduler.get_last_lr()[0]
        gpu_mem = _get_peak_memory_gb(device)

        metrics = StepMetrics(
            step=step,
            loss=step_loss,  # Already accumulated
            tokens_per_sec=tps,
            lr=lr,
            elapsed_sec=elapsed,
            gpu_mem_gb=gpu_mem,
        )
        all_metrics.append(metrics)

        if on_step:
            on_step(metrics)

        if (local_step + 1) % log_every == 0:
            logger.info(
                "step %d | loss %.4f | %.0f tok/s | lr %.2e | mem %.1fGB",
                step,
                metrics.loss,
                tps,
                lr,
                gpu_mem,
            )

    return all_metrics


def train_steps_timed(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: SimpleTextDataset,
    *,
    window_seconds: float,
    batch_size: int,
    device: torch.device,
    start_step: int = 0,
    log_every: int = 10,
    on_step: Optional[Callable[[StepMetrics], None]] = None,
    mixed_precision: str = "auto",
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    estimated_steps: int = 500,
) -> Tuple[List[StepMetrics], int]:
    """Run training for a fixed time window, returning metrics and step count.

    Unlike :func:`train_steps`, which runs for a fixed number of steps,
    this function runs as many steps as possible within *window_seconds*.
    Faster hardware will complete more steps, enabling compute-proportional
    aggregation in heterogeneous clusters.

    Parameters
    ----------
    model, optimizer, dataset, batch_size, device, start_step, log_every,
    on_step, mixed_precision, gradient_accumulation_steps, warmup_steps,
    min_lr_ratio, max_grad_norm :
        Same as :func:`train_steps`.
    window_seconds : float
        Maximum wall-clock time for training (in seconds).
    estimated_steps : int
        Estimated number of steps for LR schedule sizing (default 500).

    Returns
    -------
    tuple[list[StepMetrics], int]
        (metrics_list, local_steps_completed)
    """
    model.train()
    all_metrics: List[StepMetrics] = []
    loss_fn = nn.CrossEntropyLoss()
    accum = gradient_accumulation_steps

    # --- AMP setup ---
    amp_dtype = _detect_amp_dtype(device, mixed_precision)
    use_amp = amp_dtype is not None
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type) if use_scaler else None

    # --- LR schedule (use estimated_steps for schedule length) ---
    effective_warmup = warmup_steps if warmup_steps > 0 else max(1, estimated_steps // 10)
    scheduler = build_cosine_warmup_scheduler(
        optimizer, effective_warmup, estimated_steps, min_lr_ratio
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    running_loss = 0.0
    local_step = 0

    while (time.time() - t0) < window_seconds:
        step = start_step + local_step
        step_loss = 0.0

        for micro_step in range(accum):
            x, y = dataset.random_batch(batch_size)

            if use_amp:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(x)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / accum
            else:
                logits = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / accum

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_loss += loss.item()

        if use_scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        # Only step scheduler if we haven't exceeded estimated_steps
        if local_step < estimated_steps:
            scheduler.step()

        elapsed = time.time() - t0
        effective_batch = batch_size * accum
        tokens_processed = (local_step + 1) * effective_batch * dataset.seq_length
        tps = tokens_processed / max(elapsed, 1e-9)
        lr = scheduler.get_last_lr()[0]
        gpu_mem = _get_peak_memory_gb(device)

        metrics = StepMetrics(
            step=step,
            loss=step_loss,
            tokens_per_sec=tps,
            lr=lr,
            elapsed_sec=elapsed,
            gpu_mem_gb=gpu_mem,
        )
        all_metrics.append(metrics)

        if on_step:
            on_step(metrics)

        if (local_step + 1) % log_every == 0:
            logger.info(
                "step %d | loss %.4f | %.0f tok/s | lr %.2e | mem %.1fGB",
                step,
                metrics.loss,
                tps,
                lr,
                gpu_mem,
            )

        local_step += 1

    logger.info(
        "Time-bounded loop: %d steps in %.1fs (window=%.1fs)",
        local_step,
        time.time() - t0,
        window_seconds,
    )
    return all_metrics, local_step

