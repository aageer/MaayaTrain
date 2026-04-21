"""Learning rate schedulers for MaayaTrain.

Implements the gold-standard LLM training schedule:
**Linear warmup → Cosine decay to min_lr**.

Based on:
- Chinchilla (Hoffmann et al., 2022): cosine decay is optimal for LLMs
- GPT-3 (Brown et al., 2020): linear warmup prevents early instability
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def cosine_warmup_lr(
    current_step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Compute the LR multiplier for a given step.

    Returns a value in [min_lr_ratio, 1.0]:
    - During warmup: linearly ramps from 0 to 1.0
    - After warmup: cosine decays from 1.0 to min_lr_ratio

    Parameters
    ----------
    current_step : int
        Current training step (0-indexed).
    warmup_steps : int
        Number of warmup steps.
    total_steps : int
        Total number of training steps.
    min_lr_ratio : float
        Minimum LR as a fraction of peak LR (default 0.1 = 10%).

    Returns
    -------
    float
        LR multiplier to apply to base LR.
    """
    if current_step < warmup_steps:
        # Linear warmup: 0 → 1.0
        return float(current_step) / float(max(1, warmup_steps))

    if current_step >= total_steps:
        return min_lr_ratio

    # Cosine decay: 1.0 → min_lr_ratio
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a PyTorch LR scheduler with linear warmup + cosine decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule.
    warmup_steps : int
        Number of linear warmup steps.
    total_steps : int
        Total training steps (warmup + decay).
    min_lr_ratio : float
        Minimum LR as fraction of peak.

    Returns
    -------
    LambdaLR
        The scheduler. Call ``.step()`` after each optimizer step.
    """
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_warmup_lr(step, warmup_steps, total_steps, min_lr_ratio),
    )
