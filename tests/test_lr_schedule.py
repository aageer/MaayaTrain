"""Tests for cosine warmup learning rate schedule."""

from maayatrain.training.lr_schedule import build_cosine_warmup_scheduler, cosine_warmup_lr
import torch


def test_warmup_phase():
    """LR linearly increases during warmup."""
    warmup = 100
    total = 1000

    lr0 = cosine_warmup_lr(0, warmup, total, 0.1)
    lr50 = cosine_warmup_lr(50, warmup, total, 0.1)
    lr99 = cosine_warmup_lr(99, warmup, total, 0.1)

    assert lr0 == 0.0
    assert 0.4 < lr50 < 0.6  # ~0.5
    assert lr99 > 0.9  # Almost at peak


def test_peak_lr():
    """LR reaches 1.0 at end of warmup."""
    lr = cosine_warmup_lr(100, 100, 1000, 0.1)
    assert 0.99 < lr <= 1.0


def test_cosine_decay():
    """LR cosine decays from 1.0 to min_lr after warmup."""
    warmup = 100
    total = 1000
    min_ratio = 0.1

    lr_start = cosine_warmup_lr(100, warmup, total, min_ratio)
    lr_mid = cosine_warmup_lr(550, warmup, total, min_ratio)
    lr_end = cosine_warmup_lr(999, warmup, total, min_ratio)

    assert lr_start > 0.9  # Just past warmup
    assert 0.4 < lr_mid < 0.7  # Somewhere in middle of decay
    assert lr_end < 0.2  # Near minimum


def test_past_total_steps():
    """LR clamps to min_lr_ratio after total_steps."""
    lr = cosine_warmup_lr(2000, 100, 1000, 0.1)
    assert lr == 0.1


def test_scheduler_integration():
    """build_cosine_warmup_scheduler works with a real optimizer."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = build_cosine_warmup_scheduler(optimizer, 10, 100, 0.1)

    # Simulate 20 steps
    lrs = []
    for _ in range(20):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    # LR should increase during warmup (first 10 steps)
    assert lrs[1] > lrs[0]
    assert lrs[9] > lrs[1]
    # Then start decaying
    assert lrs[15] < lrs[10]


def test_zero_warmup():
    """Zero warmup steps: goes straight to cosine decay."""
    lr = cosine_warmup_lr(0, 0, 100, 0.1)
    assert 0.9 < lr <= 1.0
