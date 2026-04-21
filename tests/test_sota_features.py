"""Tests for SOTA DiLoCo features: median aggregation and streaming sync."""

import torch
import torch.nn as nn

from maayatrain.settings import DiLoCoConfig
from maayatrain.training.diloco import DiLoCoEngine


def _make_tiny_model():
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))


def test_median_aggregation():
    """Median aggregation should reject outlier pseudo-gradients."""
    model = _make_tiny_model()
    config = DiLoCoConfig(outer_lr=1.0, outer_momentum=0.0, nesterov=False)
    engine = DiLoCoEngine(model, config, torch.device("cpu"))

    # 3 workers: 2 agree, 1 is an outlier
    pg_normal_1 = {n: torch.ones_like(p) * 0.1 for n, p in model.named_parameters()}
    pg_normal_2 = {n: torch.ones_like(p) * 0.1 for n, p in model.named_parameters()}
    pg_outlier = {n: torch.ones_like(p) * 1000.0 for n, p in model.named_parameters()}

    initial = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Median should pick 0.1, not 1000
    engine.apply_outer_step([pg_normal_1, pg_normal_2, pg_outlier], aggregation="median")

    for name, param in model.named_parameters():
        diff = (param - initial[name]).abs().max()
        # With lr=1.0, momentum=0, the update should be about 0.1 per param
        assert diff < 1.0, f"Median didn't reject outlier: diff={diff}"


def test_mean_vs_median_identical_workers():
    """With identical workers, mean and median produce the same result."""
    model1 = _make_tiny_model()
    model2 = _make_tiny_model()
    # Make models identical
    model2.load_state_dict(model1.state_dict())

    config = DiLoCoConfig(outer_lr=1.0, outer_momentum=0.0, nesterov=False)
    engine1 = DiLoCoEngine(model1, config, torch.device("cpu"))
    engine2 = DiLoCoEngine(model2, config, torch.device("cpu"))

    pg = {n: torch.randn_like(p) * 0.1 for n, p in model1.named_parameters()}

    engine1.apply_outer_step([pg, pg], aggregation="mean")
    engine2.apply_outer_step([pg, pg], aggregation="median")

    for (n1, p1), (_, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), f"Mean != Median for identical workers: {n1}"


def test_streaming_shards():
    """Streaming sync splits parameters into correct number of shards."""
    model = _make_tiny_model()
    config = DiLoCoConfig()
    engine = DiLoCoEngine(model, config, torch.device("cpu"))

    shards = engine.compute_streaming_shards(num_shards=2)
    assert len(shards) == 2

    # All params present exactly once
    all_names = set()
    for shard in shards:
        for name in shard:
            assert name not in all_names, f"Duplicate: {name}"
            all_names.add(name)

    expected_names = {n for n, _ in model.named_parameters()}
    assert all_names == expected_names


def test_shard_outer_step():
    """Applying outer step shard-by-shard should modify only those params."""
    model = _make_tiny_model()
    config = DiLoCoConfig(outer_lr=1.0, outer_momentum=0.0, nesterov=False)
    engine = DiLoCoEngine(model, config, torch.device("cpu"))

    shards = engine.compute_streaming_shards(num_shards=2)
    pg = {n: torch.ones_like(p) * 0.1 for n, p in model.named_parameters()}

    initial = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Apply only to shard 0
    engine.apply_outer_step_shard([pg], shard_names=shards[0])

    # Shard 0 params should have changed
    param_dict = dict(model.named_parameters())
    for name in shards[0]:
        diff = (param_dict[name] - initial[name]).abs().max()
        assert diff > 0, f"Shard 0 param {name} didn't change"

    # Shard 1 params should be unchanged
    for name in shards[1]:
        diff = (param_dict[name] - initial[name]).abs().max()
        assert diff == 0, f"Shard 1 param {name} changed unexpectedly"


def test_streaming_full_update_matches_single_step():
    """Applying all shards should produce the same result as a single outer step."""
    model1 = _make_tiny_model()
    model2 = _make_tiny_model()
    model2.load_state_dict(model1.state_dict())

    config = DiLoCoConfig(outer_lr=1.0, outer_momentum=0.0, nesterov=False)
    engine1 = DiLoCoEngine(model1, config, torch.device("cpu"))
    engine2 = DiLoCoEngine(model2, config, torch.device("cpu"))

    pg = {n: torch.randn_like(p) * 0.1 for n, p in model1.named_parameters()}

    # Method 1: single full outer step
    engine1.apply_outer_step([pg])

    # Method 2: streaming shard-by-shard
    shards = engine2.compute_streaming_shards(num_shards=3)
    for shard in shards:
        engine2.apply_outer_step_shard([pg], shard_names=shard)

    # Should produce identical results
    for (n1, p1), (_, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), f"Streaming mismatch: {n1}"
