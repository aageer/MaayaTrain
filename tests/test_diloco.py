"""Tests for DiLoCo engine."""

import torch
import torch.nn as nn

from maayatrain.settings import DiLoCoConfig
from maayatrain.training.diloco import DiLoCoEngine


def _make_tiny_model():
    """Create a minimal model for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    )


def test_snapshot_and_pseudo_gradient():
    """Pseudo-gradient = θ_global − θ_local after training."""
    model = _make_tiny_model()
    config = DiLoCoConfig(inner_steps=5, inner_lr=0.01, outer_lr=0.5, outer_momentum=0.9)
    engine = DiLoCoEngine(model, config, torch.device("cpu"))

    # Snapshot
    engine.snapshot_global()

    # Simulate a few inner steps (modify params)
    optimizer = engine.inner_optimizer
    for _ in range(3):
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Compute pseudo-gradient
    pg = engine.compute_pseudo_gradient()

    # Should have entries for all named parameters
    param_names = [n for n, _ in model.named_parameters()]
    assert set(pg.keys()) == set(param_names)

    # Pseudo-gradients should be nonzero (params changed)
    for name, tensor in pg.items():
        assert tensor.abs().sum() > 0, f"Pseudo-gradient for {name} is all zeros"


def test_outer_step_moves_params():
    """Outer step should modify the model parameters."""
    model = _make_tiny_model()
    config = DiLoCoConfig(inner_steps=5, inner_lr=0.01, outer_lr=0.5, outer_momentum=0.9)
    engine = DiLoCoEngine(model, config, torch.device("cpu"))

    # Record initial params
    initial_params = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Create fake pseudo-gradients (nonzero)
    pg = {n: torch.randn_like(p) * 0.1 for n, p in model.named_parameters()}

    # Apply outer step
    engine.apply_outer_step([pg])

    # Params should have changed
    for name, param in model.named_parameters():
        diff = (param - initial_params[name]).abs().sum()
        assert diff > 0, f"Parameter {name} unchanged after outer step"


def test_outer_step_averages_workers():
    """With multiple workers, outer step should use the mean pseudo-gradient."""
    model = _make_tiny_model()
    config = DiLoCoConfig(inner_steps=5, inner_lr=0.01, outer_lr=1.0, outer_momentum=0.0, nesterov=False)
    engine = DiLoCoEngine(model, config, torch.device("cpu"))

    # Two workers with opposite pseudo-gradients should cancel out
    pg1 = {n: torch.ones_like(p) for n, p in model.named_parameters()}
    pg2 = {n: -torch.ones_like(p) for n, p in model.named_parameters()}

    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    engine.apply_outer_step([pg1, pg2])

    # Mean of [1, -1] = 0, so params should barely change
    # (first outer step: v = 0 + 0 = 0, update = -lr * v = 0)
    for name, param in model.named_parameters():
        diff = (param - initial[name]).abs().max()
        assert diff < 1e-6, f"Params changed too much for zero-mean gradients: {diff}"


def test_momentum_accumulates():
    """Momentum buffer should accumulate across outer steps."""
    model = _make_tiny_model()
    config = DiLoCoConfig(outer_lr=1.0, outer_momentum=0.9, nesterov=False)
    engine = DiLoCoEngine(model, config, torch.device("cpu"))

    pg = {n: torch.ones_like(p) * 0.1 for n, p in model.named_parameters()}

    # Step 1: v = 0.9*0 + 0.1 = 0.1
    engine.apply_outer_step([pg])

    # Step 2: v = 0.9*0.1 + 0.1 = 0.19
    engine.apply_outer_step([pg])

    # Momentum should have grown
    for name, v in engine._momentum_buffer.items():
        expected_v = 0.9 * 0.1 + 0.1  # 0.19
        assert torch.allclose(v, torch.ones_like(v) * expected_v, atol=1e-5)


def test_get_and_load_weights():
    """Weights can be extracted and loaded back."""
    model = _make_tiny_model()
    config = DiLoCoConfig()
    engine = DiLoCoEngine(model, config, torch.device("cpu"))

    weights = engine.get_global_weights()
    assert isinstance(weights, dict)
    assert len(weights) > 0

    # Modify model
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(42.0)

    # Load saved weights back
    engine.load_global_weights(weights)

    # Should restore original values (not 42)
    for name, param in model.named_parameters():
        assert not torch.allclose(param, torch.tensor(42.0))
