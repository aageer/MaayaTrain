"""Tests for Phase 3: Compute-proportional asynchronous aggregation."""

import torch
import torch.nn as nn

from maayatrain.comms.wire_format import MsgKind, decode_bytes, encode
from maayatrain.settings import DiLoCoConfig
from maayatrain.training.diloco import DiLoCoEngine


def _make_tiny_model():
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))


# -------------------------------------------------------------------
# Task 1: Settings validation
# -------------------------------------------------------------------


def test_settings_sync_mode_steps():
    """Default sync_mode is 'steps'."""
    config = DiLoCoConfig()
    assert config.sync_mode == "steps"
    assert config.sync_window_seconds == 60.0


def test_settings_sync_mode_time():
    """sync_mode='time' is accepted with custom window."""
    config = DiLoCoConfig(sync_mode="time", sync_window_seconds=30.0)
    assert config.sync_mode == "time"
    assert config.sync_window_seconds == 30.0


# -------------------------------------------------------------------
# Task 2: Wire protocol — local_steps in SYNC_GRADIENTS header
# -------------------------------------------------------------------


def test_wire_format_includes_local_steps():
    """SYNC_GRADIENTS frame includes local_steps in the JSON header."""
    local_steps = 347
    raw = encode(
        MsgKind.SYNC_GRADIENTS,
        sender_id="worker-abc",
        payload=b"fake_payload",
        extra={"local_steps": local_steps},
    )
    frame = decode_bytes(raw)

    assert frame.kind == MsgKind.SYNC_GRADIENTS
    assert frame.header["local_steps"] == 347
    assert frame.payload == b"fake_payload"


def test_wire_format_missing_local_steps_defaults():
    """Frames without local_steps (backward compat) should not crash."""
    raw = encode(
        MsgKind.SYNC_GRADIENTS,
        sender_id="worker-old",
        payload=b"",
    )
    frame = decode_bytes(raw)
    # When local_steps is missing, coordinator should fall back to inner_steps
    assert "local_steps" not in frame.header


# -------------------------------------------------------------------
# Task 3: Compute-proportional weighted aggregation
# -------------------------------------------------------------------


def test_weighted_aggregation_proportional():
    """Worker that did 3× more steps should contribute 3× more weight."""
    model = _make_tiny_model()
    config = DiLoCoConfig(outer_lr=1.0, outer_momentum=0.0, nesterov=False)
    engine = DiLoCoEngine(model, config, torch.device("cpu"))

    # Worker A: did 100 steps, gradient = +1.0
    pg_a = {n: torch.ones_like(p) for n, p in model.named_parameters()}
    # Worker B: did 300 steps, gradient = -1.0
    pg_b = {n: -torch.ones_like(p) for n, p in model.named_parameters()}

    initial = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Weighted: (100/400)*1 + (300/400)*(-1) = 0.25 - 0.75 = -0.5
    engine.apply_outer_step_weighted(
        [pg_a, pg_b],
        [100, 300],
        aggregation="mean",
    )

    for name, param in model.named_parameters():
        delta = param - initial[name]
        # With lr=1.0, momentum=0, nesterov=False:
        # v = 0 + (-0.5) = -0.5
        # param = param - 1.0 * (-0.5) = param + 0.5
        # Wait: standard momentum (non-nesterov): param -= lr * v = param -= 1*(-0.5) = param + 0.5
        # Actually let's trace exactly:
        # delta_agg = 0.25*1 + 0.75*(-1) = -0.5
        # v = 0 + (-0.5) = -0.5
        # nesterov=False: param -= lr * v = param -= 1.0 * (-0.5) = param + 0.5
        expected = 0.5
        assert torch.allclose(delta, torch.ones_like(delta) * expected, atol=1e-5), (
            f"{name}: expected delta={expected}, got {delta.mean().item():.4f}"
        )


def test_weighted_equal_steps_equals_mean():
    """When all workers have equal steps, weighted == standard mean."""
    model1 = _make_tiny_model()
    model2 = _make_tiny_model()
    model2.load_state_dict(model1.state_dict())

    config = DiLoCoConfig(outer_lr=1.0, outer_momentum=0.0, nesterov=False)
    engine1 = DiLoCoEngine(model1, config, torch.device("cpu"))
    engine2 = DiLoCoEngine(model2, config, torch.device("cpu"))

    pg = {n: torch.randn_like(p) * 0.1 for n, p in model1.named_parameters()}

    # Standard mean
    engine1.apply_outer_step([pg, pg], aggregation="mean")

    # Weighted with equal steps → should be identical
    engine2.apply_outer_step_weighted([pg, pg], [500, 500], aggregation="mean")

    for (n1, p1), (_, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), f"Weighted != mean for equal steps: {n1}"


def test_weighted_median_falls_back():
    """Median aggregation bypasses weighting (Byzantine safety)."""
    model1 = _make_tiny_model()
    model2 = _make_tiny_model()
    model2.load_state_dict(model1.state_dict())

    config = DiLoCoConfig(outer_lr=1.0, outer_momentum=0.0, nesterov=False)
    engine1 = DiLoCoEngine(model1, config, torch.device("cpu"))
    engine2 = DiLoCoEngine(model2, config, torch.device("cpu"))

    pg = {n: torch.randn_like(p) * 0.1 for n, p in model1.named_parameters()}

    # Direct median
    engine1.apply_outer_step([pg, pg], aggregation="median")

    # Weighted + median → should fall back to standard median
    engine2.apply_outer_step_weighted([pg, pg], [100, 900], aggregation="median")

    for (n1, p1), (_, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), f"Median fallback mismatch: {n1}"


def test_weighted_single_worker():
    """Single worker gets weight 1.0 — same as standard outer step."""
    model1 = _make_tiny_model()
    model2 = _make_tiny_model()
    model2.load_state_dict(model1.state_dict())

    config = DiLoCoConfig(outer_lr=1.0, outer_momentum=0.0, nesterov=False)
    engine1 = DiLoCoEngine(model1, config, torch.device("cpu"))
    engine2 = DiLoCoEngine(model2, config, torch.device("cpu"))

    pg = {n: torch.randn_like(p) * 0.1 for n, p in model1.named_parameters()}

    engine1.apply_outer_step([pg], aggregation="mean")
    engine2.apply_outer_step_weighted([pg], [500], aggregation="mean")

    for (n1, p1), (_, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), f"Single worker mismatch: {n1}"


def test_weighted_empty_gradients():
    """Empty gradient list should be handled gracefully."""
    model = _make_tiny_model()
    config = DiLoCoConfig()
    engine = DiLoCoEngine(model, config, torch.device("cpu"))

    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    engine.apply_outer_step_weighted([], [], aggregation="mean")

    # Model should be unchanged
    for name, param in model.named_parameters():
        assert torch.equal(param, initial[name])
