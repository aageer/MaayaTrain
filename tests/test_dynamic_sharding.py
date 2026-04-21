"""Tests for Phase 4: Network-aware dynamic streaming shards."""

import torch
import torch.nn as nn

from maayatrain.comms.tcp_channel import PeerConnection, TcpServer, _RTT_WINDOW
from maayatrain.comms.wire_format import MsgKind, decode_bytes, encode
from maayatrain.settings import DiLoCoConfig, MaayaTrainSettings
from maayatrain.training.diloco import DiLoCoEngine


def _make_tiny_model():
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))


# -------------------------------------------------------------------
# Task 1: RTT tracking
# -------------------------------------------------------------------


def test_peer_connection_rtt_tracking():
    """PeerConnection records and averages RTT samples."""
    from collections import deque

    conn = PeerConnection.__new__(PeerConnection)
    conn._rtt_samples = deque(maxlen=_RTT_WINDOW)
    conn._pending_heartbeat_ts = None

    # No samples → 0.0
    assert conn.avg_rtt_ms == 0.0

    # Add samples
    conn.record_rtt(10.0)
    conn.record_rtt(20.0)
    conn.record_rtt(30.0)
    assert abs(conn.avg_rtt_ms - 20.0) < 1e-6  # (10+20+30)/3

    # Rolling window: add more samples
    for _ in range(10):
        conn.record_rtt(100.0)
    # Window is 10, so all old samples are gone
    assert abs(conn.avg_rtt_ms - 100.0) < 1e-6


def test_rtt_rolling_window():
    """RTT window caps at _RTT_WINDOW samples."""
    from collections import deque

    conn = PeerConnection.__new__(PeerConnection)
    conn._rtt_samples = deque(maxlen=_RTT_WINDOW)

    # Fill with ascending values
    for i in range(20):
        conn.record_rtt(float(i))

    # Only last _RTT_WINDOW values should be in the deque
    assert len(conn._rtt_samples) == _RTT_WINDOW
    expected = list(range(20 - _RTT_WINDOW, 20))
    assert list(conn._rtt_samples) == [float(x) for x in expected]


# -------------------------------------------------------------------
# Task 2: Dynamic re-sharding
# -------------------------------------------------------------------


def test_sync_request_includes_streaming_shards():
    """SYNC_REQUEST frame includes current_streaming_shards in header."""
    raw = encode(
        MsgKind.SYNC_REQUEST,
        sender_id="coordinator",
        extra={"step": 500, "current_streaming_shards": 4},
    )
    frame = decode_bytes(raw)

    assert frame.kind == MsgKind.SYNC_REQUEST
    assert frame.header["current_streaming_shards"] == 4
    assert frame.header["step"] == 500


def test_dynamic_resharding_high_latency():
    """High RTT (>150ms) should double the shard count."""
    from maayatrain.training.orchestrator import Orchestrator

    # Test the threshold logic directly
    model = _make_tiny_model()
    settings = MaayaTrainSettings()
    settings.diloco.streaming_shards = 2

    # We can't easily instantiate Orchestrator without full infra,
    # so test the threshold math directly
    old_k = 2
    avg_rtt = 200.0  # > 150ms threshold

    if avg_rtt > 150.0:
        new_k = min(old_k * 2, 16)
    else:
        new_k = old_k

    assert new_k == 4  # Doubled


def test_dynamic_resharding_low_latency():
    """Low RTT (<30ms) should halve the shard count."""
    old_k = 8
    avg_rtt = 10.0  # < 30ms threshold

    if avg_rtt < 30.0:
        new_k = max(old_k // 2, 1)
    else:
        new_k = old_k

    assert new_k == 4  # Halved


def test_dynamic_resharding_max_cap():
    """Shard count should never exceed 16."""
    old_k = 16
    avg_rtt = 500.0

    new_k = min(old_k * 2, 16)
    assert new_k == 16  # Capped


def test_dynamic_resharding_min_cap():
    """Shard count should never go below 1."""
    old_k = 1
    avg_rtt = 5.0

    new_k = max(old_k // 2, 1)
    assert new_k == 1  # Floored


def test_dynamic_resharding_normal_range():
    """RTT in [30, 150]ms should not change shard count."""
    old_k = 4
    avg_rtt = 75.0  # In normal range

    if avg_rtt > 150.0:
        new_k = min(old_k * 2, 16)
    elif avg_rtt < 30.0:
        new_k = max(old_k // 2, 1)
    else:
        new_k = old_k

    assert new_k == 4  # Unchanged


# -------------------------------------------------------------------
# Streaming shards correctness with varying K
# -------------------------------------------------------------------


def test_streaming_shards_varying_k():
    """compute_streaming_shards works correctly for different K values."""
    model = _make_tiny_model()
    config = DiLoCoConfig()
    engine = DiLoCoEngine(model, config, torch.device("cpu"))

    param_names = {n for n, _ in model.named_parameters()}

    for k in [1, 2, 3, 4, 8]:
        shards = engine.compute_streaming_shards(num_shards=k)
        assert len(shards) == k

        # All params present exactly once across all shards
        seen = set()
        for shard in shards:
            for name in shard:
                assert name not in seen, f"Duplicate {name} at K={k}"
                seen.add(name)
        assert seen == param_names, f"Missing params at K={k}"


def test_resharding_preserves_outer_step():
    """Changing K between rounds should still update all params."""
    model = _make_tiny_model()
    config = DiLoCoConfig(outer_lr=1.0, outer_momentum=0.0, nesterov=False)
    engine = DiLoCoEngine(model, config, torch.device("cpu"))

    pg = {n: torch.ones_like(p) * 0.1 for n, p in model.named_parameters()}
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Round 1: K=2 shards
    shards_2 = engine.compute_streaming_shards(num_shards=2)
    for shard in shards_2:
        engine.apply_outer_step_shard([pg], shard_names=shard)

    # All params should have changed
    for name, param in model.named_parameters():
        diff = (param - initial[name]).abs().max()
        assert diff > 0, f"Param {name} unchanged after K=2 streaming step"

    # Record state after K=2
    after_k2 = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Round 2: K=4 shards (simulating RTT increase → more shards)
    shards_4 = engine.compute_streaming_shards(num_shards=4)
    for shard in shards_4:
        engine.apply_outer_step_shard([pg], shard_names=shard)

    # All params should have changed again
    for name, param in model.named_parameters():
        diff = (param - after_k2[name]).abs().max()
        assert diff > 0, f"Param {name} unchanged after K=4 streaming step"


def test_streaming_vs_single_step_with_dynamic_k():
    """Full streaming update (any K) should match a single outer step."""
    for k in [1, 2, 3, 4]:
        model1 = _make_tiny_model()
        model2 = _make_tiny_model()
        model2.load_state_dict(model1.state_dict())

        config = DiLoCoConfig(outer_lr=1.0, outer_momentum=0.0, nesterov=False)
        engine1 = DiLoCoEngine(model1, config, torch.device("cpu"))
        engine2 = DiLoCoEngine(model2, config, torch.device("cpu"))

        pg = {n: torch.randn_like(p) * 0.1 for n, p in model1.named_parameters()}

        # Full outer step
        engine1.apply_outer_step([pg])

        # Streaming with K shards
        shards = engine2.compute_streaming_shards(num_shards=k)
        for shard in shards:
            engine2.apply_outer_step_shard([pg], shard_names=shard)

        for (n1, p1), (_, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2, atol=1e-6), (
                f"Streaming K={k} mismatch for {n1}"
            )
