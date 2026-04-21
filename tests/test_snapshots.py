"""Tests for checkpoint snapshot system."""

import json
from pathlib import Path

import torch
import torch.nn as nn

from maayatrain.training.snapshots import (
    SnapshotMeta,
    export_relay,
    load_snapshot,
    save_snapshot,
    step_directory,
)


def _tiny_model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))


def test_save_and_load(tmp_path: Path):
    """Save a snapshot and load it back — params should match."""
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train one step to populate optimizer state
    x = torch.randn(2, 8)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    # Save
    meta = SnapshotMeta(model_name="test", global_step=100, loss=2.5)
    ckpt_dir = tmp_path / "ckpt"
    save_snapshot(ckpt_dir, model, optimizer, meta)

    # Verify files exist
    assert (ckpt_dir / "model.safetensors").exists()
    assert (ckpt_dir / "optimizer.pt").exists()
    assert (ckpt_dir / "meta.json").exists()

    # Load into fresh model
    model2 = _tiny_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    loaded_meta = load_snapshot(ckpt_dir, model2, optimizer2)

    assert loaded_meta.global_step == 100
    assert loaded_meta.loss == 2.5
    assert loaded_meta.model_name == "test"

    # Params should match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)


def test_meta_json_format(tmp_path: Path):
    """meta.json should be valid JSON with expected fields."""
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    meta = SnapshotMeta(
        model_name="gpt2-small",
        global_step=5000,
        loss=3.42,
        total_compute_hours=2.5,
        contributors=["alice", "bob"],
    )
    ckpt_dir = tmp_path / "ckpt"
    save_snapshot(ckpt_dir, model, optimizer, meta)

    data = json.loads((ckpt_dir / "meta.json").read_text())
    assert data["model_name"] == "gpt2-small"
    assert data["global_step"] == 5000
    assert data["contributors"] == ["alice", "bob"]
    assert "created_at" in data


def test_step_directory():
    p = step_directory("./checkpoints", 5000)
    assert str(p).endswith("step-005000")


def test_export_relay(tmp_path: Path):
    """export_relay copies checkpoint and adds relay metadata."""
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    meta = SnapshotMeta(model_name="test", global_step=1000)
    src = tmp_path / "original"
    save_snapshot(src, model, optimizer, meta)

    dst = tmp_path / "relay"
    export_relay(src, dst, description="Need more compute")

    data = json.loads((dst / "meta.json").read_text())
    assert data["description"] == "Need more compute"
    assert "relay_exported_at" in data
    assert (dst / "model.safetensors").exists()
