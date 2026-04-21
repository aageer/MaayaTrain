"""Checkpoint snapshot system for MaayaTrain.

Saves complete training state as a portable directory::

    checkpoints/step-5000/
    ├── model.safetensors     # Model weights (HuggingFace safetensors)
    ├── optimizer.pt          # Optimizer state (PyTorch native)
    ├── momentum.pt           # DiLoCo outer momentum buffers
    └── meta.json             # Training metadata

Supports:
* Automatic periodic saves
* Interrupt-safe saves (SIGINT handler)
* Export/import for relay handoffs
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from safetensors.torch import load_file as st_load, save_file as st_save
from torch import Tensor, nn

logger = logging.getLogger("maayatrain.snapshots")


class SnapshotMeta:
    """Metadata accompanying a saved checkpoint."""

    def __init__(
        self,
        *,
        version: str = "0.1.0",
        model_name: str = "unknown",
        global_step: int = 0,
        loss: float = float("inf"),
        total_compute_hours: float = 0.0,
        contributors: Optional[List[str]] = None,
        description: str = "",
        created_at: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.version = version
        self.model_name = model_name
        self.global_step = global_step
        self.loss = loss
        self.total_compute_hours = total_compute_hours
        self.contributors = contributors or []
        self.description = description
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.extra = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "model_name": self.model_name,
            "global_step": self.global_step,
            "loss": self.loss,
            "total_compute_hours": self.total_compute_hours,
            "contributors": self.contributors,
            "description": self.description,
            "created_at": self.created_at,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnapshotMeta":
        known_keys = {
            "version", "model_name", "global_step", "loss",
            "total_compute_hours", "contributors", "description", "created_at",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            version=data.get("version", "0.1.0"),
            model_name=data.get("model_name", "unknown"),
            global_step=data.get("global_step", 0),
            loss=data.get("loss", float("inf")),
            total_compute_hours=data.get("total_compute_hours", 0.0),
            contributors=data.get("contributors", []),
            description=data.get("description", ""),
            created_at=data.get("created_at"),
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_snapshot(
    directory: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    meta: SnapshotMeta,
    momentum_buffer: Optional[Dict[str, Tensor]] = None,
) -> Path:
    """Save a complete training snapshot to *directory*.

    Creates the directory if it doesn't exist. Returns the directory path.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # 1. Model weights → safetensors
    state_dict = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    st_save(state_dict, str(directory / "model.safetensors"))

    # 2. Optimizer state → PyTorch native
    torch.save(optimizer.state_dict(), directory / "optimizer.pt")

    # 3. Outer momentum buffer (if provided)
    if momentum_buffer:
        torch.save(
            {k: v.cpu() for k, v in momentum_buffer.items()},
            directory / "momentum.pt",
        )

    # 4. Metadata → JSON
    (directory / "meta.json").write_text(
        json.dumps(meta.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Snapshot saved: %s (step %d, loss %.4f)", directory, meta.global_step, meta.loss)
    return directory


def load_snapshot(
    directory: Path | str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> SnapshotMeta:
    """Load a snapshot from *directory* into the given model (and optionally optimizer).

    Returns the loaded metadata.
    """
    directory = Path(directory)

    # 1. Model weights
    weights = st_load(str(directory / "model.safetensors"), device=device)
    model.load_state_dict(weights, strict=False)

    # 2. Optimizer state
    opt_path = directory / "optimizer.pt"
    if optimizer and opt_path.exists():
        opt_state = torch.load(opt_path, map_location=device, weights_only=True)
        optimizer.load_state_dict(opt_state)

    # 3. Metadata
    meta_path = directory / "meta.json"
    meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = SnapshotMeta.from_dict(meta_data)

    logger.info(
        "Snapshot loaded: %s (step %d, loss %.4f)",
        directory,
        meta.global_step,
        meta.loss,
    )
    return meta


def load_momentum_buffer(
    directory: Path | str,
    device: str = "cpu",
) -> Dict[str, Tensor]:
    """Load the outer momentum buffer from a snapshot, if present."""
    path = Path(directory) / "momentum.pt"
    if path.exists():
        return torch.load(path, map_location=device, weights_only=True)
    return {}


# ---------------------------------------------------------------------------
# Auto-name helper
# ---------------------------------------------------------------------------


def step_directory(base: str, step: int) -> Path:
    """Generate ``base/step-NNNNN`` directory path."""
    return Path(base) / f"step-{step:06d}"


# ---------------------------------------------------------------------------
# Relay export / import
# ---------------------------------------------------------------------------


def export_relay(
    checkpoint_dir: Path | str,
    output_dir: Path | str,
    description: str = "",
) -> Path:
    """Package a checkpoint for relay handoff.

    Copies checkpoint to output_dir and updates metadata with a relay description.
    """
    src = Path(checkpoint_dir)
    dst = Path(output_dir)

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    # Update meta with description
    meta_path = dst / "meta.json"
    meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_data["description"] = description
    meta_data["relay_exported_at"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(
        json.dumps(meta_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Relay checkpoint exported: %s → %s", src, dst)
    return dst


def import_relay(
    relay_dir: Path | str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> SnapshotMeta:
    """Import a relay checkpoint into model + optimizer."""
    return load_snapshot(relay_dir, model, optimizer, device)
