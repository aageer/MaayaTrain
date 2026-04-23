"""Pydantic v2 settings for MaayaTrain.

Loads configuration from ``maayatrain.toml`` (project root or working directory),
environment variables prefixed ``MAAYA_``, and CLI flag overrides.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        import tomllib  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Which model architecture to train."""

    name: str = Field(default="gpt2-small", description="Model name from the catalog")


class DatasetConfig(BaseModel):
    """Dataset paths and tokenisation settings."""

    path: str = Field(default="./data/wikitext.txt", description="Path to training text file")
    seq_length: int = Field(default=512, ge=32, le=8192, description="Sequence length in tokens")


class TrainingConfig(BaseModel):
    """General training hyper-parameters."""

    batch_size: int = Field(default=8, ge=1, description="Per-device micro-batch size")
    max_steps: int = Field(default=100_000, ge=1, description="Total training steps")
    checkpoint_dir: str = Field(default="./checkpoints", description="Where to save checkpoints")
    checkpoint_every: int = Field(default=1000, ge=1, description="Save a checkpoint every N steps")
    log_every: int = Field(default=10, ge=1, description="Log metrics every N steps")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    mixed_precision: Literal["off", "fp16", "bf16", "auto"] = Field(
        default="auto", description="AMP precision: auto detects best for your GPU"
    )
    gradient_accumulation_steps: int = Field(
        default=1, ge=1, description="Accumulate gradients over N micro-batches"
    )
    warmup_steps: int = Field(
        default=0, ge=0, description="Linear warmup steps (0 = auto 10%% of inner_steps)"
    )
    min_lr_ratio: float = Field(
        default=0.1, ge=0, le=1, description="Minimum LR as ratio of peak (for cosine decay)"
    )
    max_grad_norm: float = Field(default=1.0, gt=0, description="Gradient clipping max norm")


class DiLoCoConfig(BaseModel):
    """DiLoCo algorithm parameters (arXiv:2311.08105).

    inner_steps (H):  Number of local training steps between global syncs.
    inner_lr (α):     Learning rate for the inner AdamW optimizer.
    outer_lr (η):     Learning rate for the outer Nesterov SGD step.
    outer_momentum (β): Momentum coefficient for the outer optimizer.
    """

    inner_steps: int = Field(default=500, ge=1, description="Local steps (H) before sync")
    inner_lr: float = Field(default=3e-4, gt=0, description="Inner AdamW learning rate")
    inner_optimizer: Literal["adamw"] = Field(default="adamw")
    inner_weight_decay: float = Field(default=0.1, ge=0, description="AdamW weight decay")
    outer_lr: float = Field(default=0.7, gt=0, description="Outer Nesterov SGD learning rate")
    outer_momentum: float = Field(default=0.9, ge=0, le=1, description="Outer momentum (β)")
    nesterov: bool = Field(default=True, description="Use Nesterov momentum in outer step")
    gradient_compression: bool = Field(default=True, description="Compress pseudo-gradients")
    compress_fp16: bool = Field(default=True, description="Cast to FP16 before compression")
    compress_int8: bool = Field(default=False, description="Use INT8 quantization (even smaller)")
    streaming_shards: int = Field(
        default=1, ge=1, description="Split params into N shards for streaming sync (1=off)"
    )
    aggregation: Literal["mean", "median"] = Field(
        default="mean", description="Pseudo-gradient aggregation: mean or median (Byzantine-tolerant)"
    )
    sync_mode: Literal["steps", "time"] = Field(
        default="steps",
        description=(
            "Inner loop sync mode: 'steps' = fixed H steps (standard DiLoCo), "
            "'time' = run as many steps as possible within sync_window_seconds"
        ),
    )
    sync_window_seconds: float = Field(
        default=60.0, gt=0, description="Seconds per inner-loop window when sync_mode='time'"
    )


class NetworkConfig(BaseModel):
    """TCP transport settings."""

    port: int = Field(default=7471, ge=1024, le=65535, description="TCP port for peer comms")
    heartbeat_interval: int = Field(default=15, ge=1, description="Heartbeat period in seconds")
    bind_address: str = Field(default="0.0.0.0", description="Address to bind the TCP server")


class DashboardConfig(BaseModel):
    """Local monitoring dashboard."""

    port: int = Field(default=8471, ge=1024, le=65535, description="Dashboard HTTP port")
    enabled: bool = Field(default=False, description="Start dashboard with training")


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------


class MaayaTrainSettings(BaseModel):
    """Root configuration assembled from TOML file + overrides."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    diloco: DiLoCoConfig = Field(default_factory=DiLoCoConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_CONFIG_FILENAMES = ("maayatrain.toml", "MaayaTrain.toml")


def _find_config(start: Optional[Path] = None) -> Optional[Path]:
    """Walk up from *start* (default: cwd) looking for a config file."""
    directory = (start or Path.cwd()).resolve()
    for _ in range(20):  # safety limit
        for name in _CONFIG_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        parent = directory.parent
        if parent == directory:
            break
        directory = parent
    return None


def load_settings(config_path: Optional[Path] = None) -> MaayaTrainSettings:
    """Load settings from TOML, falling back to defaults if no file is found."""
    path = config_path or _find_config()
    if path is not None and path.is_file():
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode())
        return MaayaTrainSettings.model_validate(data)
    return MaayaTrainSettings()
