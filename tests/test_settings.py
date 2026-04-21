"""Tests for settings loader."""

from pathlib import Path

from maayatrain.settings import MaayaTrainSettings, load_settings


def test_default_settings():
    """Default settings match DiLoCo paper values."""
    s = MaayaTrainSettings()
    assert s.diloco.inner_steps == 500
    assert s.diloco.inner_lr == 3e-4
    assert s.diloco.outer_lr == 0.7
    assert s.diloco.outer_momentum == 0.9
    assert s.diloco.nesterov is True
    assert s.diloco.inner_weight_decay == 0.1
    assert s.model.name == "gpt2-small"
    assert s.network.port == 7471
    assert s.dashboard.port == 8471


def test_load_from_toml(tmp_path: Path):
    """Load settings from a TOML file."""
    toml_content = """
[model]
name = "gpt2-medium"

[training]
batch_size = 16
max_steps = 50000

[diloco]
inner_steps = 250
outer_lr = 0.5
"""
    config_file = tmp_path / "maayatrain.toml"
    config_file.write_text(toml_content, encoding="utf-8")

    s = load_settings(config_file)
    assert s.model.name == "gpt2-medium"
    assert s.training.batch_size == 16
    assert s.training.max_steps == 50000
    assert s.diloco.inner_steps == 250
    assert s.diloco.outer_lr == 0.5
    # Unspecified values keep defaults
    assert s.diloco.outer_momentum == 0.9
    assert s.network.port == 7471


def test_load_missing_file():
    """Missing config file returns defaults."""
    s = load_settings(Path("/nonexistent/config.toml"))
    assert s.model.name == "gpt2-small"


def test_validation():
    """Pydantic validates field constraints."""
    import pytest

    with pytest.raises(Exception):
        MaayaTrainSettings(diloco={"inner_steps": -1})

    with pytest.raises(Exception):
        MaayaTrainSettings(network={"port": 999999})
