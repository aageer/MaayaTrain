"""Model catalog — registry of available architectures.

Usage::

    from maayatrain.architectures.catalog import create_model
    model = create_model("gpt2-small", vocab_size=256)
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from torch import nn

from .gpt2 import GPT2Config, GPT2Model, GPT2_CONFIGS

# Type for model factory functions
ModelFactory = Callable[..., nn.Module]

# Global registry
_REGISTRY: Dict[str, ModelFactory] = {}


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Decorator to register a model factory under a given name."""

    def wrapper(fn: ModelFactory) -> ModelFactory:
        _REGISTRY[name] = fn
        return fn

    return wrapper


def _gpt2_factory(config_name: str, vocab_size: int = 256, seq_length: int = 512) -> GPT2Model:
    """Create a GPT-2 model from a named config."""
    cfg = GPT2_CONFIGS[config_name]
    cfg = GPT2Config(
        vocab_size=vocab_size,
        seq_length=seq_length,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    return GPT2Model(cfg)


# Register all GPT-2 variants
for _name in GPT2_CONFIGS:
    _REGISTRY[_name] = lambda vs=256, sl=512, n=_name: _gpt2_factory(n, vs, sl)


def create_model(
    name: str,
    vocab_size: int = 256,
    seq_length: int = 512,
) -> nn.Module:
    """Create a model by name from the registry.

    Parameters
    ----------
    name : str
        Model name (e.g. ``"gpt2-small"``).
    vocab_size : int
        Vocabulary size (determined by dataset).
    seq_length : int
        Maximum sequence length.

    Returns
    -------
    nn.Module
        The instantiated model.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")

    return _REGISTRY[name](vocab_size, seq_length)


def list_models() -> list[str]:
    """Return all registered model names."""
    return sorted(_REGISTRY.keys())
