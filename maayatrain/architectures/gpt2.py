"""GPT-2 transformer implemented in PyTorch.

This is an independent implementation based on the publicly documented
GPT-2 architecture (Radford et al., 2019). Runs on any PyTorch-supported
device: CUDA, MPS, XPU, or CPU.

Supported configurations:
    gpt2-small   — 124M params (12 layers, 768 dim, 12 heads)
    gpt2-medium  — 355M params (24 layers, 1024 dim, 16 heads)
    gpt2-large   — 774M params (36 layers, 1280 dim, 20 heads)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("maayatrain.gpt2")


@dataclass
class GPT2Config:
    """GPT-2 model configuration."""

    vocab_size: int = 256  # character-level default (overridden by dataset)
    seq_length: int = 512
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072  # 4 * d_model
    dropout: float = 0.1
    bias: bool = True


# Pre-defined configurations
GPT2_CONFIGS = {
    "gpt2-tiny": GPT2Config(n_layers=4, n_heads=4, d_model=256, d_ff=1024),
    "gpt2-small": GPT2Config(n_layers=12, n_heads=12, d_model=768, d_ff=3072),
    "gpt2-medium": GPT2Config(n_layers=24, n_heads=16, d_model=1024, d_ff=4096),
    "gpt2-large": GPT2Config(n_layers=36, n_heads=20, d_model=1280, d_ff=5120),
}


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention.

    Uses ``F.scaled_dot_product_attention`` (PyTorch ≥ 2.0) to
    automatically dispatch to FlashAttention-2, xFormers memory-
    efficient attention, or the math fallback depending on hardware.
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        # Combined Q/K/V projection
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Native scaled dot-product attention with causal mask.
        # Automatically selects FlashAttention-2 on CUDA Ampere+,
        # memory-efficient attention on other GPUs, or math fallback
        # on MPS/CPU.
        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm → Attention → LayerNorm → MLP."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Model(nn.Module):
    """GPT-2 language model.

    Architecture: token_embed + pos_embed → N × TransformerBlock → LayerNorm → LM head

    The LM head shares weights with the token embedding (weight tying).
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.seq_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)

        # LM head (weight-tied with token_embed)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # weight tying

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: Tensor) -> Tensor:
        """
        Parameters
        ----------
        idx : Tensor
            Token indices, shape (batch, seq_length).

        Returns
        -------
        Tensor
            Logits, shape (batch, seq_length, vocab_size).
        """
        B, T = idx.shape
        assert T <= self.config.seq_length, f"Sequence length {T} exceeds max {self.config.seq_length}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        tok_emb = self.token_embed(idx)  # (B, T, d_model)
        pos_emb = self.pos_embed(pos)  # (T, d_model)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_parameters_millions(self) -> float:
        return self.num_parameters / 1e6


# ---------------------------------------------------------------------------
# torch.compile wrapper (graceful fallback for MPS / older PyTorch)
# ---------------------------------------------------------------------------


def try_compile(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
    """Attempt to compile the model with ``torch.compile``.

    Falls back to eager mode on platforms that don't support compilation
    (e.g. Apple MPS, PyTorch < 2.0, or Windows without triton).

    Parameters
    ----------
    model : nn.Module
        The model to compile.
    mode : str
        Compilation mode (``"reduce-overhead"`` or ``"max-autotune"``).

    Returns
    -------
    nn.Module
        The compiled model, or the original model if compilation fails.
    """
    if not hasattr(torch, "compile"):
        logger.info("torch.compile not available (PyTorch < 2.0) — using eager mode")
        return model

    try:
        compiled = torch.compile(model, mode=mode)
        logger.info("Model compiled with torch.compile(mode=%r)", mode)
        return compiled
    except Exception as exc:
        logger.warning("torch.compile failed (%s) — falling back to eager mode", exc)
        return model
