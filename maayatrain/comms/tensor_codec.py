"""Tensor compression codec for MaayaTrain gradient exchange.

Pseudo-gradients are large (124M params × 4 bytes = ~500 MB in FP32).
This codec applies staged compression:

1. **Precision reduction** — cast to FP16 (~2×) or INT8 (~4×).
2. **zstd compression** — faster than gzip with comparable ratios,
   using multi-threaded compression for large payloads.

Compression modes and ratios:
- FP16 + zstd: ~4–6× (default — safe for gradient averages)
- INT8 + zstd: ~8–12× (aggressive — block-wise quantization)
- zstd only:   ~2–3× (lossless for the given precision)

INT8 quantization uses **block-wise** (block_size=128) affine scaling
rather than per-tensor scaling. This preserves gradient quality in the
presence of LLM outlier features by computing independent scale/zero_point
for each 128-element block.

Based on:
- DiLoCo (arXiv:2311.08105): FP16 pseudo-gradients preserve convergence
- Streaming DiLoCo (arXiv:2501.18512): 4-bit quantization is viable
- Block-wise quantization: independent implementation inspired by
  bitsandbytes/QLoRA block-wise design (Dettmers et al., 2023)

Author: Akhil Ageer — MaayaTrain project
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Tuple

import torch
import zstandard as zstd
from torch import Tensor

# Shared compressor / decompressor instances (thread-safe in zstandard)
_COMPRESSOR = zstd.ZstdCompressor(level=3, threads=-1)
_DECOMPRESSOR = zstd.ZstdDecompressor()

# Block size for block-wise INT8 quantization.
# 128 elements ≈ 512 bytes in FP32. Small enough to track outlier
# features, large enough to amortize the per-block overhead (2 × FP16
# scale params per block).
_BLOCK_SIZE = 128


# ---------------------------------------------------------------------------
# Block-wise INT8 quantization (independent implementation)
# ---------------------------------------------------------------------------


def _quantize_int8(
    tensor: Tensor,
    block_size: int = _BLOCK_SIZE,
) -> Tuple[Tensor, Tensor, Tensor, int, List[int]]:
    """Quantize a floating-point tensor to INT8 using block-wise scaling.

    The tensor is flattened and split into contiguous blocks of
    *block_size* elements. Each block gets its own (scale, x_min) pair,
    so outlier values in one block don't destroy precision everywhere.

    Steps:
    1. Flatten → pad to multiple of *block_size* → view as (-1, block_size).
    2. Per-block: x_min, x_max along dim=1.
    3. scale = (x_max - x_min) / 254,  clamped ≥ 1e-8.
    4. q = round((block - x_min) / scale) - 127, cast to int8.

    Parameters
    ----------
    tensor : Tensor
        Any floating-point tensor.
    block_size : int
        Elements per quantization block (default 128).

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, int, list[int]]
        (q_int8, scales_fp16, mins_fp16, pad_len, original_shape)
    """
    original_shape = list(tensor.shape)
    t = tensor.detach().float().reshape(-1)
    numel = t.numel()

    # Pad to multiple of block_size
    pad_len = (block_size - numel % block_size) % block_size
    if pad_len > 0:
        t = torch.nn.functional.pad(t, (0, pad_len), value=0.0)

    # View as (n_blocks, block_size)
    blocks = t.view(-1, block_size)

    # Per-block statistics
    x_min = blocks.min(dim=1, keepdim=True).values
    x_max = blocks.max(dim=1, keepdim=True).values
    s = ((x_max - x_min) / 254.0).clamp(min=1e-8)

    # Quantize: map [x_min, x_max] → [-127, 127]
    q = torch.round((blocks - x_min) / s) - 127.0
    q = q.clamp(-127, 127).to(torch.int8)

    # Store scales and mins as FP16 to save space (2 bytes vs 4 per block)
    return (
        q.reshape(-1),           # flat int8
        s.squeeze(1).half(),     # (n_blocks,) fp16
        x_min.squeeze(1).half(), # (n_blocks,) fp16
        pad_len,
        original_shape,
    )


def _dequantize_int8(
    quantized: Tensor,
    scales: Tensor,
    mins: Tensor,
    pad_len: int,
    original_shape: List[int],
    block_size: int = _BLOCK_SIZE,
) -> Tensor:
    """Restore a floating-point tensor from block-wise INT8 quantized form.

    Reverses :func:`_quantize_int8`: unflatten → per-block de-scale →
    remove padding → reshape to original_shape.

    Parameters
    ----------
    quantized : Tensor
        Flat INT8 tensor from :func:`_quantize_int8`.
    scales : Tensor
        Per-block scale factors (n_blocks,).
    mins : Tensor
        Per-block minimum values (n_blocks,).
    pad_len : int
        Number of padding elements appended during quantization.
    original_shape : list[int]
        Original tensor shape before flatten + pad.
    block_size : int
        Must match the block_size used during quantization.

    Returns
    -------
    Tensor
        Restored FP32 tensor with the original shape.
    """
    # Reshape to (n_blocks, block_size)
    blocks = quantized.float().view(-1, block_size)
    s = scales.float().unsqueeze(1)       # (n_blocks, 1)
    x_min = mins.float().unsqueeze(1)     # (n_blocks, 1)

    # Dequantize: reverse of q = round((x - x_min) / s) - 127
    restored = (blocks + 127.0) * s + x_min

    # Flatten, strip padding, reshape
    flat = restored.reshape(-1)
    if pad_len > 0:
        flat = flat[: flat.numel() - pad_len]

    return flat.reshape(original_shape)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compress(
    tensors: Dict[str, Tensor],
    *,
    use_fp16: bool = True,
    use_int8: bool = False,
    gzip_level: int = 6,  # kept for API compat, ignored
) -> bytes:
    """Compress a dict of named tensors into a zstd-compressed byte blob.

    Parameters
    ----------
    tensors : dict[str, Tensor]
        Named tensors (e.g. model state_dict diff).
    use_fp16 : bool
        If True, cast each tensor to float16 before serialization.
        Ignored if use_int8 is True.
    use_int8 : bool
        If True, quantize each tensor to INT8 (more aggressive, ~8× total).
        Takes precedence over use_fp16.
    gzip_level : int
        Deprecated — kept for API backward compatibility, ignored internally.

    Returns
    -------
    bytes
        The compressed blob, ready to embed as a wire-format payload.
    """
    if use_int8:
        return _compress_int8(tensors)

    prepared: Dict[str, Tensor] = {}
    for name, tensor in tensors.items():
        t = tensor.detach().cpu()
        if use_fp16 and t.is_floating_point():
            t = t.half()
        prepared[name] = t

    buf = io.BytesIO()
    torch.save(prepared, buf)
    raw = buf.getvalue()

    return _COMPRESSOR.compress(raw)


def _compress_int8(tensors: Dict[str, Tensor]) -> bytes:
    """Block-wise INT8 quantization path — higher compression, good quality.

    Serializes each tensor's quantized data (q, scales, mins, pad_len,
    original_shape) via ``torch.save`` into a single ``io.BytesIO`` buffer,
    then zstd-compresses the result.
    """
    payload: Dict[str, Any] = {}
    for name, tensor in tensors.items():
        t = tensor.detach().cpu()
        if t.is_floating_point():
            q, scales, mins, pad_len, orig_shape = _quantize_int8(t)
            payload[name] = {
                "q": q,
                "s": scales,
                "m": mins,
                "p": pad_len,
                "shape": orig_shape,
            }
        else:
            payload[name] = {"raw": t}

    buf = io.BytesIO()
    torch.save(payload, buf)
    return _COMPRESSOR.compress(buf.getvalue())


def decompress(
    data: bytes,
    *,
    restore_fp32: bool = True,
    map_location: str = "cpu",
    is_int8: bool = False,
) -> Dict[str, Tensor]:
    """Decompress a blob back into a dict of named tensors.

    Parameters
    ----------
    data : bytes
        A blob previously created by :func:`compress`.
    restore_fp32 : bool
        If True, cast all floating-point tensors back to float32.
    map_location : str
        Device to map tensors onto (default ``"cpu"``).
    is_int8 : bool
        If True, expect block-wise INT8-quantized format.

    Returns
    -------
    dict[str, Tensor]
    """
    raw = _DECOMPRESSOR.decompress(data)
    buf = io.BytesIO(raw)
    loaded = torch.load(buf, map_location=map_location, weights_only=False)

    if is_int8:
        result: Dict[str, Tensor] = {}
        for name, entry in loaded.items():
            if isinstance(entry, dict) and "q" in entry:
                result[name] = _dequantize_int8(
                    entry["q"],
                    entry["s"],
                    entry["m"],
                    entry["p"],
                    entry["shape"],
                )
            elif isinstance(entry, dict) and "raw" in entry:
                result[name] = entry["raw"]
            else:
                result[name] = entry
        return result

    if restore_fp32:
        for name in loaded:
            if loaded[name].is_floating_point():
                loaded[name] = loaded[name].float()

    return loaded


def compression_tag(use_fp16: bool = True, use_int8: bool = False) -> str:
    """Return the compression descriptor string for wire-format headers."""
    if use_int8:
        return "int8_zstd"
    return "fp16_zstd" if use_fp16 else "zstd"
