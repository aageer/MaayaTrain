"""Tests for block-wise INT8 quantized compression."""

import torch

from maayatrain.comms.tensor_codec import (
    compress,
    compression_tag,
    decompress,
    _quantize_int8,
    _dequantize_int8,
    _BLOCK_SIZE,
)


def test_int8_quantize_roundtrip():
    """Block-wise INT8 quantize → dequantize preserves approximate values."""
    t = torch.randn(100, 100)
    q, scales, mins, pad_len, orig_shape = _quantize_int8(t)

    assert q.dtype == torch.int8
    assert scales.dtype == torch.float16
    assert mins.dtype == torch.float16
    assert orig_shape == [100, 100]

    # Number of blocks = ceil(10000 / 128) = 79  (79 * 128 = 10112)
    expected_blocks = (10000 + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    assert scales.shape == (expected_blocks,)
    assert mins.shape == (expected_blocks,)
    assert q.numel() == expected_blocks * _BLOCK_SIZE

    restored = _dequantize_int8(q, scales, mins, pad_len, orig_shape)
    assert restored.dtype == torch.float32
    assert restored.shape == t.shape

    # Block-wise quantization should have much better accuracy than per-tensor
    max_error = (restored - t).abs().max().item()
    t_range = t.max().item() - t.min().item()
    relative_err = max_error / max(t_range, 1e-6)
    assert relative_err < 0.02, f"Block-wise INT8 relative error too high: {relative_err:.4f}"


def test_int8_block_structure():
    """Verify block-wise quantization produces per-block scales."""
    # Create a tensor with very different value ranges in different regions
    t = torch.zeros(256)
    t[:128] = torch.randn(128) * 0.01   # tiny values
    t[128:] = torch.randn(128) * 1000.0  # huge values

    q, scales, mins, pad_len, orig_shape = _quantize_int8(t)

    # Should have 2 blocks with very different scales
    assert scales.numel() == 2
    assert scales[0].item() != scales[1].item()

    # The small-value block should have a much smaller scale
    assert scales[0].float().item() < scales[1].float().item()


def test_int8_outlier_resilience():
    """Block-wise quantization should handle outliers much better than per-tensor."""
    # 99% normal values, 1% extreme outliers in a separate block region
    t = torch.randn(1280)  # 10 blocks of 128
    t[0] = 10000.0  # single outlier in block 0

    q, scales, mins, pad_len, orig_shape = _quantize_int8(t)
    restored = _dequantize_int8(q, scales, mins, pad_len, orig_shape)

    # Non-outlier blocks (blocks 1-9) should have excellent accuracy
    non_outlier = slice(128, 1280)
    max_err_clean = (restored[non_outlier] - t[non_outlier]).abs().max().item()
    clean_range = t[non_outlier].max().item() - t[non_outlier].min().item()
    rel_err_clean = max_err_clean / max(clean_range, 1e-6)
    assert rel_err_clean < 0.02, (
        f"Non-outlier blocks too inaccurate: {rel_err_clean:.4f}"
    )


def test_int8_compress_decompress():
    """Full block-wise INT8 compress/decompress pipeline works."""
    tensors = {
        "weight": torch.randn(64, 128),
        "bias": torch.randn(128),
    }

    blob = compress(tensors, use_int8=True)
    recovered = decompress(blob, is_int8=True)

    assert set(recovered.keys()) == set(tensors.keys())
    for name in tensors:
        assert recovered[name].shape == tensors[name].shape
        assert torch.allclose(recovered[name], tensors[name], atol=0.05)


def test_int8_compression_ratio():
    """INT8 should achieve better compression than FP16."""
    tensors = {"gradient": torch.randn(1000, 1000)}
    raw_size = tensors["gradient"].numel() * 4  # FP32 bytes

    fp16_blob = compress(tensors, use_fp16=True, use_int8=False)
    int8_blob = compress(tensors, use_int8=True)

    fp16_ratio = raw_size / len(fp16_blob)
    int8_ratio = raw_size / len(int8_blob)

    # INT8 should compress more than FP16
    assert int8_ratio > fp16_ratio, f"INT8 ({int8_ratio:.1f}x) not better than FP16 ({fp16_ratio:.1f}x)"


def test_compression_tag_int8():
    assert compression_tag(use_int8=True) == "int8_zstd"
    assert compression_tag(use_fp16=True, use_int8=False) == "fp16_zstd"
    assert compression_tag(use_fp16=False, use_int8=False) == "zstd"


def test_int8_constant_tensor():
    """Constant tensors (zero range) don't crash with block-wise quantization."""
    t = torch.ones(256) * 42.0
    q, scales, mins, pad_len, orig_shape = _quantize_int8(t)
    restored = _dequantize_int8(q, scales, mins, pad_len, orig_shape)
    # All values should be approximately 42
    assert torch.allclose(restored, t, atol=0.1)


def test_int8_small_tensor():
    """Tensors smaller than one block are handled correctly."""
    t = torch.randn(50)  # Less than 128
    q, scales, mins, pad_len, orig_shape = _quantize_int8(t)

    assert orig_shape == [50]
    assert pad_len == 128 - 50  # Padded to 128
    assert scales.numel() == 1  # Single block

    restored = _dequantize_int8(q, scales, mins, pad_len, orig_shape)
    assert restored.shape == t.shape
    assert torch.allclose(restored, t, atol=0.05)


def test_int8_exact_block_multiple():
    """Tensors that are exact multiples of block_size need no padding."""
    t = torch.randn(256)  # Exactly 2 blocks
    q, scales, mins, pad_len, orig_shape = _quantize_int8(t)

    assert pad_len == 0
    assert scales.numel() == 2

    restored = _dequantize_int8(q, scales, mins, pad_len, orig_shape)
    assert restored.shape == t.shape


def test_backward_compatibility():
    """FP16 compress/decompress still works with updated codec."""
    tensors = {"param": torch.randn(50, 50)}
    blob = compress(tensors, use_fp16=True, use_int8=False)
    recovered = decompress(blob, restore_fp32=True, is_int8=False)
    assert torch.allclose(recovered["param"], tensors["param"], atol=0.01)
