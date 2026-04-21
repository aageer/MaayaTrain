"""Tests for tensor compression codec."""

import torch

from maayatrain.comms.tensor_codec import compress, compression_tag, decompress


def test_roundtrip_fp16_zstd():
    """Compress and decompress restores tensor shapes and approximate values."""
    tensors = {
        "weight": torch.randn(128, 256),
        "bias": torch.randn(256),
        "large": torch.randn(1024, 512),
    }

    blob = compress(tensors, use_fp16=True)
    recovered = decompress(blob, restore_fp32=True)

    assert set(recovered.keys()) == set(tensors.keys())
    for name in tensors:
        assert recovered[name].shape == tensors[name].shape
        assert recovered[name].dtype == torch.float32
        # FP16 roundtrip introduces small errors
        assert torch.allclose(recovered[name], tensors[name], atol=0.01)


def test_roundtrip_fp32_zstd():
    """Without FP16 casting, values are exactly preserved."""
    tensors = {"param": torch.randn(64, 64)}

    blob = compress(tensors, use_fp16=False)
    recovered = decompress(blob, restore_fp32=False)

    assert torch.equal(recovered["param"], tensors["param"].cpu())


def test_compression_ratio():
    """FP16+zstd should achieve meaningful compression."""
    tensors = {"gradient": torch.randn(1000, 1000)}
    raw_size = tensors["gradient"].numel() * 4  # FP32 = 4 bytes
    blob = compress(tensors, use_fp16=True)

    ratio = raw_size / len(blob)
    # Should be at least 1.5x compression (FP16 alone = 2x)
    assert ratio > 1.5, f"Compression ratio too low: {ratio:.2f}x"


def test_compression_tag():
    assert compression_tag(use_fp16=True, use_int8=False) == "fp16_zstd"
    assert compression_tag(use_fp16=False, use_int8=False) == "zstd"
    assert compression_tag(use_int8=True) == "int8_zstd"


def test_empty_tensors():
    """Handle dict with no tensors."""
    blob = compress({})
    recovered = decompress(blob)
    assert recovered == {}
