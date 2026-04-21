"""Tests for hardware detection."""

import torch

from maayatrain.hardware import DeviceProfile, detect_device


def test_detect_device_returns_profile():
    """detect_device() always returns a valid DeviceProfile."""
    profile = detect_device()
    assert isinstance(profile, DeviceProfile)
    assert isinstance(profile.device, torch.device)
    assert profile.backend in ("cuda", "mps", "xpu", "cpu")
    assert profile.memory_gb >= 0
    assert profile.compute_tflops >= 0
    assert len(profile.os_name) > 0
    assert len(profile.hostname) > 0


def test_summary_string():
    """summary() returns a human-readable string."""
    profile = detect_device()
    s = profile.summary()
    assert isinstance(s, str)
    assert "GB" in s
    assert "TFLOPS" in s


def test_device_is_usable():
    """The detected device can be used with PyTorch tensors."""
    profile = detect_device()
    t = torch.zeros(10, device=profile.device)
    assert t.device.type == profile.device.type
