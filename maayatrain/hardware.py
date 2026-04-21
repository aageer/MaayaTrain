"""Cross-platform GPU and hardware detection for MaayaTrain.

Probes the system to find the best available compute device (CUDA, ROCm, MPS, XPU, or CPU)
and reports hardware capabilities for peer advertisement.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(frozen=True)
class DeviceProfile:
    """Immutable snapshot of the local compute device."""

    device: torch.device
    backend: str  # "cuda", "mps", "xpu", "cpu"
    device_name: str  # e.g. "NVIDIA RTX 4090", "Apple M4 Pro", "AMD Instinct MI300X"
    memory_gb: float  # total VRAM or system RAM available to the device
    compute_tflops: float  # estimated peak FP32 TFLOPS
    os_name: str  # "Darwin", "Linux", "Windows"
    hostname: str
    extras: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.device_name} | {self.memory_gb:.1f} GB | "
            f"~{self.compute_tflops:.1f} TFLOPS | {self.backend.upper()} | {self.os_name}"
        )


# ---------------------------------------------------------------------------
# Estimated TFLOPS lookup (FP32, conservative).  These are rough figures used
# only for display / peer advertisement, not for scheduling decisions.
# ---------------------------------------------------------------------------

_APPLE_CHIP_TFLOPS: dict[str, float] = {
    "Apple M1": 2.6,
    "Apple M1 Pro": 5.3,
    "Apple M1 Max": 10.6,
    "Apple M1 Ultra": 21.2,
    "Apple M2": 3.6,
    "Apple M2 Pro": 6.8,
    "Apple M2 Max": 13.6,
    "Apple M2 Ultra": 27.2,
    "Apple M3": 4.1,
    "Apple M3 Pro": 7.4,
    "Apple M3 Max": 14.2,
    "Apple M3 Ultra": 28.4,
    "Apple M4": 4.6,
    "Apple M4 Pro": 8.7,
    "Apple M4 Max": 18.0,
    "Apple M5": 5.2,
    "Apple M5 Pro": 10.0,
    "Apple M5 Max": 20.0,
}


def _detect_apple_chip() -> Optional[str]:
    """Return the Apple Silicon chip name via sysctl, or None."""
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _system_ram_gb() -> float:
    """Return total system RAM in GB (platform-independent)."""
    try:
        import os

        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024**3)
        # Windows fallback
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / (1024**3)
    except Exception:
        return 0.0


def _estimate_tflops_cuda(device_idx: int) -> float:
    """Rough FP32 TFLOPS estimate from CUDA device properties."""
    props = torch.cuda.get_device_properties(device_idx)
    # TFLOPS ≈ cores × clock_MHz × 2 (FMA) / 1e6
    return (props.multi_processor_count * 128 * props.clock_rate * 2) / 1e9


def detect_device() -> DeviceProfile:
    """Detect the best available compute device and return a DeviceProfile."""
    os_name = platform.system()
    hostname = platform.node()

    # --- CUDA (NVIDIA) -------------------------------------------------
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        mem_gb = props.total_mem / (1024**3)
        tflops = _estimate_tflops_cuda(idx)
        return DeviceProfile(
            device=torch.device("cuda", idx),
            backend="cuda",
            device_name=name,
            memory_gb=mem_gb,
            compute_tflops=tflops,
            os_name=os_name,
            hostname=hostname,
            extras={"cuda_capability": f"{props.major}.{props.minor}"},
        )

    # --- MPS (Apple Silicon) -------------------------------------------
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        chip = _detect_apple_chip() or "Apple Silicon"
        mem_gb = _system_ram_gb()  # MPS shares unified memory
        tflops = 0.0
        for prefix, val in _APPLE_CHIP_TFLOPS.items():
            if chip.startswith(prefix):
                tflops = val
                break
        return DeviceProfile(
            device=torch.device("mps"),
            backend="mps",
            device_name=chip,
            memory_gb=mem_gb,
            compute_tflops=tflops or 3.0,
            os_name=os_name,
            hostname=hostname,
        )

    # --- XPU (Intel Arc / Data Center) ---------------------------------
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        idx = torch.xpu.current_device()
        name = torch.xpu.get_device_name(idx)
        props = torch.xpu.get_device_properties(idx)
        mem_gb = getattr(props, "total_memory", 0) / (1024**3)
        return DeviceProfile(
            device=torch.device("xpu", idx),
            backend="xpu",
            device_name=name,
            memory_gb=mem_gb or _system_ram_gb(),
            compute_tflops=2.0,  # conservative default
            os_name=os_name,
            hostname=hostname,
        )

    # --- CPU fallback --------------------------------------------------
    cpu_name = platform.processor() or "CPU"
    return DeviceProfile(
        device=torch.device("cpu"),
        backend="cpu",
        device_name=cpu_name,
        memory_gb=_system_ram_gb(),
        compute_tflops=0.1,  # CPU training is slow
        os_name=os_name,
        hostname=hostname,
    )
