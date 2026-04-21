"""Cluster state tracking for MaayaTrain."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PeerInfo:
    peer_id: str
    device_name: str = "unknown"
    memory_gb: float = 0.0
    compute_tflops: float = 0.0
    joined_at: float = field(default_factory=time.time)
    status: str = "connected"


class ClusterState:
    """Tracks the current state of the training cluster."""

    def __init__(self, coordinator_id: str) -> None:
        self.coordinator_id = coordinator_id
        self.peers: Dict[str, PeerInfo] = {}

    def add_peer(self, peer_id: str, handshake_header: Dict[str, Any]) -> None:
        info = PeerInfo(
            peer_id=peer_id,
            device_name=handshake_header.get("device_name", "unknown"),
            memory_gb=handshake_header.get("memory_gb", 0.0),
            compute_tflops=handshake_header.get("compute_tflops", 0.0),
        )
        self.peers[peer_id] = info

    def remove_peer(self, peer_id: str) -> None:
        self.peers.pop(peer_id, None)

    @property
    def peer_count(self) -> int:
        return len(self.peers)

    @property
    def total_tflops(self) -> float:
        return sum(p.compute_tflops for p in self.peers.values())

    def to_dict(self, global_step: int = 0, loss: float = 0.0) -> Dict[str, Any]:
        return {
            "coordinator_id": self.coordinator_id,
            "peer_count": self.peer_count,
            "total_tflops": self.total_tflops,
            "global_step": global_step,
            "loss": loss,
            "peers": [
                {
                    "peer_id": p.peer_id,
                    "device_name": p.device_name,
                    "memory_gb": p.memory_gb,
                    "compute_tflops": p.compute_tflops,
                    "status": p.status,
                }
                for p in self.peers.values()
            ],
        }
