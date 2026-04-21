"""Active peer roster — thread-safe registry of known peers."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RosterEntry:
    """One known peer."""

    peer_id: str
    address: str
    device_name: str = "unknown"
    memory_gb: float = 0.0
    compute_tflops: float = 0.0
    status: str = "connected"
    joined_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    contribution_hours: float = 0.0


class PeerRoster:
    """Thread-safe in-memory peer registry.

    Used by both the coordinator and dashboard to track who's in the cluster.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._peers: Dict[str, RosterEntry] = {}

    def add(self, entry: RosterEntry) -> None:
        with self._lock:
            self._peers[entry.peer_id] = entry

    def remove(self, peer_id: str) -> Optional[RosterEntry]:
        with self._lock:
            return self._peers.pop(peer_id, None)

    def touch(self, peer_id: str) -> None:
        """Update last_seen timestamp."""
        with self._lock:
            if peer_id in self._peers:
                self._peers[peer_id].last_seen = time.time()

    def get(self, peer_id: str) -> Optional[RosterEntry]:
        with self._lock:
            return self._peers.get(peer_id)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._peers)

    def all_entries(self) -> List[RosterEntry]:
        with self._lock:
            return list(self._peers.values())

    def total_tflops(self) -> float:
        with self._lock:
            return sum(e.compute_tflops for e in self._peers.values())

    def prune_stale(self, stale_seconds: float = 30.0) -> List[str]:
        """Remove peers not seen for *stale_seconds*. Returns removed IDs."""
        now = time.time()
        removed = []
        with self._lock:
            for pid in list(self._peers.keys()):
                if now - self._peers[pid].last_seen > stale_seconds:
                    del self._peers[pid]
                    removed.append(pid)
        return removed
