"""Zero-config LAN peer discovery via mDNS/Bonjour.

Uses the `python-zeroconf` library (pure Python, cross-platform —
works on Windows, Linux, and macOS) to advertise and browse for
MaayaTrain coordinator nodes on the local network.

Service type: ``_maayatrain._tcp.local.``
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from zeroconf import ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf

logger = logging.getLogger("maayatrain.discovery")

SERVICE_TYPE = "_maayatrain._tcp.local."


@dataclass
class DiscoveredPeer:
    """A coordinator discovered on the LAN."""

    name: str
    host: str
    port: int
    model: str
    device: str
    memory_gb: float
    status: str
    discovered_at: float


class ZeroconfAdvertiser:
    """Advertise this node (coordinator) on the LAN via mDNS.

    Usage::

        adv = ZeroconfAdvertiser(port=7471, model="gpt2-small", device="Apple M4 Pro", memory_gb=48)
        adv.start()    # begin advertising
        ...
        adv.stop()     # clean up
    """

    def __init__(
        self,
        port: int = 7471,
        model: str = "unknown",
        device: str = "unknown",
        memory_gb: float = 0.0,
        hostname: str = "",
    ) -> None:
        self.port = port
        self.model = model
        self.device_name = device
        self.memory_gb = memory_gb
        self.hostname = hostname or socket.gethostname()

        self._zc: Optional[Zeroconf] = None
        self._info: Optional[ServiceInfo] = None

    def start(self) -> None:
        """Register the mDNS service."""
        self._zc = Zeroconf()

        # Get local IP address
        local_ip = self._get_local_ip()

        self._info = ServiceInfo(
            SERVICE_TYPE,
            f"coordinator.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(local_ip)],
            port=self.port,
            properties={
                "model": self.model,
                "device": self.device_name,
                "memory_gb": str(self.memory_gb),
                "status": "training",
                "hostname": self.hostname,
            },
        )
        self._zc.register_service(self._info)
        logger.info(
            "mDNS service registered: %s on %s:%d (model=%s)",
            SERVICE_TYPE,
            local_ip,
            self.port,
            self.model,
        )

    def stop(self) -> None:
        """Unregister and close."""
        if self._zc and self._info:
            self._zc.unregister_service(self._info)
            self._zc.close()
            logger.info("mDNS service unregistered")

    @staticmethod
    def _get_local_ip() -> str:
        """Determine the local LAN IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"


class ZeroconfBrowser:
    """Browse for MaayaTrain coordinators on the LAN.

    Usage::

        browser = ZeroconfBrowser()
        browser.on_found = lambda peer: print(f"Found: {peer}")
        browser.start()
        ...
        browser.stop()
    """

    def __init__(self) -> None:
        self.discovered: Dict[str, DiscoveredPeer] = {}
        self.on_found: Optional[Callable[[DiscoveredPeer], None]] = None
        self.on_removed: Optional[Callable[[str], None]] = None

        self._zc: Optional[Zeroconf] = None
        self._browser: Optional[ServiceBrowser] = None

    def start(self) -> None:
        """Start browsing for coordinators."""
        self._zc = Zeroconf()
        self._browser = ServiceBrowser(self._zc, SERVICE_TYPE, handlers=[self._on_change])
        logger.info("Browsing for %s services…", SERVICE_TYPE)

    def stop(self) -> None:
        """Stop browsing."""
        if self._zc:
            self._zc.close()
            logger.info("mDNS browser stopped")

    def wait_for_coordinator(self, timeout: float = 30.0) -> Optional[DiscoveredPeer]:
        """Block until a coordinator is found or *timeout* elapses."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.discovered:
                return next(iter(self.discovered.values()))
            time.sleep(0.5)
        return None

    def _on_change(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        if state_change == ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info:
                addresses = info.parsed_addresses()
                host = addresses[0] if addresses else "unknown"
                props = {
                    k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
                    for k, v in (info.properties or {}).items()
                }

                peer = DiscoveredPeer(
                    name=name,
                    host=host,
                    port=info.port or 7471,
                    model=props.get("model", "unknown"),
                    device=props.get("device", "unknown"),
                    memory_gb=float(props.get("memory_gb", 0)),
                    status=props.get("status", "unknown"),
                    discovered_at=time.time(),
                )
                self.discovered[name] = peer
                logger.info("Discovered coordinator: %s at %s:%d", peer.model, peer.host, peer.port)
                if self.on_found:
                    self.on_found(peer)

        elif state_change == ServiceStateChange.Removed:
            self.discovered.pop(name, None)
            logger.info("Coordinator removed: %s", name)
            if self.on_removed:
                self.on_removed(name)
