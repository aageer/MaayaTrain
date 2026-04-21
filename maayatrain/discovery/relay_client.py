"""HTTP relay client for internet-based peer discovery.

When mDNS won't work (different networks, enterprise Wi-Fi that blocks
multicast), peers can use an HTTP signaling server to find each other.

The relay only handles *discovery* metadata — all training data flows
directly peer-to-peer via TCP.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("maayatrain.discovery.relay")


class RelayClient:
    """Client for the MaayaTrain HTTP discovery relay.

    Usage::

        relay = RelayClient("https://maayatrain.dev/api/relay")
        await relay.register_session(model="gpt2-small", port=7471, ...)
        sessions = await relay.list_sessions()
    """

    def __init__(self, relay_url: str, timeout: float = 10.0) -> None:
        self.relay_url = relay_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def register_session(
        self,
        *,
        model: str,
        connect_address: str,
        device_name: str = "unknown",
        memory_gb: float = 0.0,
        description: str = "",
    ) -> Dict[str, Any]:
        """Register a new training session with the relay.

        Returns the session record (including assigned ID).
        """
        payload = {
            "model_name": model,
            "connect_address": connect_address,
            "device_name": device_name,
            "memory_gb": memory_gb,
            "description": description,
            "status": "active",
        }
        resp = await self._client.post(f"{self.relay_url}/sessions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        logger.info("Session registered with relay: %s", data.get("id"))
        return data

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """Fetch all active training sessions from the relay."""
        resp = await self._client.get(f"{self.relay_url}/sessions")
        resp.raise_for_status()
        return resp.json()

    async def heartbeat(self, session_id: str) -> None:
        """Send a heartbeat to keep the session alive."""
        resp = await self._client.post(f"{self.relay_url}/sessions/{session_id}/heartbeat")
        resp.raise_for_status()

    async def unregister_session(self, session_id: str) -> None:
        """Remove a session from the relay."""
        resp = await self._client.delete(f"{self.relay_url}/sessions/{session_id}")
        resp.raise_for_status()
        logger.info("Session %s unregistered from relay", session_id)

    async def close(self) -> None:
        await self._client.aclose()
