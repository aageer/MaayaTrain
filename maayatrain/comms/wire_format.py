"""Binary wire protocol for MaayaTrain peer-to-peer messages.

Frame layout (on the wire)::

    ┌──────────────┬──────────────┬─────────────────┐
    │ header_len   │ JSON header  │ binary payload   │
    │ (4 bytes BE) │ (variable)   │ (variable, opt.) │
    └──────────────┴──────────────┴─────────────────┘

* **header_len** — ``uint32`` big-endian, size of the JSON header in bytes.
* **JSON header** — UTF-8 encoded JSON with metadata (msg_type, sender_id, …).
* **binary payload** — optional raw bytes (e.g. compressed tensors).

This module provides ``encode`` / ``decode`` plus the ``MsgKind`` enum.
"""

from __future__ import annotations

import json
import struct
import time
import uuid
from enum import Enum
from typing import Any, Optional

# 4-byte big-endian unsigned int for the header length prefix
_HEADER_LEN_FMT = "!I"
_HEADER_LEN_SIZE = struct.calcsize(_HEADER_LEN_FMT)

# Maximum header size: 64 KiB (sanity guard against corrupt streams)
_MAX_HEADER_BYTES = 65_536


class MsgKind(str, Enum):
    """All message types exchanged between MaayaTrain peers."""

    HANDSHAKE = "handshake"
    SYNC_REQUEST = "sync_request"
    SYNC_GRADIENTS = "sync_gradients"
    MODEL_WEIGHTS = "model_weights"
    HEARTBEAT = "heartbeat"
    PEER_JOIN = "peer_join"
    PEER_LEAVE = "peer_leave"
    STATUS_QUERY = "status_query"
    STATUS_RESPONSE = "status_response"
    ERROR = "error"


def _make_header(
    kind: MsgKind,
    sender_id: str,
    payload_size: int = 0,
    compression: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    hdr: dict[str, Any] = {
        "msg_type": kind.value,
        "sender_id": sender_id,
        "timestamp": time.time(),
        "payload_size": payload_size,
    }
    if compression:
        hdr["compression"] = compression
    if extra:
        hdr.update(extra)
    return hdr


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------


def encode(
    kind: MsgKind,
    sender_id: str,
    payload: bytes = b"",
    compression: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> bytes:
    """Serialise a message into the binary wire format.

    Parameters
    ----------
    kind : MsgKind
        The message type.
    sender_id : str
        Unique identifier of the sending peer.
    payload : bytes
        Optional binary payload (e.g. compressed tensors).
    compression : str | None
        Compression method used on the payload (``"fp16_gzip"`` etc.).
    extra : dict | None
        Extra key-value pairs merged into the JSON header.

    Returns
    -------
    bytes
        The complete frame ready to send over TCP.
    """
    header = _make_header(kind, sender_id, len(payload), compression, extra)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack(_HEADER_LEN_FMT, len(header_bytes)) + header_bytes + payload


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


class Frame:
    """A decoded wire-protocol frame."""

    __slots__ = ("header", "payload")

    def __init__(self, header: dict[str, Any], payload: bytes) -> None:
        self.header = header
        self.payload = payload

    @property
    def kind(self) -> MsgKind:
        return MsgKind(self.header["msg_type"])

    @property
    def sender_id(self) -> str:
        return self.header["sender_id"]

    @property
    def timestamp(self) -> float:
        return self.header.get("timestamp", 0.0)

    @property
    def compression(self) -> Optional[str]:
        return self.header.get("compression")

    def __repr__(self) -> str:
        return (
            f"Frame(kind={self.kind.value}, sender={self.sender_id!r}, "
            f"payload={len(self.payload)} bytes)"
        )


async def read_frame(reader: "asyncio.StreamReader") -> Frame:  # noqa: F821
    """Read exactly one frame from an asyncio StreamReader.

    Raises
    ------
    ConnectionError
        If the stream is closed before a full frame is received.
    ValueError
        If the header is too large or malformed.
    """
    import asyncio  # local to avoid import cost when not used async

    # 1. Read header length
    raw_len = await reader.readexactly(_HEADER_LEN_SIZE)
    (header_len,) = struct.unpack(_HEADER_LEN_FMT, raw_len)
    if header_len > _MAX_HEADER_BYTES:
        raise ValueError(f"Header too large: {header_len} bytes (max {_MAX_HEADER_BYTES})")

    # 2. Read JSON header
    header_bytes = await reader.readexactly(header_len)
    header: dict[str, Any] = json.loads(header_bytes)

    # 3. Read binary payload (if any)
    payload_size = header.get("payload_size", 0)
    payload = await reader.readexactly(payload_size) if payload_size > 0 else b""

    return Frame(header, payload)


def decode_bytes(data: bytes) -> Frame:
    """Decode a frame from a contiguous byte buffer (non-async, for testing)."""
    if len(data) < _HEADER_LEN_SIZE:
        raise ValueError("Buffer too short for header length prefix")

    (header_len,) = struct.unpack(_HEADER_LEN_FMT, data[:_HEADER_LEN_SIZE])
    if header_len > _MAX_HEADER_BYTES:
        raise ValueError(f"Header too large: {header_len}")

    offset = _HEADER_LEN_SIZE
    header_bytes = data[offset : offset + header_len]
    header: dict[str, Any] = json.loads(header_bytes)

    offset += header_len
    payload_size = header.get("payload_size", 0)
    payload = data[offset : offset + payload_size]

    return Frame(header, payload)


def new_peer_id() -> str:
    """Generate a short, unique peer identifier."""
    return uuid.uuid4().hex[:12]
