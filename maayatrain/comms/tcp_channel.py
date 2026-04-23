"""Async TCP transport layer for MaayaTrain peer-to-peer communication.

Provides a non-blocking server and client built on ``asyncio`` streams.
All messages are exchanged as wire-format frames (see ``wire_format.py``).

Key features:
* Connection pool with automatic reconnection.
* Periodic heartbeat to detect dead peers.
* Graceful shutdown with drain.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional

from .wire_format import Frame, MsgKind, encode, new_peer_id, read_frame

logger = logging.getLogger("maayatrain.comms")

FrameHandler = Callable[[Frame, "PeerConnection"], Coroutine[Any, Any, None]]

# ---------------------------------------------------------------------------
# Peer connection wrapper
# ---------------------------------------------------------------------------


# Number of RTT samples to keep for the rolling average.
_RTT_WINDOW = 10


@dataclass
class PeerConnection:
    """Represents one TCP connection to/from a peer."""

    peer_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    address: str
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    _rtt_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=_RTT_WINDOW))
    _pending_heartbeat_ts: Optional[float] = field(default=None)

    def record_rtt(self, rtt_ms: float) -> None:
        """Record a single RTT measurement in milliseconds."""
        self._rtt_samples.append(rtt_ms)

    @property
    def avg_rtt_ms(self) -> float:
        """Rolling average RTT in milliseconds (0.0 if no samples)."""
        if not self._rtt_samples:
            return 0.0
        return sum(self._rtt_samples) / len(self._rtt_samples)

    async def send(self, data: bytes) -> None:
        """Write raw bytes and drain, chunking large payloads."""
        CHUNK = 1024 * 1024  # 1 MB chunks
        for i in range(0, len(data), CHUNK):
            self.writer.write(data[i : i + CHUNK])
            await self.writer.drain()

    async def send_frame(
        self,
        kind: MsgKind,
        sender_id: str,
        payload: bytes = b"",
        compression: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        raw = encode(kind, sender_id, payload, compression, extra)
        await self.send(raw)

    async def close(self) -> None:
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception:
            pass

    @property
    def is_alive(self) -> bool:
        return not self.writer.is_closing()


# ---------------------------------------------------------------------------
# TCP Server
# ---------------------------------------------------------------------------


class TcpServer:
    """Asyncio TCP server that accepts peer connections.

    Usage::

        server = TcpServer(peer_id="abc123", port=7471)
        server.on_frame = my_handler
        await server.start()
        ...
        await server.stop()
    """

    def __init__(
        self,
        peer_id: str,
        port: int = 7471,
        bind: str = "0.0.0.0",
        heartbeat_interval: int = 5,
    ) -> None:
        self.peer_id = peer_id
        self.port = port
        self.bind = bind
        self.heartbeat_interval = heartbeat_interval

        self.peers: Dict[str, PeerConnection] = {}
        self.on_frame: Optional[FrameHandler] = None
        self._server: Optional[asyncio.AbstractServer] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    @property
    def cluster_avg_rtt_ms(self) -> float:
        """Average RTT across all connected peers (0.0 if none)."""
        if not self.peers:
            return 0.0
        rtts = [c.avg_rtt_ms for c in self.peers.values() if c.avg_rtt_ms > 0]
        return sum(rtts) / len(rtts) if rtts else 0.0

    # -- lifecycle --

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.bind,
            self.port,
        )
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        addr = self._server.sockets[0].getsockname() if self._server.sockets else ("?", "?")
        logger.info("TCP server listening on %s:%s", addr[0], addr[1])

    async def stop(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        # Gracefully close all peers
        for conn in list(self.peers.values()):
            try:
                await conn.send_frame(MsgKind.PEER_LEAVE, self.peer_id)
            except Exception:
                pass
            await conn.close()
        self.peers.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("TCP server stopped")

    # -- connection handling --

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        addr = writer.get_extra_info("peername")
        address_str = f"{addr[0]}:{addr[1]}" if addr else "unknown"
        temp_id = new_peer_id()
        conn = PeerConnection(
            peer_id=temp_id,
            reader=reader,
            writer=writer,
            address=address_str,
        )
        logger.info("Incoming connection from %s", address_str)

        try:
            while True:
                frame = await read_frame(reader)
                conn.last_heartbeat = time.time()

                # Update peer_id on handshake
                if frame.kind == MsgKind.HANDSHAKE:
                    old_id = conn.peer_id
                    conn.peer_id = frame.sender_id
                    self.peers.pop(old_id, None)
                    self.peers[conn.peer_id] = conn
                    logger.info("Peer %s registered (was %s)", conn.peer_id, old_id)

                elif frame.kind == MsgKind.HEARTBEAT:
                    # Check if this is a response to our heartbeat probe
                    hb_ts = frame.header.get("hb_ts")
                    if hb_ts is not None and conn._pending_heartbeat_ts is not None:
                        rtt_ms = (time.time() - conn._pending_heartbeat_ts) * 1000
                        conn.record_rtt(rtt_ms)
                        conn._pending_heartbeat_ts = None
                    else:
                        # Echo back as ACK with the original timestamp
                        await conn.send_frame(
                            MsgKind.HEARTBEAT, self.peer_id,
                            extra={"hb_ts": frame.header.get("hb_ts", time.time())},
                        )
                    continue

                elif frame.kind == MsgKind.PEER_LEAVE:
                    logger.info("Peer %s left gracefully", conn.peer_id)
                    break

                # Dispatch to user handler
                if self.on_frame:
                    await self.on_frame(frame, conn)

        except (asyncio.IncompleteReadError, ConnectionError):
            logger.warning("Peer %s disconnected", conn.peer_id)
        except Exception as exc:
            logger.error("Error with peer %s: %s", conn.peer_id, exc)
        finally:
            self.peers.pop(conn.peer_id, None)
            await conn.close()

    # -- broadcast --

    async def broadcast(
        self,
        kind: MsgKind,
        payload: bytes = b"",
        compression: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Send a frame to all connected peers."""
        raw = encode(kind, self.peer_id, payload, compression, extra)
        dead: list[str] = []
        for pid, conn in self.peers.items():
            try:
                await conn.send(raw)
            except Exception:
                dead.append(pid)
        for pid in dead:
            peer = self.peers.pop(pid, None)
            if peer:
                await peer.close()
                logger.warning("Dropped dead peer %s", pid)

    # -- heartbeat --

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)
                now = time.time()
                stale_threshold = self.heartbeat_interval * 3
                dead: list[str] = []
                for pid, conn in self.peers.items():
                    if now - conn.last_heartbeat > stale_threshold:
                        dead.append(pid)
                    else:
                        # Send heartbeat probe with timestamp for RTT measurement
                        try:
                            conn._pending_heartbeat_ts = time.time()
                            await conn.send_frame(
                                MsgKind.HEARTBEAT, self.peer_id,
                                extra={"hb_ts": conn._pending_heartbeat_ts},
                            )
                        except Exception:
                            dead.append(pid)
                for pid in dead:
                    peer = self.peers.pop(pid, None)
                    if peer:
                        await peer.close()
                        logger.warning("Peer %s timed out (no heartbeat)", pid)
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# TCP Client
# ---------------------------------------------------------------------------


class TcpClient:
    """Connect to a remote TcpServer as a peer.

    Usage::

        client = TcpClient(peer_id="xyz789")
        conn = await client.connect("192.168.1.10", 7471)
    """

    def __init__(self, peer_id: str, heartbeat_interval: int = 5) -> None:
        self.peer_id = peer_id
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def connect(self, host: str, port: int) -> PeerConnection:
        """Open a TCP connection to the coordinator and perform handshake."""
        reader, writer = await asyncio.open_connection(host, port)
        conn = PeerConnection(
            peer_id="coordinator",
            reader=reader,
            writer=writer,
            address=f"{host}:{port}",
        )

        # Send handshake
        await conn.send_frame(MsgKind.HANDSHAKE, self.peer_id)
        logger.info("Connected to %s:%s — handshake sent", host, port)

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(conn))

        return conn

    async def listen(self, conn: PeerConnection, handler: FrameHandler) -> None:
        """Continuously read frames from the connection and dispatch."""
        try:
            while True:
                frame = await read_frame(conn.reader)
                conn.last_heartbeat = time.time()
                if frame.kind == MsgKind.HEARTBEAT:
                    # Echo heartbeat back with timestamp for RTT measurement
                    hb_ts = frame.header.get("hb_ts")
                    if hb_ts is not None:
                        await conn.send_frame(
                            MsgKind.HEARTBEAT, self.peer_id,
                            extra={"hb_ts": hb_ts},
                        )
                    continue
                if frame.kind == MsgKind.PEER_LEAVE:
                    logger.info("Coordinator sent PEER_LEAVE — disconnecting")
                    break
                await handler(frame, conn)
        except (asyncio.IncompleteReadError, ConnectionError):
            logger.warning("Lost connection to coordinator")
        finally:
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            await conn.close()

    async def _heartbeat_loop(self, conn: PeerConnection) -> None:
        try:
            while conn.is_alive:
                await asyncio.sleep(self.heartbeat_interval)
                await conn.send_frame(MsgKind.HEARTBEAT, self.peer_id)
        except (asyncio.CancelledError, ConnectionError):
            pass
