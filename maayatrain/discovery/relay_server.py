"""Lightweight HTTP relay server for internet-based peer discovery.

When mDNS won't work (different networks, enterprise Wi-Fi that blocks
multicast), peers can use this HTTP signaling server to find each other.

The relay only handles *discovery* metadata — all training data flows
directly peer-to-peer via TCP.

Usage::

    # Start the relay server (default port 7480)
    python -m maayatrain.discovery.relay_server

    # Or with uvicorn directly
    uvicorn maayatrain.discovery.relay_server:app --host 0.0.0.0 --port 7480
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class SessionCreate(BaseModel):
    """Request body for creating a new training session."""

    model_name: str = "gpt2-small"
    connect_address: str  # e.g. "192.168.1.10:7471"
    device_name: str = "unknown"
    memory_gb: float = 0.0
    description: str = ""


class SessionRecord(BaseModel):
    """A registered training session."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_name: str = "gpt2-small"
    connect_address: str = ""
    device_name: str = "unknown"
    memory_gb: float = 0.0
    description: str = ""
    status: str = "active"
    created_at: float = Field(default_factory=time.time)
    last_heartbeat: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MaayaTrain Relay Server",
    description=(
        "Lightweight signaling server for MaayaTrain peer discovery. "
        "Handles discovery metadata only — training data flows peer-to-peer."
    ),
    version="0.1.0",
)

# In-memory session store (sufficient for small-scale relay)
_sessions: Dict[str, SessionRecord] = {}

# Sessions expire after 5 minutes without a heartbeat
_HEARTBEAT_TIMEOUT = 300.0


def _prune_stale() -> None:
    """Remove sessions that haven't sent a heartbeat recently."""
    now = time.time()
    stale = [
        sid
        for sid, s in _sessions.items()
        if now - s.last_heartbeat > _HEARTBEAT_TIMEOUT
    ]
    for sid in stale:
        del _sessions[sid]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "sessions": len(_sessions)}


@app.post("/sessions", response_model=SessionRecord, status_code=201)
async def create_session(body: SessionCreate) -> SessionRecord:
    """Register a new training session."""
    _prune_stale()
    record = SessionRecord(
        model_name=body.model_name,
        connect_address=body.connect_address,
        device_name=body.device_name,
        memory_gb=body.memory_gb,
        description=body.description,
    )
    _sessions[record.id] = record
    return record


@app.get("/sessions", response_model=list[SessionRecord])
async def list_sessions(model: Optional[str] = None) -> list[SessionRecord]:
    """List all active training sessions, optionally filtered by model."""
    _prune_stale()
    sessions = list(_sessions.values())
    if model:
        sessions = [s for s in sessions if s.model_name == model]
    return sessions


@app.get("/sessions/{session_id}", response_model=SessionRecord)
async def get_session(session_id: str) -> SessionRecord:
    """Get a specific session by ID."""
    _prune_stale()
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return _sessions[session_id]


@app.post("/sessions/{session_id}/heartbeat")
async def heartbeat(session_id: str) -> dict:
    """Send a heartbeat to keep a session alive."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    _sessions[session_id].last_heartbeat = time.time()
    return {"status": "ok"}


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    """Unregister a training session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("⚡ MaayaTrain Relay Server")
    print("  Endpoints:")
    print("    POST   /sessions              — Register a session")
    print("    GET    /sessions              — List active sessions")
    print("    POST   /sessions/{id}/heartbeat — Keep session alive")
    print("    DELETE /sessions/{id}         — Unregister session")
    print()
    uvicorn.run(app, host="0.0.0.0", port=7480, log_level="info")
