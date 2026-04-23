"""Local monitoring dashboard — FastAPI + WebSocket.

Serves a real-time web UI at ``http://localhost:8471`` showing:
- Training loss curve (Chart.js)
- Connected peers table
- Throughput metrics
- Checkpoint timeline
- **Training controls** (start/stop/config from browser)
- **Worker join portal** (one-click command for LAN peers)

Data is pushed via WebSocket for immediate updates.
API endpoints let the browser act as a full control panel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("maayatrain.monitor")

STATIC_DIR = Path(__file__).parent / "static"


def _get_local_ip() -> str:
    """Get the LAN IP so workers know where to connect."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def create_dashboard_app() -> FastAPI:
    """Create the FastAPI dashboard application."""
    app = FastAPI(title="MaayaTrain Dashboard", version="0.3.0")

    # Shared state
    app.state.metrics_buffer: List[Dict[str, Any]] = []
    app.state.cluster_info: Dict[str, Any] = {}
    app.state.checkpoints: List[Dict[str, Any]] = []
    app.state.ws_clients: Set[WebSocket] = set()
    # Training control references (set by orchestrator on startup)
    app.state.orchestrator = None
    app.state.training_config: Dict[str, Any] = {}
    app.state.training_active: bool = False
    app.state.start_time: float = time.time()

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = STATIC_DIR / "index.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>MaayaTrain Dashboard</h1><p>Static files not found.</p>")

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        app.state.ws_clients.add(ws)
        logger.info("Dashboard client connected (%d total)", len(app.state.ws_clients))

        try:
            # Send current state on connect
            await ws.send_json({
                "type": "init",
                "metrics": app.state.metrics_buffer[-100:],
                "cluster": app.state.cluster_info,
                "checkpoints": app.state.checkpoints,
                "config": app.state.training_config,
                "training_active": app.state.training_active,
            })

            # Keep alive — read messages (e.g. commands from dashboard)
            while True:
                data = await ws.receive_text()
                logger.debug("Received from dashboard: %s", data)

        except WebSocketDisconnect:
            pass
        finally:
            app.state.ws_clients.discard(ws)
            logger.info("Dashboard client disconnected (%d remaining)", len(app.state.ws_clients))

    # ----------------------------------------------------------------
    # REST API — Training Control (Unsloth Studio pattern)
    # ----------------------------------------------------------------

    @app.get("/api/status")
    async def status():
        """Full cluster status for the dashboard."""
        uptime = time.time() - app.state.start_time
        return {
            "status": "training" if app.state.training_active else "idle",
            "uptime_seconds": round(uptime),
            "cluster": app.state.cluster_info,
            "latest_metrics": app.state.metrics_buffer[-1] if app.state.metrics_buffer else None,
            "total_steps": app.state.metrics_buffer[-1].get("step", 0) if app.state.metrics_buffer else 0,
        }

    @app.get("/api/config")
    async def get_config():
        """Return current training configuration for UI display."""
        return app.state.training_config

    @app.get("/api/join")
    async def join_info():
        """Return everything a worker needs to join this training session."""
        local_ip = _get_local_ip()
        port = app.state.training_config.get("port", 7471)
        model = app.state.training_config.get("model", "gpt2-tiny")
        repo = "https://github.com/aageer/MaayaTrain.git"
        return {
            "coordinator_ip": local_ip,
            "coordinator_port": port,
            "model": model,
            "dashboard_url": f"http://{local_ip}:8471",
            "install_command": f"pip install git+{repo}",
            "join_command": "maayatrain quickstart join",
            "one_liner": f"pip install git+{repo} && maayatrain quickstart join",
            "curl_join": f"curl -s http://{local_ip}:8471/join.sh | bash",
        }

    @app.get("/join.sh")
    async def join_script():
        """Serve a shell script that auto-installs and joins training."""
        local_ip = _get_local_ip()
        repo = "https://github.com/aageer/MaayaTrain.git"
        script = f"""#!/bin/bash
# MaayaTrain Auto-Join Script
# Run: curl -s http://{local_ip}:8471/join.sh | bash
set -e
echo "⚡ Installing MaayaTrain..."
pip install -q git+{repo}
echo "🔗 Joining coordinator at {local_ip}..."
maayatrain quickstart join
"""
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(script, media_type="text/x-shellscript")

    @app.post("/api/training/stop")
    async def stop_training():
        """Signal the training loop to stop gracefully."""
        if app.state.orchestrator and hasattr(app.state.orchestrator, '_stop_requested'):
            app.state.orchestrator._stop_requested = True
            app.state.training_active = False
            return {"status": "stopping", "message": "Training will stop after current round"}
        return {"status": "error", "message": "No active training to stop"}

    return app


async def broadcast_to_dashboard(app: FastAPI, message: Dict[str, Any]) -> None:
    """Push a message to all connected WebSocket clients."""
    dead: List[WebSocket] = []
    for ws in app.state.ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        app.state.ws_clients.discard(ws)


def push_metrics(
    app: FastAPI,
    step: int,
    loss: float,
    tokens_per_sec: float,
    lr: float,
    peers: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a training step metric and schedule broadcast."""
    entry: Dict[str, Any] = {
        "type": "metric",
        "step": step,
        "loss": loss,
        "tokens_per_sec": tokens_per_sec,
        "lr": lr,
        "timestamp": time.time(),
    }

    # Include peer info if available
    if peers is not None:
        entry["peers"] = peers
        entry["peer_count"] = len(peers)
    else:
        entry["peer_count"] = 0
        entry["peers"] = {}

    app.state.metrics_buffer.append(entry)

    # Keep buffer bounded
    if len(app.state.metrics_buffer) > 10_000:
        app.state.metrics_buffer = app.state.metrics_buffer[-5_000:]

    # Schedule async broadcast
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(broadcast_to_dashboard(app, entry))
    except RuntimeError:
        pass  # No event loop running (e.g. during tests)


def push_cluster_update(app: FastAPI, cluster_info: Dict[str, Any]) -> None:
    """Update cluster state and broadcast."""
    app.state.cluster_info = cluster_info
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(broadcast_to_dashboard(app, {"type": "cluster", **cluster_info}))
    except RuntimeError:
        pass


def push_checkpoint(app: FastAPI, step: int, loss: float, path: str) -> None:
    """Record a checkpoint save event."""
    entry = {"step": step, "loss": loss, "path": path, "timestamp": time.time()}
    app.state.checkpoints.append(entry)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(broadcast_to_dashboard(app, {"type": "checkpoint", **entry}))
    except RuntimeError:
        pass

