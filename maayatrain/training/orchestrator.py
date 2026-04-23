"""Orchestrator — coordinator node logic for MaayaTrain.

The orchestrator is the "primary" node that:
1. Accepts worker connections.
2. Distributes initial model weights.
3. Collects pseudo-gradients after each inner loop.
4. Runs the DiLoCo outer step (averaging + Nesterov SGD).
5. Broadcasts updated global weights.

Handles workers joining and leaving mid-training gracefully.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor, nn

from ..comms.tcp_channel import PeerConnection, TcpServer
from ..comms.tensor_codec import compress, compression_tag, decompress
from ..comms.wire_format import Frame, MsgKind
from ..hardware import DeviceProfile
from ..settings import MaayaTrainSettings
from .cluster_info import ClusterState
from .diloco import DiLoCoEngine
from .loop import SimpleTextDataset, StepMetrics, train_steps, train_steps_timed
from .snapshots import SnapshotMeta, save_snapshot, step_directory

logger = logging.getLogger("maayatrain.orchestrator")


class Orchestrator:
    """Coordinator node that manages distributed DiLoCo training.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    settings : MaayaTrainSettings
        Configuration.
    device_profile : DeviceProfile
        Local hardware info.
    """

    def __init__(
        self,
        model: nn.Module,
        settings: MaayaTrainSettings,
        device_profile: DeviceProfile,
        dataset: SimpleTextDataset,
    ) -> None:
        self.model = model
        self.settings = settings
        self.device_profile = device_profile
        self.dataset = dataset

        self.device = device_profile.device
        self.model.to(self.device)

        # DiLoCo engine
        self.diloco = DiLoCoEngine(model, settings.diloco, self.device)

        # Training state
        self.global_step = 0
        self.training_start_time = time.time()
        self._received_gradients: Dict[str, Dict[str, Tensor]] = {}
        self._received_local_steps: Dict[str, int] = {}
        self._metrics_history: List[StepMetrics] = []

        # TCP server
        from ..comms.wire_format import new_peer_id

        self.peer_id = new_peer_id()
        self.server = TcpServer(
            peer_id=self.peer_id,
            port=settings.network.port,
            heartbeat_interval=settings.network.heartbeat_interval,
        )
        self.server.on_frame = self._on_frame

        # Status tracking
        self.cluster = ClusterState(coordinator_id=self.peer_id)

        # Callbacks
        self.on_metrics: Optional[Callable[[StepMetrics], None]] = None
        self.on_outer_step: Optional[Callable[[int], None]] = None
        self._stop_event = asyncio.Event()

        # Dynamic streaming shards (starts from config, adjusted by RTT)
        self._current_streaming_shards: int = settings.diloco.streaming_shards

    async def run(self) -> None:
        """Start the training loop (blocks until max_steps or interruption)."""
        await self.server.start()
        logger.info(
            "Orchestrator %s started | model=%s | device=%s | port=%d",
            self.peer_id,
            self.settings.model.name,
            self.device_profile.summary(),
            self.settings.network.port,
        )

        # Register interrupt handler
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: self._stop_event.set())
            except NotImplementedError:
                pass  # Windows doesn't support add_signal_handler in asyncio

        try:
            while self.global_step < self.settings.training.max_steps:
                if self._stop_event.is_set():
                    break
                await self._run_one_round()
        except asyncio.CancelledError:
            pass
        finally:
            # Save final checkpoint
            self._save_checkpoint()
            await self.server.stop()
            logger.info("Orchestrator stopped at step %d", self.global_step)

    async def _run_one_round(self) -> None:
        """Execute one DiLoCo outer round: inner loop + sync + outer step."""
        H = self.settings.diloco.inner_steps
        sync_mode = self.settings.diloco.sync_mode
        aggregation = self.settings.diloco.aggregation

        # 0. Dynamic re-sharding based on cluster RTT
        self._adapt_streaming_shards()

        # 1. Snapshot global params
        self.diloco.snapshot_global()

        # 2. Broadcast current weights to all workers
        await self._broadcast_weights()

        # 3. Run local inner loop (coordinator also trains)
        self.diloco.reset_inner_optimizer()

        if sync_mode == "time":
            window = self.settings.diloco.sync_window_seconds
            metrics, own_local_steps = train_steps_timed(
                self.model,
                self.diloco.inner_optimizer,
                self.dataset,
                window_seconds=window,
                batch_size=self.settings.training.batch_size,
                device=self.device,
                start_step=self.global_step,
                log_every=self.settings.training.log_every,
                on_step=self.on_metrics,
                estimated_steps=H,
            )
        else:
            metrics = train_steps(
                self.model,
                self.diloco.inner_optimizer,
                self.dataset,
                num_steps=H,
                batch_size=self.settings.training.batch_size,
                device=self.device,
                start_step=self.global_step,
                log_every=self.settings.training.log_every,
                on_step=self.on_metrics,
            )
            own_local_steps = H

        self._metrics_history.extend(metrics)

        # 4. Compute our own pseudo-gradient
        own_pg = self.diloco.compute_pseudo_gradient()

        # 5. Request sync from workers — include current_streaming_shards
        self._received_gradients.clear()
        self._received_local_steps.clear()
        if self.server.peers:
            await self.server.broadcast(
                MsgKind.SYNC_REQUEST,
                extra={
                    "step": self.global_step,
                    "current_streaming_shards": self._current_streaming_shards,
                },
            )

            # Wait for all workers to send pseudo-gradients (with timeout)
            deadline = time.time() + 60  # 60s timeout
            while len(self._received_gradients) < len(self.server.peers):
                if time.time() > deadline:
                    logger.warning(
                        "Sync timeout — got %d/%d gradients, proceeding",
                        len(self._received_gradients),
                        len(self.server.peers),
                    )
                    break
                await asyncio.sleep(0.1)

        # 6. Gather all pseudo-gradients (ours + workers')
        all_pgs = [own_pg] + list(self._received_gradients.values())
        all_local_steps = [own_local_steps] + [
            self._received_local_steps.get(pid, H)
            for pid in self._received_gradients
        ]

        # 7. Apply outer step (streaming shards or single-shot)
        K = self._current_streaming_shards
        if K > 1:
            shards = self.diloco.compute_streaming_shards(num_shards=K)
            for shard in shards:
                if sync_mode == "time":
                    # For streaming + time mode: use standard shard step
                    # (weighted aggregation is not implemented per-shard)
                    self.diloco.apply_outer_step_shard(
                        all_pgs, shard_names=shard, aggregation=aggregation
                    )
                else:
                    self.diloco.apply_outer_step_shard(
                        all_pgs, shard_names=shard, aggregation=aggregation
                    )
        else:
            if sync_mode == "time":
                self.diloco.apply_outer_step_weighted(
                    all_pgs, all_local_steps, aggregation=aggregation
                )
            else:
                self.diloco.apply_outer_step(all_pgs, aggregation=aggregation)

        # 8. Update global step
        self.global_step += own_local_steps

        # 9. Checkpoint
        if self.global_step % self.settings.training.checkpoint_every < max(H, own_local_steps):
            self._save_checkpoint()

        if self.on_outer_step:
            self.on_outer_step(self.global_step)

        last_loss = metrics[-1].loss if metrics else float("inf")
        logger.info(
            "Outer round complete | step=%d | loss=%.4f | workers=%d | mode=%s | shards=%d",
            self.global_step,
            last_loss,
            len(self.server.peers),
            sync_mode,
            K,
        )

    # -- Dynamic shard adaptation --

    # RTT thresholds in milliseconds
    _RTT_HIGH = 150.0   # Above this → increase shards (reduce payload size)
    _RTT_LOW = 30.0     # Below this → decrease shards (reduce loop overhead)
    _SHARDS_MIN = 1
    _SHARDS_MAX = 16

    def _adapt_streaming_shards(self) -> None:
        """Adjust streaming_shards based on cluster-wide average RTT.

        Called at the start of each outer round. If the cluster latency is
        high (congested Wi-Fi), we increase shards to reduce the per-message
        payload. If latency is excellent (ethernet/localhost), we decrease
        shards to reduce Python loop overhead.
        """
        if not self.server.peers:
            return  # No peers → nothing to adapt

        avg_rtt = self.server.cluster_avg_rtt_ms
        if avg_rtt <= 0:
            return  # No RTT data yet

        old_k = self._current_streaming_shards

        if avg_rtt > self._RTT_HIGH:
            # High latency → more shards (smaller payloads)
            new_k = min(old_k * 2, self._SHARDS_MAX)
        elif avg_rtt < self._RTT_LOW:
            # Low latency → fewer shards (less overhead)
            new_k = max(old_k // 2, self._SHARDS_MIN)
        else:
            return  # In the acceptable range — no change

        if new_k != old_k:
            logger.info(
                "Dynamic re-sharding: %d → %d (avg_rtt=%.1fms)",
                old_k, new_k, avg_rtt,
            )
            self._current_streaming_shards = new_k

    async def _on_frame(self, frame: Frame, conn: PeerConnection) -> None:
        """Handle incoming messages from workers."""
        if frame.kind == MsgKind.SYNC_GRADIENTS:
            # Decompress pseudo-gradient
            pg = decompress(frame.payload, restore_fp32=True)
            self._received_gradients[frame.sender_id] = pg
            # Track local_steps reported by this worker
            local_steps = frame.header.get("local_steps", self.settings.diloco.inner_steps)
            self._received_local_steps[frame.sender_id] = local_steps
            logger.debug(
                "Received pseudo-gradient from %s (local_steps=%d)",
                frame.sender_id,
                local_steps,
            )

        elif frame.kind == MsgKind.HANDSHAKE:
            # Send current weights to newly joined worker
            logger.info("Worker %s joined — sending model weights", frame.sender_id)
            self.cluster.add_peer(frame.sender_id, frame.header)
            await self._send_weights_to(conn)

        elif frame.kind == MsgKind.STATUS_QUERY:
            # Respond with cluster status
            status = self.cluster.to_dict(self.global_step, self._latest_loss())
            import json

            payload = json.dumps(status).encode()
            await conn.send_frame(MsgKind.STATUS_RESPONSE, self.peer_id, payload)

    async def _broadcast_weights(self) -> None:
        """Compress and send current model weights to all peers."""
        if not self.server.peers:
            return
        state = self.diloco.get_global_weights()
        use_fp16 = self.settings.diloco.compress_fp16
        payload = compress(state, use_fp16=use_fp16)
        tag = compression_tag(use_fp16)
        await self.server.broadcast(MsgKind.MODEL_WEIGHTS, payload, compression=tag)
        logger.debug("Broadcasted weights (%d bytes) to %d peers", len(payload), len(self.server.peers))

    async def _send_weights_to(self, conn: PeerConnection) -> None:
        """Send model weights to a single peer."""
        state = self.diloco.get_global_weights()
        use_fp16 = self.settings.diloco.compress_fp16
        payload = compress(state, use_fp16=use_fp16)
        tag = compression_tag(use_fp16)
        await conn.send_frame(MsgKind.MODEL_WEIGHTS, self.peer_id, payload, compression=tag)

    def _save_checkpoint(self) -> None:
        """Save a training checkpoint."""
        ckpt_dir = step_directory(self.settings.training.checkpoint_dir, self.global_step)
        meta = SnapshotMeta(
            model_name=self.settings.model.name,
            global_step=self.global_step,
            loss=self._latest_loss(),
            total_compute_hours=self._compute_hours(),
            contributors=[self.device_profile.hostname]
            + [p for p in self.server.peers.keys()],
        )
        save_snapshot(
            ckpt_dir,
            self.model,
            self.diloco.inner_optimizer,
            meta,
            momentum_buffer=self.diloco._momentum_buffer,
        )

        # --- SOTA TELEMETRY EXPORT ---
        # Auto-log cluster metrics for paper benchmarks and debugging.
        # Produces a CSV that generate_benchmarks.py can ingest directly.
        import csv

        telemetry_file = Path(self.settings.training.checkpoint_dir) / "telemetry.csv"
        file_exists = telemetry_file.exists()
        with open(telemetry_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "global_step", "loss", "compute_hours",
                    "cluster_rtt_ms", "active_peers", "streaming_shards",
                ])
            writer.writerow([
                self.global_step,
                self._latest_loss(),
                self._compute_hours(),
                self.server.cluster_avg_rtt_ms,
                len(self.server.peers),
                self._current_streaming_shards,
            ])

    def _latest_loss(self) -> float:
        return self._metrics_history[-1].loss if self._metrics_history else float("inf")

    def _compute_hours(self) -> float:
        return (time.time() - self.training_start_time) / 3600
