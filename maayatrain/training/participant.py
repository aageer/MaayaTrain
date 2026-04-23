"""Participant — worker node logic for MaayaTrain.

A participant connects to the coordinator, receives model weights,
runs the inner training loop independently, and sends back its
pseudo-gradient for the outer DiLoCo step.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from ..comms.tcp_channel import PeerConnection, TcpClient
from ..comms.tensor_codec import compress, compression_tag, decompress
from ..comms.wire_format import Frame, MsgKind, new_peer_id
from ..hardware import DeviceProfile
from ..settings import MaayaTrainSettings
from .diloco import DiLoCoEngine
from .loop import SimpleTextDataset, StepMetrics, train_steps, train_steps_timed
from .snapshots import SnapshotMeta, save_snapshot, step_directory

logger = logging.getLogger("maayatrain.participant")


class Participant:
    """Worker node that trains locally and syncs with the coordinator.

    Parameters
    ----------
    model : nn.Module
        The model architecture (weights will be overwritten by coordinator).
    settings : MaayaTrainSettings
        Configuration.
    device_profile : DeviceProfile
        Local hardware info.
    dataset : SimpleTextDataset
        Local training data.
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

        # Network
        self.peer_id = new_peer_id()
        self.client = TcpClient(self.peer_id, settings.network.heartbeat_interval)
        self._conn: Optional[PeerConnection] = None

        # State
        self.global_step = 0
        self.training_start_time = time.time()
        self._metrics_history: List[StepMetrics] = []
        self._awaiting_sync = asyncio.Event()
        self._stop = False

    async def connect_and_train(self, host: str, port: int) -> None:
        """Connect to coordinator, then enter the training loop."""
        self._conn = await self.client.connect(host, port)
        logger.info(
            "Participant %s connected to %s:%d | device=%s",
            self.peer_id,
            host,
            port,
            self.device_profile.summary(),
        )

        # Start listening for coordinator messages
        listen_task = asyncio.create_task(
            self.client.listen(self._conn, self._on_frame)
        )

        # Wait for initial weights
        logger.info("Waiting for initial model weights from coordinator...")
        await self._awaiting_sync.wait()
        self._awaiting_sync.clear()

        try:
            while not self._stop:
                # Wait for SYNC_REQUEST (which comes after coordinator has
                # broadcast weights at the start of each round)
                await self._awaiting_sync.wait()
                self._awaiting_sync.clear()

                if self._stop:
                    break

                # Run inner training loop
                await self._run_inner_loop()

        except asyncio.CancelledError:
            pass
        finally:
            # Save local checkpoint before exiting
            self._save_local_checkpoint()
            if self._conn:
                try:
                    await self._conn.send_frame(MsgKind.PEER_LEAVE, self.peer_id)
                except Exception:
                    pass
                await self._conn.close()
            listen_task.cancel()
            logger.info("Participant stopped at step %d", self.global_step)

    async def _run_inner_loop(self) -> None:
        """One round: inner training + send pseudo-gradient."""
        H = self.settings.diloco.inner_steps
        sync_mode = self.settings.diloco.sync_mode

        # 1. Snapshot global params (received from coordinator)
        self.diloco.snapshot_global()

        # 2. Train locally
        self.diloco.reset_inner_optimizer()

        if sync_mode == "time":
            window = self.settings.diloco.sync_window_seconds
            metrics, local_steps_completed = train_steps_timed(
                self.model,
                self.diloco.inner_optimizer,
                self.dataset,
                window_seconds=window,
                batch_size=self.settings.training.batch_size,
                device=self.device,
                start_step=self.global_step,
                log_every=self.settings.training.log_every,
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
            )
            local_steps_completed = H

        self._metrics_history.extend(metrics)

        # 3. Compute pseudo-gradient
        pg = self.diloco.compute_pseudo_gradient()

        # 4. Compress and send to coordinator
        use_fp16 = self.settings.diloco.compress_fp16
        payload = compress(pg, use_fp16=use_fp16)
        tag = compression_tag(use_fp16)

        if self._conn and self._conn.is_alive:
            await self._conn.send_frame(
                MsgKind.SYNC_GRADIENTS,
                self.peer_id,
                payload,
                compression=tag,
                extra={"local_steps": local_steps_completed},
            )
            logger.info(
                "Sent pseudo-gradient (%d bytes, %d local steps) at step %d",
                len(payload),
                local_steps_completed,
                self.global_step,
            )

        self.global_step += local_steps_completed

    async def _on_frame(self, frame: Frame, conn: PeerConnection) -> None:
        """Handle messages from the coordinator."""
        if frame.kind == MsgKind.MODEL_WEIGHTS:
            # Decompress and load weights
            state = decompress(frame.payload, restore_fp32=True, map_location=str(self.device))
            self.diloco.load_global_weights(state)
            logger.debug("Received and loaded model weights from coordinator")
            self._awaiting_sync.set()

        elif frame.kind == MsgKind.SYNC_REQUEST:
            step = frame.header.get("step", 0)
            self.global_step = step
            # Signal the training loop to start/continue
            self._awaiting_sync.set()

        elif frame.kind == MsgKind.PEER_LEAVE:
            logger.info("Coordinator sent PEER_LEAVE — stopping")
            self._stop = True
            self._awaiting_sync.set()  # unblock any wait

    def _save_local_checkpoint(self) -> None:
        """Save a local checkpoint for fault recovery."""
        if not self._metrics_history:
            return
        ckpt_dir = step_directory(
            self.settings.training.checkpoint_dir + "/local", self.global_step
        )
        meta = SnapshotMeta(
            model_name=self.settings.model.name,
            global_step=self.global_step,
            loss=self._metrics_history[-1].loss if self._metrics_history else float("inf"),
            total_compute_hours=(time.time() - self.training_start_time) / 3600,
            contributors=[self.device_profile.hostname],
            description="Local worker checkpoint",
        )
        save_snapshot(ckpt_dir, self.model, self.diloco.inner_optimizer, meta)
