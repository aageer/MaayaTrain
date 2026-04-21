"""DiLoCo — Distributed Low-Communication training algorithm.

Implements the algorithm from Douillard et al. (arXiv:2311.08105), validated
at scale by PrimeIntellect's OpenDiLoCo (arXiv:2407.07852).

Overview
--------
DiLoCo decouples local training from global synchronization:

1. **Inner loop** — each worker runs *H* steps of AdamW independently.
2. **Outer loop** — workers compute *pseudo-gradients* (difference between
   the global snapshot and their locally-trained params). The coordinator
   averages these and applies a Nesterov-momentum SGD step to the global
   model.

This reduces communication by a factor of *H* (default 500×) compared to
traditional DDP that syncs every step.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from ..settings import DiLoCoConfig

logger = logging.getLogger("maayatrain.diloco")

# Default chunk size for stream-chunked median (in number of elements).
# 5M floats × 4 bytes × N_workers fits comfortably in consumer RAM.
_MEDIAN_CHUNK_SIZE = 5_000_000


class DiLoCoEngine:
    """Manages the DiLoCo inner/outer loop lifecycle.

    This object is shared by both the coordinator and worker roles:
    * Workers call :meth:`snapshot_global`, :meth:`run_inner_steps`,
      and :meth:`compute_pseudo_gradient`.
    * The coordinator calls :meth:`apply_outer_step` with the collected
      pseudo-gradients from all workers.

    Parameters
    ----------
    model : nn.Module
        The model being trained (lives on the worker's device).
    config : DiLoCoConfig
        Algorithm hyperparameters.
    device : torch.device
        Where tensors should live.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DiLoCoConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

        # θ_global snapshot taken before each inner loop
        self._global_snapshot: Optional[Dict[str, Tensor]] = None

        # Outer optimizer Nesterov momentum buffer (v)
        self._momentum_buffer: Dict[str, Tensor] = {}

        # Build inner optimizer
        self.inner_optimizer = self._build_inner_optimizer()

        self.outer_step_count = 0

    def _build_inner_optimizer(self) -> torch.optim.Optimizer:
        """Create the inner AdamW optimizer from config."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.inner_lr,
            weight_decay=self.config.inner_weight_decay,
        )

    # ------------------------------------------------------------------
    # Inner loop (executed on each worker)
    # ------------------------------------------------------------------

    def snapshot_global(self) -> None:
        """Save θ_global — a deep copy of current model params.

        Must be called **before** the inner training loop begins.
        """
        self._global_snapshot = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }
        logger.debug("Global snapshot saved (%d params)", len(self._global_snapshot))

    def compute_pseudo_gradient(self) -> Dict[str, Tensor]:
        """Compute Δθ = θ_global − θ_local (pseudo-gradient).

        After the inner loop, the model params are θ_local.
        The pseudo-gradient captures "how far we moved" during local training.
        Note the sign convention: Δθ = θ_global − θ_local, so it points
        *back* toward the starting point. The outer step will move the
        global model in the averaged direction of all workers' exploration.

        Returns
        -------
        dict[str, Tensor]
            Named pseudo-gradient tensors (on CPU for transmission).
        """
        if self._global_snapshot is None:
            raise RuntimeError("snapshot_global() must be called before compute_pseudo_gradient()")

        pseudo_grads: Dict[str, Tensor] = {}
        for name, param in self.model.named_parameters():
            delta = self._global_snapshot[name].to(self.device) - param.detach()
            pseudo_grads[name] = delta.cpu()

        logger.debug("Pseudo-gradient computed (%d tensors)", len(pseudo_grads))
        return pseudo_grads

    # ------------------------------------------------------------------
    # Outer loop (executed on the coordinator)
    # ------------------------------------------------------------------

    def apply_outer_step(
        self,
        pseudo_gradient_list: List[Dict[str, Tensor]],
        *,
        aggregation: str = "mean",
    ) -> None:
        """Average pseudo-gradients and apply the outer Nesterov SGD step.

        Supports two aggregation modes:
        - **mean**: Standard averaging (default, from DiLoCo paper).
        - **median**: Coordinate-wise median (Byzantine-tolerant), using
          a stream-chunked implementation that avoids OOM on large models.
          Tolerates up to 1/3 malicious or faulty workers. Based on
          robust aggregation research (Blanchard et al., 2017; SPARTA 2024).

        This implements the outer update::

            Δθ_agg = aggregate(Δθ_1, …, Δθ_n)  # mean or median
            v = β·v + Δθ_agg
            θ_global = θ_global − η·(Δθ_agg + β·v)     [Nesterov]

        Parameters
        ----------
        pseudo_gradient_list : list[dict[str, Tensor]]
            One dict per worker, each containing named pseudo-gradient tensors.
        aggregation : str
            Aggregation method: "mean" (default) or "median" (Byzantine-tolerant).
        """
        n_workers = len(pseudo_gradient_list)
        if n_workers == 0:
            logger.warning("No pseudo-gradients received — skipping outer step")
            return

        η = self.config.outer_lr
        β = self.config.outer_momentum
        use_nesterov = self.config.nesterov

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Aggregate pseudo-gradients
                if aggregation == "median":
                    delta_agg = _chunked_median(
                        [pg[name] for pg in pseudo_gradient_list],
                        device=self.device,
                    )
                else:
                    # Mean: accumulate in-place to avoid stacking full tensors
                    # Force FP32 to prevent FP16 overflow (max 65504) when
                    # summing many workers' gradients.
                    delta_agg = pseudo_gradient_list[0][name].to(self.device).float()
                    for pg in pseudo_gradient_list[1:]:
                        delta_agg.add_(pg[name].to(self.device).float())
                    delta_agg.div_(n_workers)

                # Momentum update: v = β·v + Δθ_agg
                if name not in self._momentum_buffer:
                    self._momentum_buffer[name] = torch.zeros_like(delta_agg)
                v = self._momentum_buffer[name]
                v.mul_(β).add_(delta_agg)

                # Parameter update
                if use_nesterov:
                    # Nesterov: θ = θ − η·(Δθ_agg + β·v)
                    param.sub_(η * (delta_agg + β * v))
                else:
                    # Standard momentum: θ = θ − η·v
                    param.sub_(η * v)

        self.outer_step_count += 1
        logger.info(
            "Outer step %d applied (%s of %d workers)",
            self.outer_step_count,
            aggregation,
            n_workers,
        )

    def apply_outer_step_weighted(
        self,
        pseudo_gradient_list: List[Dict[str, Tensor]],
        local_steps_list: List[int],
        *,
        aggregation: str = "mean",
    ) -> None:
        """Compute-proportional outer step: weight each worker by local steps.

        In time-bounded sync mode, faster workers complete more steps and
        contribute proportionally more to the aggregated gradient::

            weight_k = local_steps_k / sum(local_steps)
            Δθ_agg = Σ weight_k · Δθ_k

        For **median** aggregation (Byzantine mode), compute-proportionality
        is bypassed — standard unweighted median is used instead, since
        weighting would break the Byzantine fault tolerance guarantee.

        Parameters
        ----------
        pseudo_gradient_list : list[dict[str, Tensor]]
            One dict per worker.
        local_steps_list : list[int]
            Number of inner steps each worker completed in its time window.
        aggregation : str
            "mean" (default, weighted) or "median" (unweighted, Byzantine).
        """
        n_workers = len(pseudo_gradient_list)
        if n_workers == 0:
            logger.warning("No pseudo-gradients received — skipping outer step")
            return

        # Median: fall back to unweighted (Byzantine safety)
        if aggregation == "median":
            logger.warning(
                "Compute-proportional weighting bypassed for median aggregation "
                "(Byzantine fault tolerance requires equal weight)"
            )
            return self.apply_outer_step(
                pseudo_gradient_list, aggregation="median"
            )

        # Compute weights from local step counts
        total_steps = sum(local_steps_list)
        if total_steps == 0:
            logger.warning("All workers reported 0 steps — skipping outer step")
            return

        weights = [s / total_steps for s in local_steps_list]
        logger.info(
            "Weighted aggregation: steps=%s, weights=%s",
            local_steps_list,
            [f"{w:.3f}" for w in weights],
        )

        η = self.config.outer_lr
        β = self.config.outer_momentum
        use_nesterov = self.config.nesterov

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Weighted sum: Σ weight_k · grad_k (in-place accumulation)
                # Force FP32 to prevent FP16 overflow (max 65504) when
                # accumulating many workers' weighted gradients.
                delta_agg = pseudo_gradient_list[0][name].to(self.device).float() * weights[0]
                for pg, w in zip(pseudo_gradient_list[1:], weights[1:]):
                    delta_agg.add_(pg[name].to(self.device).float(), alpha=w)

                # Momentum + parameter update (same as apply_outer_step)
                if name not in self._momentum_buffer:
                    self._momentum_buffer[name] = torch.zeros_like(delta_agg)
                v = self._momentum_buffer[name]
                v.mul_(β).add_(delta_agg)

                if use_nesterov:
                    param.sub_(η * (delta_agg + β * v))
                else:
                    param.sub_(η * v)

        self.outer_step_count += 1
        logger.info(
            "Weighted outer step %d applied (%d workers, %d total steps)",
            self.outer_step_count,
            n_workers,
            total_steps,
        )

    def compute_streaming_shards(
        self, num_shards: int = 4
    ) -> List[List[str]]:
        """Split parameter names into streaming sync shards.

        Implements the Streaming DiLoCo approach (arXiv:2501.18512):
        instead of syncing all parameters at once, split them into
        K groups that can be synced incrementally during training.

        Parameters
        ----------
        num_shards : int
            Number of shards to split parameters into.

        Returns
        -------
        list[list[str]]
            Each inner list contains parameter names for one shard.
        """
        param_names = [n for n, _ in self.model.named_parameters()]
        shards: List[List[str]] = [[] for _ in range(num_shards)]
        for i, name in enumerate(param_names):
            shards[i % num_shards].append(name)
        return shards

    def apply_outer_step_shard(
        self,
        pseudo_gradient_list: List[Dict[str, Tensor]],
        shard_names: List[str],
        *,
        aggregation: str = "mean",
    ) -> None:
        """Apply the outer step to only a subset of parameters (one shard).

        Used for streaming sync: sync and update parameters shard by shard
        while continuing training on other shards.

        Parameters
        ----------
        pseudo_gradient_list : list[dict[str, Tensor]]
            Full pseudo-gradients from all workers.
        shard_names : list[str]
            Parameter names in this shard.
        aggregation : str
            "mean" or "median".
        """
        n_workers = len(pseudo_gradient_list)
        if n_workers == 0:
            return

        η = self.config.outer_lr
        β = self.config.outer_momentum
        use_nesterov = self.config.nesterov

        with torch.no_grad():
            param_dict = dict(self.model.named_parameters())
            for name in shard_names:
                if name not in param_dict:
                    continue
                param = param_dict[name]

                if aggregation == "median":
                    delta_agg = _chunked_median(
                        [pg[name] for pg in pseudo_gradient_list],
                        device=self.device,
                    )
                else:
                    # Force FP32 to prevent FP16 overflow during accumulation
                    delta_agg = pseudo_gradient_list[0][name].to(self.device).float()
                    for pg in pseudo_gradient_list[1:]:
                        delta_agg.add_(pg[name].to(self.device).float())
                    delta_agg.div_(n_workers)

                if name not in self._momentum_buffer:
                    self._momentum_buffer[name] = torch.zeros_like(delta_agg)
                v = self._momentum_buffer[name]
                v.mul_(β).add_(delta_agg)

                if use_nesterov:
                    param.sub_(η * (delta_agg + β * v))
                else:
                    param.sub_(η * v)

        logger.debug("Streaming shard applied (%d params)", len(shard_names))

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def load_global_weights(self, state_dict: Dict[str, Tensor]) -> None:
        """Replace the model's parameters with a new global state dict.

        Called on workers after receiving updated weights from the coordinator.
        """
        self.model.load_state_dict(state_dict)
        logger.debug("Global weights loaded into model")

    def get_global_weights(self) -> Dict[str, Tensor]:
        """Return the current model state dict (CPU copies)."""
        return {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
        }

    def reset_inner_optimizer(self) -> None:
        """Reset the inner optimizer state for the next inner loop.

        Called at the start of each inner loop so the AdamW momentum buffers
        don't carry over across outer steps.
        """
        self.inner_optimizer = self._build_inner_optimizer()


# ---------------------------------------------------------------------------
# Stream-chunked median aggregation (module-level helper)
# ---------------------------------------------------------------------------


def _chunked_median(
    tensors: List[Tensor],
    *,
    device: torch.device,
    chunk_size: int = _MEDIAN_CHUNK_SIZE,
) -> Tensor:
    """Compute coordinate-wise median of a list of tensors without OOM.

    Instead of stacking all worker tensors at once (which multiplies
    memory by N_workers), this function:

    1. Allocates a single flat output buffer on *device*.
    2. Iterates through the flattened tensors in chunks of *chunk_size*.
    3. For each chunk, stacks only that slice across workers, computes
       ``torch.median(dim=0)``, and writes to the output buffer.

    This reduces peak memory from O(N × params) to O(N × chunk_size).

    Parameters
    ----------
    tensors : list[Tensor]
        One tensor per worker (same shape).
    device : torch.device
        Device for the output tensor and intermediate computation.
    chunk_size : int
        Number of elements per chunk (default 5M).

    Returns
    -------
    Tensor
        Median tensor on *device*, same shape as inputs.
    """
    n_workers = len(tensors)
    ref_shape = tensors[0].shape

    if n_workers == 1:
        return tensors[0].to(device)

    # Flatten all worker tensors (keep on CPU to save device memory)
    flats = [t.reshape(-1) for t in tensors]
    total = flats[0].numel()
    out = torch.empty(total, dtype=flats[0].dtype, device=device)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        # Stack only the current chunk across workers → (N, chunk_len)
        chunk_stack = torch.stack(
            [f[start:end].to(device) for f in flats]
        )
        # .clone() breaks any reference chain from the median return tuple
        # back to chunk_stack, ensuring del actually frees memory.
        out[start:end] = chunk_stack.median(dim=0).values.clone()
        del chunk_stack  # free immediately

    return out.reshape(ref_shape)
