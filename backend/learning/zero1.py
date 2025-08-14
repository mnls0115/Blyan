#!/usr/bin/env python3
"""
Zero-1 Optimizer State Sharding using native PyTorch ZeroRedundancyOptimizer.

This provides ZeRO Stage-1 equivalent behavior without external deps.

Usage:
  - Initialize torch.distributed process group before calling wrap.
  - Call wrap_optimizer_with_zero1 on a standard optimizer to shard states.
"""

from __future__ import annotations

from typing import Iterable, Optional
import logging

import torch

logger = logging.getLogger(__name__)


def init_distributed_from_env(backend: str = "nccl", init_method: str = "env://") -> None:
    """Initialize torch.distributed from environment variables if not already initialized.
    Requires WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT to be set by the launcher.
    """
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available on this installation")
    if torch.distributed.is_initialized():
        return
    torch.distributed.init_process_group(backend=backend, init_method=init_method)
    logger.info("Initialized torch.distributed process group via env: world_size=%s rank=%s", 
                torch.distributed.get_world_size(), torch.distributed.get_rank())


def wrap_optimizer_with_zero1(
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    overlap_with_ddp: bool = False,
) -> torch.optim.Optimizer:
    """Wrap a standard optimizer with ZeroRedundancyOptimizer to shard optimizer states.
    This is similar to ZeRO-1 (optimizer state partitioning).
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed is not initialized; cannot enable ZeRO-1")

    from torch.distributed.optim import ZeroRedundancyOptimizer  # lazy import

    zero_optim = ZeroRedundancyOptimizer(
        params,
        optimizer_class=type(optimizer),
        overlap_with_ddp=overlap_with_ddp,
        # copy base optimizer defaults
        **{k: v for k, v in optimizer.defaults.items()}
    )

    # Transfer per-parameter state if needed is non-trivial; encourage enabling at construction time.
    logger.info("Enabled ZeRO-1 using ZeroRedundancyOptimizer for optimizer class %s", type(optimizer).__name__)
    return zero_optim

