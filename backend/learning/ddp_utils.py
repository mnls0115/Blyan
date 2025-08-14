#!/usr/bin/env python3
"""
DDP Utilities (minimal, production-ready wrappers)
"""

from __future__ import annotations

import os
import torch
import torch.distributed as dist


def init_distributed_if_needed() -> None:
    if dist.is_available() and not dist.is_initialized():
        backend = os.getenv('TORCH_DDP_BACKEND', 'nccl' if torch.cuda.is_available() else 'gloo')
        dist.init_process_group(backend=backend)


def wrap_model_ddp(model: torch.nn.Module) -> torch.nn.Module:
    if dist.is_available() and dist.is_initialized():
        return torch.nn.parallel.DistributedDataParallel(model)
    return model

#!/usr/bin/env python3
"""
DDP utilities for wrapping models and initializing distributed backend.

Relies on torch.distributed and uses the env:// rendezvous for production launchers.
"""

from __future__ import annotations

from typing import Optional
import logging

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def init_distributed_if_needed(backend: str = "nccl", init_method: str = "env://") -> bool:
    """Initialize process group from env if not already. Returns True if initialized here."""
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available")
    if torch.distributed.is_initialized():
        return False
    torch.distributed.init_process_group(backend=backend, init_method=init_method)
    logger.info("DDP process group initialized: world_size=%s rank=%s", torch.distributed.get_world_size(), torch.distributed.get_rank())
    return True


def wrap_model_ddp(
    model: torch.nn.Module,
    device_id: Optional[int] = None,
    broadcast_buffers: bool = True,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
) -> torch.nn.Module:
    """Wrap a model with DistributedDataParallel on the given device.
    Assumes process group already initialized.
    """
    if device_id is None and torch.cuda.is_available():
        device_id = torch.cuda.current_device()
    if device_id is not None:
        model = model.to(device_id)
    ddp = DDP(
        model,
        device_ids=[device_id] if device_id is not None else None,
        output_device=device_id,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
    )
    return ddp

