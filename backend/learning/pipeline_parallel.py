#!/usr/bin/env python3
"""
Pipeline Parallel Trainer (Skeleton)

Implements a basic 1F1B schedule skeleton with activation/grad RPC hooks.
This file is a non-functional scaffold to be wired with real model layers and RPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import asyncio
import time

from .pipeline_rpc import PipelineRPCClient
from .pipeline_rpc_client_grpc import PipelineGrpcClient
from .pipeline_metrics import get_pipeline_metrics


@dataclass
class StageConfig:
    device_id: str
    start_layer: int
    end_layer: int


class PipelineParallelTrainer:
    def __init__(self, stages: List[StageConfig], rpc_peers: Optional[List[str]] = None, rpc_timeout_s: float = 5.0, use_zero1: bool = False, use_activation_checkpointing: bool = False, rpc_transport: str = "http"):
        self.stages = stages
        self.rpc_peers = rpc_peers or []
        self.rpc_timeout_s = rpc_timeout_s
        self.metrics = get_pipeline_metrics()
        self.use_zero1 = use_zero1
        self.use_activation_checkpointing = use_activation_checkpointing
        self.rpc_transport = rpc_transport

    async def run_1f1b(self, microbatches: List[Dict[str, Any]], timeout_s: float = 10.0) -> None:
        """Run 1F1B pipeline loop (skeleton).

        Hooks to implement:
          - send_activations(stage_idx, mb)
          - recv_activations(stage_idx)
          - forward_stage(stage_idx, acts)
          - send_grads(stage_idx, grads)
          - recv_grads(stage_idx)
          - backward_stage(stage_idx, grads)
        """

        num_stages = len(self.stages)
        num_mbs = len(microbatches)

        # Warmup forwards
        for t in range(num_stages - 1):
            if t < num_mbs:
                await self._forward_step(stage_idx=t, microbatch=microbatches[t])

        # 1F1B steady state
        for t in range(num_mbs - (num_stages - 1)):
            f_idx = (num_stages - 1)
            b_idx = 0
            # forward newest micro-batch at last stage
            await self._forward_step(stage_idx=f_idx, microbatch=microbatches[t + (num_stages - 1)])
            # backward oldest in flight at first stage
            await self._backward_step(stage_idx=b_idx)

        # Drain backwards
        for t in range(num_stages - 1):
            await self._backward_step(stage_idx=t + 1)

    async def _forward_step(self, stage_idx: int, microbatch: Dict[str, Any]) -> None:
        start = time.time()
        try:
            # Example: send activations to next stage peer if exists
            if stage_idx < len(self.stages) - 1 and stage_idx < len(self.rpc_peers):
                peer = self.rpc_peers[stage_idx]
                if peer:
                    if self.rpc_transport == "grpc":
                        client = PipelineGrpcClient(peer, timeout_s=self.rpc_timeout_s)
                        try:
                            await client.send_activations(stage_idx + 1, microbatch.get("id", "mb"), b"acts")
                        finally:
                            await client.close()
                    else:
                        async with PipelineRPCClient(peer, timeout_s=self.rpc_timeout_s) as client:
                            await client.send_activations(stage_idx + 1, microbatch.get("id", "mb"), b"acts")
            await asyncio.sleep(0)
        finally:
            wait = max(0.0, time.time() - start)
            await self.metrics.observe_microbatch_wait(wait)

    async def _backward_step(self, stage_idx: int) -> None:
        start = time.time()
        try:
            # Example: receive gradients from next stage during backward
            if stage_idx < len(self.stages) - 1 and stage_idx < len(self.rpc_peers):
                peer = self.rpc_peers[stage_idx]
                if peer:
                    if self.rpc_transport == "grpc":
                        client = PipelineGrpcClient(peer, timeout_s=self.rpc_timeout_s)
                        try:
                            _ = await client.recv_grads(stage_idx + 1, "mb")
                        finally:
                            await client.close()
                    else:
                        async with PipelineRPCClient(peer, timeout_s=self.rpc_timeout_s) as client:
                            _ = await client.recv_grads(stage_idx + 1, "mb")
            await asyncio.sleep(0)
        finally:
            wait = max(0.0, time.time() - start)
            await self.metrics.observe_microbatch_wait(wait)

