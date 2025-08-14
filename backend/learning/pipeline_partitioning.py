#!/usr/bin/env python3
"""
Pipeline Partitioning Solver (Skeleton)

Given layer cost report and heterogeneous device profiles, produce stage
boundaries that satisfy VRAM constraints and balance compute load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

from .pipeline_cost_model import LayerCostEstimate, ModelCostReport


@dataclass
class DeviceConstraint:
    device_id: str
    vram_gb: float
    tflops_est: float


@dataclass
class StageAssignment:
    device_id: str
    start_layer: int
    end_layer: int  # inclusive


class GreedyPartitioner:
    """Simple greedy partitioner to get initial boundaries.

    Strategy:
      - Assign consecutive layers to fastest device until either VRAM soft cap or
        target compute per stage reached, then move to next device.
      - If devices are fewer than partitions ideal, last device gets remainder.
    """

    def __init__(self, vram_soft_util: float = 0.8):
        self.vram_soft_util = vram_soft_util

    def solve(
        self,
        report: ModelCostReport,
        devices: List[DeviceConstraint],
        batch_size: int,
        seq_len: int,
        activation_bytes_per_token: int,
    ) -> List[StageAssignment]:
        # Sort devices by performance desc
        devices_sorted = sorted(devices, key=lambda d: d.tflops_est, reverse=True)
        assignments: List[StageAssignment] = []

        layers = report.layer_costs
        n = len(layers)
        i = 0
        for dev in devices_sorted:
            if i >= n:
                break
            # Compute soft vram cap (bytes)
            vram_cap_bytes = int(dev.vram_gb * 1e9 * self.vram_soft_util)

            start = i
            param_acc = 0
            # target: coarse equal flops per device
            target_flops = (report.total_flops_per_token_forward + report.total_flops_per_token_backward) / max(1, len(devices_sorted))
            flops_acc = 0.0

            while i < n:
                layer = layers[i]
                # param memory accumulation
                param_acc += layer.param_bytes
                # activation memory for this stage (approx peak per micro-batch token)
                act_mem = activation_bytes_per_token * batch_size * seq_len
                est_mem = param_acc + act_mem
                if est_mem > vram_cap_bytes and i > start:
                    break
                flops_acc += (layer.flops_per_token_forward + layer.flops_per_token_backward)
                # stop if flops roughly reach target
                if flops_acc >= target_flops and i > start:
                    i += 1
                    break
                i += 1

            end = min(i - 1, n - 1)
            if end < start:
                end = start
                i = start + 1
            assignments.append(StageAssignment(device_id=dev.device_id, start_layer=start, end_layer=end))

        # If layers remain, append to last device
        if i < n and assignments:
            last = assignments[-1]
            assignments[-1] = StageAssignment(device_id=last.device_id, start_layer=last.start_layer, end_layer=n - 1)

        return assignments

