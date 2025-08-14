#!/usr/bin/env python3
"""
Memory policy recommender for pipeline training.

Decides whether to enable ZeRO-1 optimizer state partitioning and
activation checkpointing based on estimated memory footprint.
"""

from __future__ import annotations

from typing import List, Tuple

from .pipeline_cost_model import LayerCostEstimate, ModelCostReport


def recommend_memory_policy(
    report: ModelCostReport,
    device_vram_gb: List[float],
    batch_size: int,
    seq_len: int,
    safety_margin: float = 0.8,
) -> Tuple[bool, bool]:
    """
    Returns (use_zero1, use_activation_checkpointing).

    Heuristic:
    - If any device would exceed soft cap with full optimizer states, enable ZeRO-1.
    - If activation memory dominates beyond soft cap, enable checkpointing.
    """
    # Rough per-layer activation per token upper-bound
    act_bytes = report.activation_bytes_per_token * batch_size * seq_len
    param_bytes_total = report.total_param_bytes

    # Assume equal shard for params across N devices for rough calculation
    n = max(1, len(device_vram_gb))
    param_bytes_per_device = param_bytes_total / n

    # Optimizer state ~ 2x params (e.g., Adam moments), ZeRO-1 partitions optimizer states across data-parallel ranks
    optimizer_bytes_full = 2.0 * param_bytes_per_device
    optimizer_bytes_zero1 = optimizer_bytes_full / n

    # Soft cap per device in bytes
    soft_caps = [v * 1e9 * safety_margin for v in device_vram_gb]

    need_zero1 = False
    need_ckpt = False
    for cap in soft_caps:
        # Without checkpointing we assume full activation kept; with checkpointing ~ half (heuristic)
        if (param_bytes_per_device + optimizer_bytes_full + act_bytes) > cap:
            need_zero1 = True
        if (param_bytes_per_device + (optimizer_bytes_zero1 if need_zero1 else optimizer_bytes_full) + act_bytes) > cap:
            need_ckpt = True

    return need_zero1, need_ckpt

