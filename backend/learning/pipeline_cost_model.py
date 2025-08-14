#!/usr/bin/env python3
"""
Pipeline Cost Model (Skeleton)

Provides coarse-grained parameter/activation/compute cost estimates per layer
to guide partitioning for pipeline-parallel training across heterogeneous nodes.

This intentionally uses simple, explainable formulas that can be refined over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class LayerCostEstimate:
    layer_index: int
    param_bytes: int
    activation_bytes_per_token: int
    flops_per_token_forward: float
    flops_per_token_backward: float


@dataclass
class ModelCostReport:
    layer_costs: List[LayerCostEstimate]
    total_param_bytes: int
    activation_bytes_per_token: int
    total_flops_per_token_forward: float
    total_flops_per_token_backward: float


class PipelineCostModel:
    """Coarse cost estimator for Transformer-like blocks.

    Assumptions (can be tuned):
      - Transformer block with (Q,K,V,O) projections and FFN (2 matmuls)
      - FLOPs approximated as 2 * parameter_count for matmul (per token)
      - Activation memory per token approximated as hidden_size * dtype_bytes * factor
    """

    def __init__(self, dtype_bytes: int = 2, activation_factor: float = 2.0):
        self.dtype_bytes = dtype_bytes
        self.activation_factor = activation_factor

    def estimate_layer(self, layer_index: int, hidden_size: int, ffn_hidden: int, num_heads: int) -> LayerCostEstimate:
        # Parameter counts (approx) for projections and FFN
        # Projections: Q, K, V, O each ~ hidden_size * hidden_size
        proj_params = 4 * hidden_size * hidden_size
        # FFN: hidden_size -> ffn_hidden -> hidden_size (two matmuls)
        ffn_params = hidden_size * ffn_hidden + ffn_hidden * hidden_size
        param_count = proj_params + ffn_params
        param_bytes = param_count * self.dtype_bytes

        # Activation memory per token: proportional to hidden size
        activation_bytes_per_token = int(self.activation_factor * hidden_size * self.dtype_bytes)

        # FLOPs per token (very coarse): 2 * params
        flops_per_token_forward = float(2 * param_count)
        flops_per_token_backward = float(4 * param_count)  # backward ~ 2x forward

        return LayerCostEstimate(
            layer_index=layer_index,
            param_bytes=param_bytes,
            activation_bytes_per_token=activation_bytes_per_token,
            flops_per_token_forward=flops_per_token_forward,
            flops_per_token_backward=flops_per_token_backward,
        )

    def estimate_model(
        self,
        num_layers: int,
        hidden_size: int,
        ffn_hidden: int,
        num_heads: int,
    ) -> ModelCostReport:
        layer_costs: List[LayerCostEstimate] = []
        total_param_bytes = 0
        total_flops_f = 0.0
        total_flops_b = 0.0
        act_bytes_per_token = 0

        for i in range(num_layers):
            cost = self.estimate_layer(i, hidden_size, ffn_hidden, num_heads)
            layer_costs.append(cost)
            total_param_bytes += cost.param_bytes
            total_flops_f += cost.flops_per_token_forward
            total_flops_b += cost.flops_per_token_backward
            act_bytes_per_token = max(act_bytes_per_token, cost.activation_bytes_per_token)

        return ModelCostReport(
            layer_costs=layer_costs,
            total_param_bytes=total_param_bytes,
            activation_bytes_per_token=act_bytes_per_token,
            total_flops_per_token_forward=total_flops_f,
            total_flops_per_token_backward=total_flops_b,
        )

    @staticmethod
    def estimate_runtime_ms(total_flops: float, batch_size: int, seq_len: int, tflops_device: float) -> float:
        """Rough runtime estimator: FLOPs / (TFLOPS) with ms conversion.
        Note: highly idealized; ignores comms and memory stalls.
        """
        if tflops_device <= 0:
            return float('inf')
        ops = total_flops * batch_size * seq_len
        seconds = ops / (tflops_device * 1e12)
        return seconds * 1000.0

