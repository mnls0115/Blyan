#!/usr/bin/env python3
"""
Lightweight in-process metrics collectors for pipeline parallelism.

These are read by the Prometheus exporter to expose as metrics.
Thread-safe via asyncio locks and simple atomic updates.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Histogram:
    buckets: Dict[str, int] = field(default_factory=lambda: {
        "le_0_01": 0,
        "le_0_05": 0,
        "le_0_1": 0,
        "le_0_25": 0,
        "le_0_5": 0,
        "le_1_0": 0,
        "gt_1_0": 0,
    })
    count: int = 0
    sum: float = 0.0

    def observe(self, value_seconds: float) -> None:
        self.count += 1
        self.sum += value_seconds
        if value_seconds <= 0.01:
            self.buckets["le_0_01"] += 1
        elif value_seconds <= 0.05:
            self.buckets["le_0_05"] += 1
        elif value_seconds <= 0.1:
            self.buckets["le_0_1"] += 1
        elif value_seconds <= 0.25:
            self.buckets["le_0_25"] += 1
        elif value_seconds <= 0.5:
            self.buckets["le_0_5"] += 1
        elif value_seconds <= 1.0:
            self.buckets["le_1_0"] += 1
        else:
            self.buckets["gt_1_0"] += 1


class PipelineMetrics:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # occupancy per stage_id: value in [0.0, 1.0]
        self.stage_occupancy: Dict[int, float] = {}
        # RPC error counter
        self.rpc_errors: int = 0
        # Counters for recovery/fallbacks
        self.round_failures: int = 0
        self.pipeline_resets: int = 0
        self.fallback_activations: int = 0
        # Gauges for current pipeline status
        self.fallback_mode_active: int = 0
        self.current_stage_count: int = 0
        # microbatch wait histogram (seconds)
        self.microbatch_wait_hist = Histogram()
        # last partition plan id and drift value (0 good .. 1 high drift)
        self.partition_plan_id: Optional[str] = None
        self.partition_drift: float = 0.0
        # device profile last update timestamps and staleness seconds
        self.device_profile_last_update: Dict[str, float] = {}

    async def set_stage_occupancy(self, stage_idx: int, occupancy: float) -> None:
        async with self._lock:
            self.stage_occupancy[stage_idx] = max(0.0, min(1.0, occupancy))

    async def incr_rpc_error(self, inc: int = 1) -> None:
        async with self._lock:
            self.rpc_errors += inc

    async def incr_round_failure(self, inc: int = 1) -> None:
        async with self._lock:
            self.round_failures += inc

    async def incr_pipeline_reset(self, inc: int = 1) -> None:
        async with self._lock:
            self.pipeline_resets += inc

    async def incr_fallback(self, inc: int = 1) -> None:
        async with self._lock:
            self.fallback_activations += inc

    async def observe_microbatch_wait(self, wait_seconds: float) -> None:
        async with self._lock:
            self.microbatch_wait_hist.observe(max(0.0, wait_seconds))

    async def set_partition_plan(self, plan_id: str) -> None:
        async with self._lock:
            self.partition_plan_id = plan_id
            self.partition_drift = 0.0

    async def set_partition_drift(self, drift: float) -> None:
        async with self._lock:
            self.partition_drift = max(0.0, min(1.0, drift))

    async def mark_device_profile_update(self, node_id: str) -> None:
        async with self._lock:
            self.device_profile_last_update[node_id] = time.time()

    async def export_snapshot(self) -> Dict[str, any]:
        async with self._lock:
            now = time.time()
            staleness = {
                nid: max(0.0, now - ts) for nid, ts in self.device_profile_last_update.items()
            }
            return {
                "stage_occupancy": dict(self.stage_occupancy),
                "rpc_errors": self.rpc_errors,
                "round_failures": self.round_failures,
                "pipeline_resets": self.pipeline_resets,
                "fallback_activations": self.fallback_activations,
                "fallback_mode_active": self.fallback_mode_active,
                "current_stage_count": self.current_stage_count,
                "microbatch_wait_hist": {
                    "buckets": dict(self.microbatch_wait_hist.buckets),
                    "count": self.microbatch_wait_hist.count,
                    "sum": self.microbatch_wait_hist.sum,
                },
                "partition_plan_id": self.partition_plan_id,
                "partition_drift": self.partition_drift,
                "device_profile_staleness": staleness,
            }

    async def set_fallback_active(self, active: bool) -> None:
        async with self._lock:
            self.fallback_mode_active = 1 if active else 0

    async def set_current_stage_count(self, count: int) -> None:
        async with self._lock:
            self.current_stage_count = max(0, int(count))


_global_metrics: Optional[PipelineMetrics] = None


def get_pipeline_metrics() -> PipelineMetrics:
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = PipelineMetrics()
    return _global_metrics

