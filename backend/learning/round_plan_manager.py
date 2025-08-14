#!/usr/bin/env python3
"""
Round Plan Manager

Coordinates selection of partition plans per round/epoch, tracks failures,
and provides simple fallback to latest stable plan.
"""

from __future__ import annotations

import time
from typing import Optional, List, Dict
import logging

from .partition_plan_registry import PartitionPlanRegistry, PartitionPlan
from .pipeline_metrics import get_pipeline_metrics

logger = logging.getLogger(__name__)


class RoundPlanManager:
    def __init__(self, registry: PartitionPlanRegistry):
        self.registry = registry
        self.current_epoch_id: Optional[str] = None
        self.current_round_index: int = 0
        self.round_ids: List[str] = []
        self.failure_counts: Dict[str, int] = {}
        # Env-tunable failure tolerance
        try:
            import os
            self.max_failures_before_fallback = int(os.getenv('BLYAN_ROUND_MAX_FAILURES', '3'))
        except Exception:
            self.max_failures_before_fallback = 3
        self.metrics = get_pipeline_metrics()

    def load_epoch(self, epoch_id: str) -> None:
        """Initialize manager for a new epoch, resetting round counters.
        This will scan the registry index for round_ids belonging to the epoch.
        """
        self.current_epoch_id = epoch_id
        self.current_round_index = 0
        # Best effort: collect rounds from registry index by checking saved files
        # Since we don't store a list, we try round_0..round_31
        found = []
        for i in range(32):
            plan = self.registry.resolve_for_round(epoch_id, f"round_{i}")
            if plan:
                found.append(plan.round_id)
        self.round_ids = sorted(found)
        self.failure_counts.clear()
        logger.info("RoundPlanManager loaded epoch %s with rounds: %s", epoch_id, self.round_ids)

    def get_current_plan(self) -> Optional[PartitionPlan]:
        if not self.current_epoch_id:
            return None
        if not self.round_ids:
            # Fallback to latest stable
            return self.registry.latest_stable()
        round_id = self.round_ids[self.current_round_index % len(self.round_ids)]
        return self.registry.resolve_for_round(self.current_epoch_id, round_id)

    async def mark_success(self) -> None:
        """Advance to next round on success and reset failure counters."""
        self.current_round_index += 1
        await self.metrics.set_partition_drift(0.0)

    async def mark_failure(self) -> Optional[PartitionPlan]:
        """Record failure and return a fallback plan if threshold exceeded."""
        plan = self.get_current_plan()
        if not plan:
            return None
        pid = plan.plan_id
        self.failure_counts[pid] = self.failure_counts.get(pid, 0) + 1
        logger.warning("Round plan %s failed (count=%d)", pid, self.failure_counts[pid])
        await self.metrics.incr_round_failure()
        # Simple alert hook
        try:
            from backend.security.monitoring import record_security_event
            record_security_event("pipeline_rpc_errors", "round_plan_manager", {"plan_id": pid, "count": self.failure_counts[pid]})
        except Exception:
            pass
        if self.failure_counts[pid] >= self.max_failures_before_fallback:
            latest = self.registry.latest_stable()
            if latest and latest.plan_id != pid:
                await self.metrics.set_partition_plan(latest.plan_id)
                await self.metrics.incr_fallback()
                logger.info("Falling back to latest stable plan %s", latest.plan_id)
                return latest
        return None

