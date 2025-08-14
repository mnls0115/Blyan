#!/usr/bin/env python3
"""
Stage Reallocator

Proposes a new partition plan by replacing stages whose nodes are stale or unhealthy.
Uses ExpertNodeRegistry and NodeReputationManager to select replacements.
"""

from __future__ import annotations

import time
from typing import Optional, List
import logging

from .partition_plan_registry import PartitionPlan, StagePlan, PartitionPlanRegistry

logger = logging.getLogger(__name__)


def _is_node_stale(node, now: float, heartbeat_timeout: float) -> bool:
    try:
        return (now - float(node.last_heartbeat)) > heartbeat_timeout
    except Exception:
        return True


def _meets_device_constraints(node, min_vram_gb: float, min_tflops: float) -> bool:
    dp = getattr(node, 'device_profile', None)
    if not dp:
        return False
    try:
        if dp.vram_gb is not None and float(dp.vram_gb) < min_vram_gb:
            return False
        if dp.tflops_est is not None and float(dp.tflops_est) < min_tflops:
            return False
        return True
    except Exception:
        return False


def propose_reallocated_plan(
    current_plan: PartitionPlan,
    registry,
    reputation_manager,
    plan_registry: PartitionPlanRegistry,
    heartbeat_timeout_s: float = 60.0,
    min_vram_gb: float = 8.0,
    min_tflops: float = 1.0,
) -> Optional[PartitionPlan]:
    """Return a new plan with reassigned stages if any device is stale/unhealthy; otherwise None.

    - Picks replacement nodes that are not stale, have device profiles, meet constraints.
    - Prefers higher TFLOPS among candidates.
    - Overwrites the existing (epoch, round) entry in the registry with the new plan.
    """
    now = time.time()
    replacements: List[StagePlan] = []
    modified = False

    # Build lookup of available nodes
    available_nodes = [n for n in registry.nodes.values() if not _is_node_stale(n, now, heartbeat_timeout_s)]

    for sp in current_plan.stage_plans:
        node = registry.nodes.get(sp.device_id)
        if node is None:
            logger.warning("Stage device %s not found in registry", sp.device_id)
            modified = True
            replacement = _select_best_replacement(available_nodes, min_vram_gb, min_tflops, reputation_manager)
            if not replacement:
                return None
            replacements.append(StagePlan(device_id=replacement.node_id, start_layer=sp.start_layer, end_layer=sp.end_layer))
            continue

        node_rep = reputation_manager.get_node_reputation(sp.device_id)
        if _is_node_stale(node, now, heartbeat_timeout_s) or node_rep < 50.0:
            modified = True
            replacement = _select_best_replacement(
                [n for n in available_nodes if n.node_id != sp.device_id], min_vram_gb, min_tflops, reputation_manager
            )
            if not replacement:
                return None
            replacements.append(StagePlan(device_id=replacement.node_id, start_layer=sp.start_layer, end_layer=sp.end_layer))
        else:
            replacements.append(sp)

    if not modified:
        return None

    new_plan = PartitionPlan(
        plan_id=f"plan_{int(time.time())}_realloc",
        epoch_id=current_plan.epoch_id,
        round_id=current_plan.round_id,
        created_at=time.time(),
        stage_plans=replacements,
        metadata=dict(current_plan.metadata),
    )

    # Save and update index for the (epoch,round)
    plan_registry.save_plan(new_plan)
    logger.info("Reallocated plan saved: %s (round %s)", new_plan.plan_id, new_plan.round_id)
    return new_plan


def _select_best_replacement(nodes: List, min_vram_gb: float, min_tflops: float, reputation_manager) -> Optional:
    candidates = [n for n in nodes if _meets_device_constraints(n, min_vram_gb, min_tflops)]
    if not candidates:
        return None
    # Prefer higher TFLOPS, then reputation
    def score(n):
        try:
            dp = n.device_profile
            tflops = float(dp.tflops_est or 0.0)
        except Exception:
            tflops = 0.0
        rep = reputation_manager.get_node_reputation(n.node_id) if hasattr(reputation_manager, 'get_node_reputation') else 50.0
        return (tflops, rep)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def propose_single_node_fallback_plan(
    current_plan: PartitionPlan,
    registry,
    reputation_manager,
    plan_registry: PartitionPlanRegistry,
    heartbeat_timeout_s: float = 60.0,
    min_vram_gb: float = 8.0,
    min_tflops: float = 1.0,
) -> Optional[PartitionPlan]:
    """Collapse all stages onto the best single healthy node as a last-resort fallback."""
    now = time.time()
    available_nodes = [n for n in registry.nodes.values() if not _is_node_stale(n, now, heartbeat_timeout_s)]
    best = _select_best_replacement(available_nodes, min_vram_gb, min_tflops, reputation_manager)
    if not best:
        return None
    start_layer = min(sp.start_layer for sp in current_plan.stage_plans)
    end_layer = max(sp.end_layer for sp in current_plan.stage_plans)
    new_plan = PartitionPlan(
        plan_id=f"plan_{int(time.time())}_single",
        epoch_id=current_plan.epoch_id,
        round_id=current_plan.round_id,
        created_at=time.time(),
        stage_plans=[StagePlan(device_id=best.node_id, start_layer=start_layer, end_layer=end_layer)],
        metadata=dict(current_plan.metadata),
    )
    plan_registry.save_plan(new_plan)
    logger.info("Single-node fallback plan saved: %s (round %s)", new_plan.plan_id, new_plan.round_id)
    return new_plan

