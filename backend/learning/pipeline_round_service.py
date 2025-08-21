#!/usr/bin/env python3
"""
Pipeline Round Service

Runs training rounds using frozen partition plans with failure fallback and
basic stage recovery. Integrates with RoundPlanManager and RoundExecutor.
"""

from __future__ import annotations

import asyncio
import os
from typing import List, Optional, Dict
import logging

import aiohttp

from .partition_plan_registry import PartitionPlanRegistry
from .round_plan_manager import RoundPlanManager
from .round_executor import RoundExecutor
from .pipeline_metrics import get_pipeline_metrics
from .stage_reallocator import propose_reallocated_plan, propose_single_node_fallback_plan
from .dataset_loader import build_training_dataloader
from .micro_step_trainer import MicroStepTrainer, MicroStepConfig
from backend.core.dataset_chain import DatasetChain
from backend.model.arch import ModelWrapper
from torch.optim import AdamW

logger = logging.getLogger(__name__)


class PipelineRoundService:
    def __init__(self, registry, epoch_scheduler=None, transport: str = "http", plan_registry: Optional[PartitionPlanRegistry] = None):
        """
        registry: ExpertNodeRegistry (from DistributedInferenceCoordinator)
        epoch_scheduler: EpochEventScheduler (optional; used to derive current epoch)
        transport: "http" or "grpc"
        """
        self.registry = registry
        self.epoch_scheduler = epoch_scheduler
        # Allow env override for transport (http|grpc)
        self.transport = os.getenv('BLYAN_PIPELINE_TRANSPORT', transport)
        self.plan_registry = plan_registry or PartitionPlanRegistry()
        self.plan_manager = RoundPlanManager(self.plan_registry)
        self.executor = RoundExecutor(self.plan_manager)
        self.metrics = get_pipeline_metrics()
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        # Training stack (lazy init)
        self._training_enabled = os.getenv('TRAINING_ENABLE', '0').lower() in ('1', 'true', 'yes')
        self._model_wrapper: Optional[ModelWrapper] = None
        self._optimizer = None
        self._ms_trainer: Optional[MicroStepTrainer] = None
        self._dataloader = None
        self._dataloader_iter = None

    def start(self):
        if self._task and not self._task.done():
            return
        self._stop.clear()
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._run_loop())
        logger.info("PipelineRoundService started")

    async def stop(self):
        if self._task:
            self._stop.set()
            await self._task
            self._task = None
            logger.info("PipelineRoundService stopped")

    def _resolve_epoch_id(self) -> Optional[str]:
        # Prefer current scheduler epoch
        try:
            if self.epoch_scheduler and self.epoch_scheduler.current_epoch:
                return self.epoch_scheduler.current_epoch.event_id
        except Exception:
            pass
        # Fallback to latest stable plan's epoch
        latest = self.plan_registry.latest_stable()
        return latest.epoch_id if latest else None

    def _build_rpc_peers(self, stage_device_ids: List[str]) -> List[str]:
        peers: List[str] = []
        for nid in stage_device_ids:
            node = self.registry.nodes.get(nid)
            if not node:
                raise RuntimeError(f"Unknown node_id {nid} in stage assignment")
            scheme = "https" if os.getenv("BLYAN_TLS_CERT") else "http"
            peers.append(f"{scheme}://{node.host}:{node.port}")
        return peers

    async def _reset_pipeline_nodes(self, peers: List[str]):
        try:
            async with aiohttp.ClientSession() as session:
                for base in peers:
                    try:
                        await session.post(f"{base}/pipeline/reset", timeout=aiohttp.ClientTimeout(total=3))
                    except Exception as e:
                        logger.warning(f"Pipeline reset failed for {base}: {e}")
        except Exception as e:
            logger.warning(f"Pipeline reset session failed: {e}")

    async def _run_loop(self):
        # Load epoch rounds
        epoch_id = self._resolve_epoch_id()
        if not epoch_id:
            logger.warning("No epoch id found; pipeline round service idle")
            return
        self.plan_manager.load_epoch(epoch_id)

        # Continuous loop across rounds
        while not self._stop.is_set():
            plan = self.plan_manager.get_current_plan()
            if not plan:
                await asyncio.sleep(5)
                continue

            # Compute partition drift and device profile staleness alerts
            await self._update_drift_and_alerts(plan)

            # Build peers based on plan stage order
            stage_device_ids = [sp.device_id for sp in plan.stage_plans]
            try:
                await self.metrics.set_current_stage_count(len(stage_device_ids))
            except Exception:
                pass
            peers = self._build_rpc_peers(stage_device_ids)

            # Reset pipeline buffers before each round
            await self._reset_pipeline_nodes(peers)

            # Execute round (pipeline warmup/connectivity)
            ok = await self.executor.execute_round(rpc_peers=peers, rpc_timeout_s=float(os.getenv('BLYAN_PIPELINE_TIMEOUT_S', '5.0')), transport=self.transport)
            if not ok:
                # Alert and retry with fallback if provided by plan manager
                try:
                    from backend.security.monitoring import record_security_event
                    record_security_event("pipeline_rpc_errors", "pipeline_round_service", {"round": plan.round_id, "plan_id": plan.plan_id})
                except Exception:
                    pass
                # Attempt stage reallocation based on stale/health metrics
                try:
                    from backend.p2p.node_reputation import get_reputation_manager
                    new_plan = propose_reallocated_plan(
                        current_plan=plan,
                        registry=self.registry,
                        reputation_manager=get_reputation_manager(),
                        plan_registry=self.plan_registry,
                        heartbeat_timeout_s=float(os.getenv('BLYAN_HEARTBEAT_TIMEOUT', '60.0')),
                        min_vram_gb=float(os.getenv('BLYAN_MIN_VRAM_GB', '8.0')),
                        min_tflops=float(os.getenv('BLYAN_MIN_TFLOPS', '1.0')),
                    )
                    if new_plan:
                        await self.metrics.incr_fallback()
                        await self.metrics.set_fallback_active(True)
                        # Reload peers with new assignments and reset buffers
                        stage_device_ids = [sp.device_id for sp in new_plan.stage_plans]
                        peers = self._build_rpc_peers(stage_device_ids)
                        await self._reset_pipeline_nodes(peers)
                    else:
                        # Try single node fallback plan
                        fb_plan = propose_single_node_fallback_plan(
                            current_plan=plan,
                            registry=self.registry,
                            reputation_manager=get_reputation_manager(),
                            plan_registry=self.plan_registry,
                            heartbeat_timeout_s=float(os.getenv('BLYAN_HEARTBEAT_TIMEOUT', '60.0')),
                            min_vram_gb=float(os.getenv('BLYAN_MIN_VRAM_GB', '8.0')),
                            min_tflops=float(os.getenv('BLYAN_MIN_TFLOPS', '1.0')),
                        )
                        if fb_plan:
                            await self.metrics.incr_fallback()
                            await self.metrics.set_fallback_active(True)
                            stage_device_ids = [sp.device_id for sp in fb_plan.stage_plans]
                            peers = self._build_rpc_peers(stage_device_ids)
                            await self._reset_pipeline_nodes(peers)
                except Exception as e:
                    logger.warning(f"Stage reallocation failed: {e}")
                # Rebuild peers in case of reallocation/fallback
                fallback = self.plan_manager.get_current_plan()
                if fallback and fallback.plan_id != plan.plan_id:
                    stage_device_ids = [sp.device_id for sp in fallback.stage_plans]
                    peers = self._build_rpc_peers(stage_device_ids)
                    # Alarm if single-node fallback
                    try:
                        if len(stage_device_ids) == 1:
                            from backend.security.monitoring import record_security_event
                            record_security_event('throughput_degraded_single_node', 'pipeline_round_service', {'round': plan.round_id, 'plan_id': fallback.plan_id})
                    except Exception:
                        pass
            else:
                # Run actual training micro-steps for this round if enabled
                try:
                    await self._run_training_for_round(plan)
                except Exception as e:
                    logger.error(f"Training loop error in round: {e}")
                finally:
                    try:
                        await self.metrics.set_fallback_active(False)
                    except Exception:
                        pass
            # Short delay before next round
            await asyncio.sleep(float(os.getenv("BLYAN_PIPELINE_ROUND_INTERVAL", "2.0")))

    async def _update_drift_and_alerts(self, plan):
        """Update partition drift metric and record staleness/drift alerts if needed."""
        try:
            now = asyncio.get_event_loop().time()
        except Exception:
            import time as _time
            now = _time.time()

        heartbeat_timeout = float(os.getenv('BLYAN_HEARTBEAT_TIMEOUT', '60.0'))
        drift_bad = 0
        total = len(plan.stage_plans)

        # Check stage devices for staleness or low reputation
        try:
            from backend.p2p.node_reputation import get_reputation_manager
            rep_mgr = get_reputation_manager()
        except Exception:
            rep_mgr = None

        for sp in plan.stage_plans:
            node = self.registry.nodes.get(sp.device_id)
            if not node:
                drift_bad += 1
                continue
            last_hb = getattr(node, 'last_heartbeat', 0.0)
            stale = (now - float(last_hb)) > heartbeat_timeout
            low_rep = False
            if rep_mgr:
                try:
                    low_rep = rep_mgr.get_node_reputation(sp.device_id) < 50.0
                except Exception:
                    pass
            if stale or low_rep:
                drift_bad += 1

        drift = (drift_bad / total) if total > 0 else 0.0
        try:
            await self.metrics.set_partition_drift(drift)
        except Exception:
            pass

    async def _ensure_training_stack(self):
        if not self._training_enabled:
            return False
        if self._model_wrapper is not None and self._optimizer is not None and self._ms_trainer is not None and self._dataloader is not None:
            return True
        # Initialize model/tokenizer
        model_name = os.getenv('TRAINING_MODEL_NAME', 'Qwen/Qwen1.5-MoE-A2.7B')
        self._model_wrapper = ModelWrapper(model_name, allow_mock_fallback=False)
        # Optional DDP
        if os.getenv('USE_DDP', '0').lower() in ('1', 'true', 'yes'):
            try:
                from backend.learning.ddp_utils import init_distributed_if_needed, wrap_model_ddp
                init_distributed_if_needed()
                self._model_wrapper.model = wrap_model_ddp(self._model_wrapper.model)
            except Exception as e:
                logger.warning(f"DDP init failed; proceeding without: {e}")
        # Optimizer & ZeRO-1
        self._optimizer = AdamW(self._model_wrapper.model.parameters(), lr=float(os.getenv('TRAINING_LR', '1e-4')))
        if os.getenv('USE_ZERO1', '0').lower() in ('1', 'true', 'yes'):
            try:
                from backend.learning.zero1 import init_distributed_from_env, wrap_optimizer_with_zero1
                init_distributed_from_env()
                self._optimizer = wrap_optimizer_with_zero1(self._optimizer, self._model_wrapper.model.parameters())
            except Exception as e:
                logger.warning(f"ZeRO-1 enable failed; proceeding without: {e}")
        # DataLoader from GOLD tier
        from pathlib import Path
        ds_chain = DatasetChain(Path('./data'), 'D')
        self._dataloader = build_training_dataloader(
            dataset_chain=ds_chain,
            tokenizer=self._model_wrapper.tokenizer,
            batch_size=int(os.getenv('TRAINING_BATCH_SIZE', '1')),
            seq_len=int(os.getenv('TRAINING_SEQ_LEN', '512')),
            max_uris=int(os.getenv('TRAINING_MAX_URIS', '1'))
        )
        self._dataloader_iter = iter(self._dataloader)
        # Micro-step trainer
        from .round_plan_manager import RoundPlanManager  # avoid cycles
        grad_accum = int(os.getenv('TRAINING_GRAD_ACCUM', '4'))
        self._ms_trainer = MicroStepTrainer(self._model_wrapper.model, self._optimizer, MicroStepConfig(gradient_accumulation_steps=grad_accum))
        return True

    async def _run_training_for_round(self, plan):
        if not await self._ensure_training_stack():
            return
        import torch
        max_steps = int(os.getenv('TRAINING_MAX_STEPS', '1'))
        device = None
        try:
            device = next(self._model_wrapper.model.parameters()).device
        except Exception:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        steps_done = 0
        while steps_done < max_steps:
            try:
                batch = next(self._dataloader_iter)
            except StopIteration:
                # rebuild iterator
                self._dataloader_iter = iter(self._dataloader)
                batch = next(self._dataloader_iter)
            # Move to device
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            await self._ms_trainer.train_step(batch, steps_done)
            steps_done += 1

        # Emit alerts if thresholds crossed
        try:
            from backend.security.monitoring import record_security_event
            if drift >= float(os.getenv('BLYAN_PARTITION_DRIFT_ALERT', '0.5')):
                record_security_event('partition_drift_high', 'pipeline_round_service', {'drift': drift, 'round': plan.round_id})
        except Exception:
            pass

        # Device profile staleness alerts from metrics snapshot
        try:
            snap = await self.metrics.export_snapshot()
            staleness = snap.get('device_profile_staleness', {})
            stale_threshold = float(os.getenv('BLYAN_DEVICE_PROFILE_STALENESS_ALERT_S', '600'))
            from backend.security.monitoring import record_security_event
            for nid, age in staleness.items():
                if age >= stale_threshold:
                    record_security_event('device_profile_staleness', 'pipeline_round_service', {'node': nid, 'age': age})
        except Exception:
            pass

