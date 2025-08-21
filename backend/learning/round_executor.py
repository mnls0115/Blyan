#!/usr/bin/env python3
"""
Round Executor

Orchestrates applying a frozen partition plan to a training round.
Integrates with RoundPlanManager for plan selection and failure fallback.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, List

from .round_plan_manager import RoundPlanManager
from .partition_plan_registry import PartitionPlan
from .pipeline_parallel import PipelineParallelTrainer, StageConfig
from .dataset_loader import build_training_dataloader
from backend.core.dataset_chain import DatasetChain
from backend.model.arch import ModelWrapper

logger = logging.getLogger(__name__)


class RoundExecutor:
    def __init__(self, plan_manager: RoundPlanManager):
        self.plan_manager = plan_manager

    async def execute_round(
        self,
        rpc_peers: List[str],
        rpc_timeout_s: float,
        transport: str = "http",
    ) -> bool:
        """Execute one training round using the current partition plan.
        Returns True on success, False on failure (and may switch to fallback plan).
        """
        plan: Optional[PartitionPlan] = self.plan_manager.get_current_plan()
        if not plan:
            logger.error("No partition plan available for round execution")
            return False

        # Build stage configs from plan
        stages = [
            StageConfig(device_id=sp.device_id, start_layer=sp.start_layer, end_layer=sp.end_layer)
            for sp in plan.stage_plans
        ]

        # Initialize pipeline trainer
        trainer = PipelineParallelTrainer(
            stages=stages,
            rpc_peers=rpc_peers,
            rpc_timeout_s=rpc_timeout_s,
            use_zero1=bool(plan.metadata.get("use_zero1", 0.0)),
            use_activation_checkpointing=bool(plan.metadata.get("use_activation_checkpointing", 0.0)),
            rpc_transport=transport,
        )

        try:
            # For now, perform a minimal pipeline warmup cycle to validate connectivity
            microbatches = [{"id": f"mb_{i}"} for i in range(2)]
            await trainer.run_1f1b(microbatches=microbatches, timeout_s=rpc_timeout_s)
            # Execute a minimal real training pass if enabled via environment (no mocks)
            if os.getenv('TRAINING_ENABLE', '0').lower() in ('1', 'true', 'yes'):
                try:
                    from pathlib import Path
                    import torch
                    # Build dataset loader from GOLD tier
                    ds_chain = DatasetChain(Path('./data'), 'D')
                    # Load model via ModelWrapper (must exist in the environment)
                    model_name = os.getenv('TRAINING_MODEL_NAME', 'Qwen/Qwen1.5-MoE-A2.7B')
                    wrapper = ModelWrapper(model_name, allow_mock_fallback=False)

                    # Optional DDP and ZeRO-1
                    use_ddp = os.getenv('USE_DDP', '0').lower() in ('1', 'true', 'yes')
                    use_zero1_flag = (os.getenv('USE_ZERO1', '0').lower() in ('1', 'true', 'yes')) or bool(plan.metadata.get('use_zero1', 0.0))
                    if use_ddp:
                        try:
                            from backend.learning.ddp_utils import init_distributed_if_needed, wrap_model_ddp
                            init_distributed_if_needed()
                            wrapper.model = wrap_model_ddp(wrapper.model)
                        except Exception as e:
                            logger.warning(f"DDP init failed; proceeding without: {e}")

                    # Optimizer
                    from torch.optim import AdamW
                    optimizer = AdamW(wrapper.model.parameters(), lr=float(os.getenv('TRAINING_LR', '1e-4')))
                    if use_zero1_flag:
                        try:
                            from backend.learning.zero1 import init_distributed_from_env, wrap_optimizer_with_zero1
                            init_distributed_from_env()
                            optimizer = wrap_optimizer_with_zero1(optimizer, wrapper.model.parameters())
                        except Exception as e:
                            logger.warning(f"ZeRO-1 enable failed; proceeding without: {e}")

                    # DataLoader
                    dataloader = build_training_dataloader(
                        dataset_chain=ds_chain,
                        tokenizer=wrapper.tokenizer,
                        batch_size=int(os.getenv('TRAINING_BATCH_SIZE', '1')),
                        seq_len=int(os.getenv('TRAINING_SEQ_LEN', '512')),
                        max_uris=int(os.getenv('TRAINING_MAX_URIS', '1'))
                    )

                    # Minimal training steps to validate end-to-end path
                    max_steps = int(os.getenv('TRAINING_MAX_STEPS', '1'))
                    grad_accum = int(os.getenv('TRAINING_GRAD_ACCUM', '4'))
                    from .micro_step_trainer import MicroStepTrainer, MicroStepConfig
                    trainer_cfg = MicroStepConfig(gradient_accumulation_steps=grad_accum)
                    ms_trainer = MicroStepTrainer(
                        model=wrapper.model,
                        optimizer=optimizer,
                        config=trainer_cfg,
                        use_activation_checkpointing=bool(plan.metadata.get('use_activation_checkpointing', 0.0))
                    )

                    device = None
                    try:
                        device = next(wrapper.model.parameters()).device
                    except Exception:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    data_iter = iter(dataloader)
                    for step in range(max_steps):
                        batch = next(data_iter)
                        # Move batch to device
                        for k, v in list(batch.items()):
                            if isinstance(v, torch.Tensor):
                                batch[k] = v.to(device)
                        await ms_trainer.train_step(batch, step)
                except StopIteration:
                    logger.info("Training dataloader exhausted before reaching max steps")
                except Exception as e:
                    logger.error(f"Round training step failed: {e}")
            await self.plan_manager.mark_success()
            return True
        except Exception as e:
            logger.error(f"Round execution failed: {e}")
            await self.plan_manager.mark_failure()
            return False

