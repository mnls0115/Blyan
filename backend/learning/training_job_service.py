#!/usr/bin/env python3
"""
Long-running Training Job Service

Runs background micro-step training loops with pause/resume/stop controls and
updates scheduler state (progress/estimated completion/last error).
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from backend.model.arch import ModelWrapper
from backend.core.dataset_chain import DatasetChain
from .dataset_loader import build_training_dataloader
from .micro_step_trainer import MicroStepTrainer, MicroStepConfig
from torch.optim import AdamW


@dataclass
class TrainingJobConfig:
    model_name: str
    batch_size: int
    seq_len: int
    max_uris: int
    lr: float
    grad_accum: int
    max_steps_per_epoch: int
    max_epochs: int


@dataclass
class TrainingJobState:
    running: bool = False
    paused: bool = False
    started_at: float = 0.0
    last_update: float = 0.0
    epochs_done: int = 0
    steps_done: int = 0
    total_tokens: int = 0
    average_loss: float = 0.0
    last_error: Optional[str] = None
    progress: float = 0.0  # 0..1 over planned epochs


class TrainingJobService:
    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._pause = asyncio.Event()
        self._pause.set()
        self._state = TrainingJobState()
        self._config: Optional[TrainingJobConfig] = None
        self._scheduler = None
        # Training stack
        self._wrapper: Optional[ModelWrapper] = None
        self._optimizer = None
        self._trainer: Optional[MicroStepTrainer] = None
        self._dataloader = None
        self._dataloader_iter = None

    def set_scheduler(self, scheduler):
        self._scheduler = scheduler

    def status(self) -> Dict[str, Any]:
        return asdict(self._state)

    def is_running(self) -> bool:
        return self._state.running

    def start(self, config: Optional[Dict[str, Any]] = None):
        if self._task and not self._task.done():
            return
        self._config = self._load_config(config)
        self._stop.clear()
        self._pause.set()
        self._task = asyncio.get_event_loop().create_task(self._run())

    async def stop(self):
        self._stop.set()
        if self._task:
            await self._task
        self._task = None
        self._state.running = False
        self._state.paused = False

    def pause(self):
        self._pause.clear()
        self._state.paused = True

    def resume(self):
        self._pause.set()
        self._state.paused = False

    async def reset(self):
        """Stop and clear training stack/state without starting a new job."""
        await self.stop()
        # Clear training stack and state
        self._wrapper = None
        self._optimizer = None
        self._trainer = None
        self._dataloader = None
        self._dataloader_iter = None
        self._state = TrainingJobState()

    def _load_config(self, override: Optional[Dict[str, Any]]) -> TrainingJobConfig:
        cfg = TrainingJobConfig(
            model_name=os.getenv('TRAINING_MODEL_NAME', 'Qwen/Qwen1.5-MoE-A2.7B'),
            batch_size=int(os.getenv('TRAINING_BATCH_SIZE', '1')),
            seq_len=int(os.getenv('TRAINING_SEQ_LEN', '512')),
            max_uris=int(os.getenv('TRAINING_MAX_URIS', '4')),
            lr=float(os.getenv('TRAINING_LR', '1e-4')),
            grad_accum=int(os.getenv('TRAINING_GRAD_ACCUM', '8')),
            max_steps_per_epoch=int(os.getenv('TRAINING_MAX_STEPS', '100')),
            max_epochs=int(os.getenv('TRAINING_MAX_EPOCHS', '1')),
        )
        if override:
            for k, v in override.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    async def _run(self):
        try:
            self._state = TrainingJobState(running=True, paused=False, started_at=time.time(), last_update=time.time())
            await self._ensure_training_stack()
            epochs_planned = max(1, self._config.max_epochs)
            for epoch_idx in range(self._config.max_epochs):
                if self._stop.is_set():
                    break
                await self._pause.wait()
                steps_in_epoch = 0
                while steps_in_epoch < self._config.max_steps_per_epoch:
                    if self._stop.is_set():
                        break
                    await self._pause.wait()
                    batch = self._next_batch()
                    device = next(self._wrapper.model.parameters()).device
                    tokens = 0
                    for k, v in list(batch.items()):
                        if isinstance(v, torch.Tensor):
                            tokens += int(v.numel()) if k == 'input_ids' else 0
                            batch[k] = v.to(device)
                    await self._trainer.train_step(batch, self._state.steps_done)
                    # Update state
                    self._state.steps_done += 1
                    steps_in_epoch += 1
                    self._state.total_tokens += tokens
                    self._state.average_loss = self._trainer.metrics.average_loss
                    self._state.last_update = time.time()
                    self._state.progress = min(1.0, (epoch_idx + steps_in_epoch / self._config.max_steps_per_epoch) / epochs_planned)
                    self._update_scheduler_progress()
                self._state.epochs_done += 1
            self._state.running = False
            self._update_scheduler_progress(final=True)
        except Exception as e:
            self._state.last_error = str(e)
            self._state.running = False
            self._update_scheduler_progress(error=str(e))

    def _next_batch(self) -> Dict[str, torch.Tensor]:
        try:
            return next(self._dataloader_iter)
        except StopIteration:
            self._dataloader_iter = iter(self._dataloader)
            return next(self._dataloader_iter)

    async def _ensure_training_stack(self):
        if self._wrapper is not None and self._trainer is not None and self._dataloader is not None:
            return
        # Model/tokenizer
        self._wrapper = ModelWrapper(self._config.model_name, allow_mock_fallback=False)
        # Optional DDP
        if os.getenv('USE_DDP', '0').lower() in ('1', 'true', 'yes'):
            try:
                from backend.learning.ddp_utils import init_distributed_if_needed, wrap_model_ddp
                init_distributed_if_needed()
                self._wrapper.model = wrap_model_ddp(self._wrapper.model)
            except Exception:
                pass
        # Optimizer & ZeRO-1
        self._optimizer = AdamW(self._wrapper.model.parameters(), lr=self._config.lr)
        if os.getenv('USE_ZERO1', '0').lower() in ('1', 'true', 'yes'):
            try:
                from backend.learning.zero1 import init_distributed_from_env, wrap_optimizer_with_zero1
                init_distributed_from_env()
                self._optimizer = wrap_optimizer_with_zero1(self._optimizer, self._wrapper.model.parameters())
            except Exception:
                pass
        # DataLoader
        ds_chain = DatasetChain(Path('./data'), 'D')
        self._dataloader = build_training_dataloader(
            dataset_chain=ds_chain,
            tokenizer=self._wrapper.tokenizer,
            batch_size=self._config.batch_size,
            seq_len=self._config.seq_len,
            max_uris=self._config.max_uris,
        )
        self._dataloader_iter = iter(self._dataloader)
        # Micro-step trainer
        self._trainer = MicroStepTrainer(self._wrapper.model, self._optimizer, MicroStepConfig(gradient_accumulation_steps=self._config.grad_accum))

    def _update_scheduler_progress(self, final: bool = False, error: Optional[str] = None):
        if not self._scheduler or not getattr(self._scheduler, 'current_epoch', None):
            return
        try:
            ce = self._scheduler.current_epoch
            ce.training_progress = self._state.progress
            ce.current_loss = float(self._state.average_loss)
            if final:
                ce.estimated_completion = time.time()
            if error:
                self._scheduler.current_epoch.error_log.append(error)
        except Exception:
            pass


_global_training_service: Optional[TrainingJobService] = None


def get_training_job_service() -> TrainingJobService:
    global _global_training_service
    if _global_training_service is None:
        _global_training_service = TrainingJobService()
    return _global_training_service

