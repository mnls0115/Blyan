#!/usr/bin/env python3
"""Micro-step training system with yield control for concurrent inference."""

import asyncio
import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class TrainingState(Enum):
    """Training states for micro-step control."""
    IDLE = "idle"
    TRAINING = "training"
    PAUSED = "paused"
    YIELDING = "yielding"
    CHECKPOINTING = "checkpointing"

@dataclass
class MicroStepConfig:
    """Configuration for micro-step training."""
    min_step_duration_ms: int = 50          # Minimum step duration
    max_step_duration_ms: int = 200         # Maximum step duration
    yield_check_interval_ms: int = 10       # How often to check for yield
    checkpoint_interval_steps: int = 100    # Checkpoint every N steps
    gradient_accumulation_steps: int = 4    # Accumulate gradients
    max_memory_buffer_mb: int = 1024       # Max memory for gradient buffer
    adaptive_step_sizing: bool = True       # Adjust step size based on load
    preemption_grace_ms: int = 10          # Max time to wait for graceful pause

@dataclass
class TrainingMetrics:
    """Metrics for training progress and performance."""
    total_steps: int = 0
    total_tokens: int = 0
    average_loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    step_duration_ms: float = 0.0
    yield_count: int = 0
    checkpoint_count: int = 0
    memory_used_mb: float = 0.0
    
class MicroCheckpoint:
    """Lightweight checkpoint for fast save/restore."""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.step = 0
        self.best_loss = float('inf')
        self.checkpoint_buffer = {}
        
    def save_micro(self, step: int, loss: float) -> Dict[str, Any]:
        """Save micro checkpoint to memory buffer."""
        checkpoint = {
            'step': step,
            'loss': loss,
            'model_state': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'optimizer_state': self.optimizer.state_dict(),
            'timestamp': time.time()
        }
        
        # Keep only recent checkpoints
        self.checkpoint_buffer[step] = checkpoint
        if len(self.checkpoint_buffer) > 5:
            oldest_step = min(self.checkpoint_buffer.keys())
            del self.checkpoint_buffer[oldest_step]
            
        return checkpoint
        
    async def save_persistent(self, path: Path, checkpoint: Dict[str, Any]):
        """Save checkpoint to disk asynchronously."""
        await asyncio.to_thread(torch.save, checkpoint, path)
        logger.info(f"Saved checkpoint at step {checkpoint['step']} to {path}")
        
    def restore_micro(self, step: int) -> bool:
        """Restore from micro checkpoint."""
        if step not in self.checkpoint_buffer:
            return False
            
        checkpoint = self.checkpoint_buffer[step]
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.step = checkpoint['step']
        
        logger.info(f"Restored checkpoint from step {step}")
        return True

class MicroStepTrainer:
    """Trainer with micro-stepping for concurrent inference support."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: MicroStepConfig = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or MicroStepConfig()
        
        # State management
        self.state = TrainingState.IDLE
        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Start unpaused
        self.stop_event = asyncio.Event()
        
        # Metrics
        self.metrics = TrainingMetrics()
        self.step_history: List[float] = []
        
        # Checkpointing
        self.checkpoint_manager = MicroCheckpoint(model, optimizer)
        
        # Yield control
        self.yield_requested = False
        self.last_yield_time = time.time()
        self.inference_pressure = 0.0  # 0-1 scale
        
        # Gradient accumulation
        self.accumulated_gradients = []
        self.accumulation_step = 0
        
    def request_yield(self, priority: float = 0.5):
        """Request training to yield for inference."""
        self.yield_requested = True
        self.inference_pressure = max(self.inference_pressure, priority)
        logger.debug(f"Yield requested with priority {priority}")
        
    async def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, Any]:
        """Execute a single micro training step with yield points."""
        
        self.state = TrainingState.TRAINING
        step_start = time.time()
        
        try:
            # Wait for resume if paused
            await self._wait_for_resume()
            
            # Forward pass with yield points
            self.model.train()
            
            # Micro-yield point 1: Before forward
            await self._micro_yield()
            
            outputs = self.model(**batch)
            loss = outputs.loss / self.config.gradient_accumulation_steps
            
            # Micro-yield point 2: After forward
            await self._micro_yield()
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            self.accumulation_step += 1
            
            if self.accumulation_step >= self.config.gradient_accumulation_steps:
                # Micro-yield point 3: Before optimizer step
                await self._micro_yield()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=1.0
                )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accumulation_step = 0
                
                # Update metrics
                self.metrics.gradient_norm = grad_norm.item()
                
            # Update metrics
            self.metrics.total_steps += 1
            self.metrics.total_tokens += batch['input_ids'].numel()
            self.metrics.average_loss = (
                0.9 * self.metrics.average_loss + 0.1 * loss.item()
            )
            self.metrics.learning_rate = self.optimizer.param_groups[0]['lr']
            
            step_duration = (time.time() - step_start) * 1000
            self.metrics.step_duration_ms = step_duration
            self.step_history.append(step_duration)
            
            # Adaptive step sizing
            if self.config.adaptive_step_sizing:
                await self._adaptive_delay(step_duration)
                
            # Checkpointing
            if step % self.config.checkpoint_interval_steps == 0:
                await self._checkpoint(step)
                
            return {
                'loss': loss.item() * self.config.gradient_accumulation_steps,
                'grad_norm': self.metrics.gradient_norm,
                'learning_rate': self.metrics.learning_rate,
                'step_duration_ms': step_duration,
                'tokens_processed': batch['input_ids'].numel()
            }
            
        except asyncio.CancelledError:
            logger.info("Training step cancelled")
            raise
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            raise
        finally:
            self.state = TrainingState.IDLE
            
    async def _wait_for_resume(self):
        """Wait for training to resume if paused."""
        if not self.pause_event.is_set():
            self.state = TrainingState.PAUSED
            logger.info("Training paused, waiting for resume...")
            await self.pause_event.wait()
            logger.info("Training resumed")
            
    async def _micro_yield(self):
        """Micro yield point for inference priority."""
        # Check if we should yield
        current_time = time.time()
        time_since_yield = (current_time - self.last_yield_time) * 1000
        
        should_yield = (
            self.yield_requested or
            time_since_yield > self.config.max_step_duration_ms or
            self.inference_pressure > 0.7
        )
        
        if should_yield:
            self.state = TrainingState.YIELDING
            self.metrics.yield_count += 1
            
            # Calculate yield duration based on pressure
            yield_duration = min(
                self.config.preemption_grace_ms * self.inference_pressure,
                50  # Cap at 50ms
            ) / 1000  # Convert to seconds
            
            await asyncio.sleep(yield_duration)
            
            self.last_yield_time = current_time
            self.yield_requested = False
            self.inference_pressure *= 0.9  # Decay pressure
            
    async def _adaptive_delay(self, last_step_duration: float):
        """Adaptive delay based on system load."""
        # Target step duration based on inference pressure
        target_duration = (
            self.config.min_step_duration_ms +
            (self.config.max_step_duration_ms - self.config.min_step_duration_ms) *
            (1 - self.inference_pressure)
        )
        
        # Add delay if step was too fast
        if last_step_duration < target_duration:
            delay_ms = target_duration - last_step_duration
            await asyncio.sleep(delay_ms / 1000)
            
    async def _checkpoint(self, step: int):
        """Save checkpoint with minimal interruption."""
        self.state = TrainingState.CHECKPOINTING
        checkpoint_start = time.time()
        
        try:
            # Save to memory first (fast)
            checkpoint = self.checkpoint_manager.save_micro(
                step, 
                self.metrics.average_loss
            )
            
            # Async save to disk (non-blocking)
            checkpoint_path = Path(f"checkpoints/micro_step_{step}.pt")
            checkpoint_path.parent.mkdir(exist_ok=True)
            
            asyncio.create_task(
                self.checkpoint_manager.save_persistent(checkpoint_path, checkpoint)
            )
            
            self.metrics.checkpoint_count += 1
            
            logger.info(
                f"Checkpoint {step} saved in "
                f"{(time.time() - checkpoint_start) * 1000:.1f}ms"
            )
            
        except Exception as e:
            logger.error(f"Checkpoint failed: {e}")
            
    def pause(self):
        """Pause training immediately."""
        self.pause_event.clear()
        self.request_yield(priority=1.0)
        
    def resume(self):
        """Resume training."""
        self.pause_event.set()
        self.inference_pressure = 0.0
        
    def stop(self):
        """Stop training gracefully."""
        self.stop_event.set()
        self.pause_event.set()  # Ensure not stuck in pause
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return {
            'state': self.state.value,
            'total_steps': self.metrics.total_steps,
            'total_tokens': self.metrics.total_tokens,
            'average_loss': self.metrics.average_loss,
            'learning_rate': self.metrics.learning_rate,
            'gradient_norm': self.metrics.gradient_norm,
            'average_step_duration_ms': np.mean(self.step_history[-100:]) if self.step_history else 0,
            'yield_count': self.metrics.yield_count,
            'checkpoint_count': self.metrics.checkpoint_count,
            'inference_pressure': self.inference_pressure,
            'memory_used_mb': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        }

class DataLoaderWrapper:
    """Async wrapper for PyTorch DataLoader with micro-batching."""
    
    def __init__(self, dataloader, micro_batch_size: int = 1):
        self.dataloader = dataloader
        self.micro_batch_size = micro_batch_size
        self.current_batch = None
        self.micro_batch_idx = 0
        
    async def get_micro_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get next micro batch."""
        if self.current_batch is None or self.micro_batch_idx >= len(self.current_batch['input_ids']):
            # Load next full batch
            try:
                self.current_batch = await asyncio.to_thread(next, iter(self.dataloader))
                self.micro_batch_idx = 0
            except StopIteration:
                return None
                
        # Extract micro batch
        start_idx = self.micro_batch_idx
        end_idx = min(
            self.micro_batch_idx + self.micro_batch_size,
            len(self.current_batch['input_ids'])
        )
        
        micro_batch = {
            key: value[start_idx:end_idx]
            for key, value in self.current_batch.items()
        }
        
        self.micro_batch_idx = end_idx
        return micro_batch

# Integration with concurrent inference
async def train_with_micro_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader,
    num_epochs: int = 1,
    inference_coordinator=None
):
    """Train model with micro-stepping and inference coordination."""
    
    config = MicroStepConfig(
        min_step_duration_ms=50,
        max_step_duration_ms=200,
        adaptive_step_sizing=True
    )
    
    trainer = MicroStepTrainer(model, optimizer, config)
    dataloader_wrapper = DataLoaderWrapper(train_dataloader, micro_batch_size=2)
    
    # Register with inference coordinator if available
    if inference_coordinator:
        inference_coordinator.register_trainer(trainer)
    
    logger.info("Starting micro-step training...")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        step = 0
        
        while True:
            # Get micro batch
            micro_batch = await dataloader_wrapper.get_micro_batch()
            if micro_batch is None:
                break
                
            # Check for stop signal
            if trainer.stop_event.is_set():
                logger.info("Training stopped by user")
                break
                
            # Train step
            try:
                step_result = await trainer.train_step(micro_batch, step)
                
                if step % 10 == 0:
                    metrics = trainer.get_metrics()
                    logger.info(
                        f"Epoch {epoch}, Step {step}: "
                        f"loss={step_result['loss']:.4f}, "
                        f"lr={metrics['learning_rate']:.6f}, "
                        f"duration={step_result['step_duration_ms']:.1f}ms, "
                        f"pressure={metrics['inference_pressure']:.2f}"
                    )
                    
            except Exception as e:
                logger.error(f"Training step failed: {e}")
                continue
                
            step += 1
            
        epoch_duration = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch} completed in {epoch_duration:.1f}s, "
            f"steps={step}, "
            f"avg_loss={trainer.metrics.average_loss:.4f}"
        )
        
    final_metrics = trainer.get_metrics()
    logger.info(f"Training completed: {final_metrics}")
    
    return trainer

# Production code - demo removed