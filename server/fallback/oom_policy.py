"""OOM fallback policy with deterministic sequence.

Integrates with batcher and KV manager for memory recovery.
"""

import asyncio
import torch
import gc
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import structlog

logger = structlog.get_logger()


class FallbackStep(Enum):
    """OOM fallback steps in priority order."""
    SPLIT_BATCH = "split_batch"
    SHRINK_KV = "shrink_kv" 
    LOWER_PRECISION = "lower_precision"
    REJECT_REQUEST = "reject_request"


@dataclass
class FallbackContext:
    """Context for fallback execution."""
    batch_size: int
    kv_cache_entries: int
    precision: str  # 'fp32', 'fp16', 'bf16', 'int8'
    memory_available: int
    memory_required: int
    attempt: int = 0
    max_attempts: int = 4
    error: Optional[Exception] = None


class OOMFallbackPolicy:
    """Manages OOM fallback sequence with pluggable steps."""
    
    def __init__(
        self,
        batcher=None,
        kv_manager=None,
        model_manager=None
    ):
        self.batcher = batcher
        self.kv_manager = kv_manager
        self.model_manager = model_manager
        
        # Fallback handlers
        self.handlers: Dict[FallbackStep, Callable] = {
            FallbackStep.SPLIT_BATCH: self._split_batch,
            FallbackStep.SHRINK_KV: self._shrink_kv,
            FallbackStep.LOWER_PRECISION: self._lower_precision,
            FallbackStep.REJECT_REQUEST: self._reject_request,
        }
        
        # Default sequence
        self.sequence = [
            FallbackStep.SPLIT_BATCH,
            FallbackStep.SHRINK_KV,
            FallbackStep.LOWER_PRECISION,
            FallbackStep.REJECT_REQUEST,
        ]
        
        # Metrics
        self.fallback_counts = {step: 0 for step in FallbackStep}
        self.total_oom_events = 0
        self.successful_recoveries = 0
    
    async def handle_oom(self, context: FallbackContext) -> bool:
        """Execute fallback sequence until success or rejection.
        
        Returns:
            True if OOM resolved, False if all fallbacks exhausted
        """
        self.total_oom_events += 1
        
        logger.warning(
            "OOM detected, starting fallback sequence",
            memory_required=context.memory_required,
            memory_available=context.memory_available,
            batch_size=context.batch_size
        )
        
        for step in self.sequence:
            if context.attempt >= context.max_attempts:
                logger.error("Max fallback attempts reached")
                await self._reject_request(context)
                return False
            
            logger.info(f"Attempting OOM fallback: {step.value}")
            
            try:
                handler = self.handlers[step]
                success = await handler(context)
                
                self.fallback_counts[step] += 1
                context.attempt += 1
                
                if success:
                    # Try to allocate memory again
                    if self._check_memory_available(context):
                        logger.info(f"OOM resolved with {step.value}")
                        self.successful_recoveries += 1
                        return True
                    
                    # Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Check again after GC
                    if self._check_memory_available(context):
                        logger.info(f"OOM resolved with {step.value} after GC")
                        self.successful_recoveries += 1
                        return True
                        
            except Exception as e:
                logger.error(f"Fallback step {step.value} failed: {e}")
                context.error = e
        
        # All fallbacks exhausted
        logger.error("All OOM fallbacks exhausted")
        return False
    
    async def _split_batch(self, context: FallbackContext) -> bool:
        """Split batch to reduce memory usage."""
        if not self.batcher or context.batch_size <= 1:
            logger.debug("Cannot split batch further")
            return False
        
        try:
            # Get current batch info
            metrics = self.batcher.get_metrics()
            active_requests = metrics.get("active_requests", 0)
            
            if active_requests <= 1:
                return False
            
            # Evict half of the batch
            to_evict = max(1, active_requests // 2)
            evicted = await self.batcher.evict(to_evict)
            
            if evicted:
                logger.info(f"Split batch: evicted {len(evicted)} requests")
                context.batch_size = active_requests - len(evicted)
                
                # Free KV cache for evicted requests
                if self.kv_manager:
                    for req_id in evicted:
                        # Clean up KV cache entries
                        self.kv_manager.kv_pool.free_cache(req_id)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Batch split failed: {e}")
            return False
    
    async def _shrink_kv(self, context: FallbackContext) -> bool:
        """Shrink KV cache to free memory."""
        if not self.kv_manager:
            logger.debug("No KV manager available")
            return False
        
        try:
            stats = self.kv_manager.stats()
            entries = stats.get("unique_keys", 0)
            
            if entries == 0:
                logger.debug("No KV cache entries to evict")
                return False
            
            # Evict 25% of cache using largest-first policy
            to_evict = max(1, entries // 4)
            evicted_count = 0
            
            for _ in range(to_evict):
                if self.kv_manager.evict("largest"):
                    evicted_count += 1
                else:
                    break
            
            if evicted_count > 0:
                new_stats = self.kv_manager.stats()
                memory_freed = stats.get("memory_usage_gb", 0) - new_stats.get("memory_usage_gb", 0)
                
                logger.info(
                    f"Shrunk KV cache: evicted {evicted_count} entries, "
                    f"freed {memory_freed:.2f}GB"
                )
                context.kv_cache_entries = new_stats.get("unique_keys", 0)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"KV shrink failed: {e}")
            return False
    
    async def _lower_precision(self, context: FallbackContext) -> bool:
        """Lower model precision to reduce memory usage."""
        precision_sequence = ['fp32', 'bf16', 'fp16', 'int8']
        
        try:
            current_idx = precision_sequence.index(context.precision)
            if current_idx >= len(precision_sequence) - 1:
                logger.debug(f"Already at lowest precision: {context.precision}")
                return False
            
            new_precision = precision_sequence[current_idx + 1]
            
            # Calculate memory savings
            memory_factor = {
                'fp32': 1.0,
                'bf16': 0.5,
                'fp16': 0.5,
                'int8': 0.25
            }
            
            old_factor = memory_factor[context.precision]
            new_factor = memory_factor[new_precision]
            memory_reduction = 1 - (new_factor / old_factor)
            
            logger.info(
                f"Lowering precision: {context.precision} -> {new_precision} "
                f"(~{memory_reduction*100:.0f}% memory reduction)"
            )
            
            # In production, would trigger model reload
            if self.model_manager and hasattr(self.model_manager, 'set_precision'):
                await self.model_manager.set_precision(new_precision)
            
            context.precision = new_precision
            context.memory_required = int(context.memory_required * new_factor / old_factor)
            
            return True
            
        except Exception as e:
            logger.error(f"Precision lowering failed: {e}")
            return False
    
    async def _reject_request(self, context: FallbackContext) -> bool:
        """Final fallback: reject request with 507 Insufficient Storage."""
        logger.error(
            "Rejecting request due to OOM",
            memory_required=context.memory_required,
            memory_available=context.memory_available,
            attempts=context.attempt
        )
        
        # In production, would raise HTTPException(507)
        error_msg = (
            f"Insufficient memory: need {context.memory_required/1e9:.2f}GB, "
            f"have {context.memory_available/1e9:.2f}GB available"
        )
        
        raise MemoryError(error_msg)
    
    def _check_memory_available(self, context: FallbackContext) -> bool:
        """Check if enough memory is available."""
        if torch.cuda.is_available():
            # Get actual GPU memory
            mem_free, mem_total = torch.cuda.mem_get_info()
            context.memory_available = mem_free
            
            # Add some buffer (10%)
            required_with_buffer = context.memory_required * 1.1
            return mem_free >= required_with_buffer
        else:
            # For CPU, use psutil
            try:
                import psutil
                mem = psutil.virtual_memory()
                context.memory_available = mem.available
                return mem.available >= context.memory_required
            except:
                # Fallback assumption
                return context.memory_available >= context.memory_required
    
    def get_metrics(self) -> Dict:
        """Get fallback metrics."""
        return {
            "total_oom_events": self.total_oom_events,
            "successful_recoveries": self.successful_recoveries,
            "recovery_rate": self.successful_recoveries / max(self.total_oom_events, 1),
            "fallback_counts": dict(self.fallback_counts),
            "reject_count": self.fallback_counts[FallbackStep.REJECT_REQUEST],
        }
    
    def register_handler(self, step: FallbackStep, handler: Callable):
        """Register custom fallback handler."""
        self.handlers[step] = handler
        logger.info(f"Registered custom handler for {step.value}")
    
    def set_sequence(self, sequence: List[FallbackStep]):
        """Set custom fallback sequence."""
        self.sequence = sequence
        logger.info(f"Updated fallback sequence: {[s.value for s in sequence]}")