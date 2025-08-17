"""Continuous batching facade combining BatchManager and ContinuousBatching.

Provides unified interface with prefill/decode mixing and token budget management.
"""

import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
import structlog

from backend.inference.batch_manager import BatchManager, InferenceRequest
from backend.optimization.kv_cache_manager import ContinuousBatching

logger = structlog.get_logger()


class RequestState(enum.Enum):
    """Request lifecycle states."""
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode" 
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    EVICTED = "evicted"
    TIMEOUT = "timeout"


@dataclass
class BatchRequest:
    """Enhanced request with state tracking."""
    request_id: str
    prompt_tokens: List[int]
    max_new_tokens: int
    temperature: float = 1.0
    state: RequestState = RequestState.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    generated_tokens: List[int] = field(default_factory=list)
    priority: float = 1.0
    timeout_seconds: float = 60.0
    
    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)
    
    @property
    def is_prefill(self) -> bool:
        return self.state == RequestState.PREFILL
    
    @property
    def is_decode(self) -> bool:
        return self.state == RequestState.DECODE
    
    @property 
    def is_active(self) -> bool:
        return self.state in (RequestState.PREFILL, RequestState.DECODE)


class Batcher:
    """Continuous batching with prefill/decode mixing and token budget management."""
    
    def __init__(
        self,
        max_batch_tokens: int = 2048,
        max_batch_size: int = 32,
        prefill_decode_ratio: float = 0.3,
        eviction_policy: str = "oldest_first",
        batch_manager: Optional[BatchManager] = None,
        continuous_batching: Optional[ContinuousBatching] = None
    ):
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self.prefill_decode_ratio = prefill_decode_ratio
        self.eviction_policy = eviction_policy
        
        # Underlying managers
        self.batch_manager = batch_manager or BatchManager()
        self.continuous_batching = continuous_batching or ContinuousBatching(
            max_batch_size=max_batch_size
        )
        
        # Request tracking
        self.requests: Dict[str, BatchRequest] = {}
        self.prefill_queue: List[BatchRequest] = []
        self.decode_batch: Dict[str, BatchRequest] = {}
        
        # Metrics
        self.total_processed = 0
        self.total_evicted = 0
        self.total_timeout = 0
        self.total_cancelled = 0
        
        # State
        self._lock = asyncio.Lock()
        self._shutdown = False
    
    async def enqueue(self, request: BatchRequest) -> str:
        """Add request to processing queue."""
        async with self._lock:
            # Store request
            self.requests[request.request_id] = request
            
            # Add to prefill queue
            self.prefill_queue.append(request)
            
            # Sort by priority and age
            self.prefill_queue.sort(
                key=lambda r: (-r.priority, r.created_at)
            )
            
            # Also add to underlying batch manager for caching
            inf_req = InferenceRequest(
                request_id=request.request_id,
                prompt=" ".join(str(t) for t in request.prompt_tokens[:10]),  # Mock prompt
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature
            )
            await self.batch_manager.add_request(inf_req)
            
            logger.debug(f"Enqueued request {request.request_id}")
            return request.request_id
    
    async def step(self) -> Tuple[List[BatchRequest], List[BatchRequest]]:
        """Execute one batching step with prefill/decode mixing.
        
        Returns:
            Tuple of (prefill_batch, decode_batch)
        """
        async with self._lock:
            # Check timeouts first
            await self._check_timeouts()
            
            # Calculate token budgets
            total_budget = self.max_batch_tokens
            prefill_budget = int(total_budget * self.prefill_decode_ratio)
            decode_budget = total_budget - prefill_budget
            
            # Select decode requests (ongoing generation)
            decode_batch = await self._select_decode_batch(decode_budget)
            
            # Select prefill requests (new)
            remaining_budget = total_budget - sum(r.total_tokens for r in decode_batch)
            prefill_batch = await self._select_prefill_batch(
                min(prefill_budget, remaining_budget)
            )
            
            # Update states
            for req in prefill_batch:
                req.state = RequestState.PREFILL
                req.started_at = time.time()
                self.decode_batch[req.request_id] = req
                
            for req in decode_batch:
                if req.state == RequestState.PREFILL:
                    # Transition from prefill to decode
                    req.state = RequestState.DECODE
            
            logger.info(
                f"Batch step: {len(prefill_batch)} prefill, {len(decode_batch)} decode, "
                f"token usage: {sum(r.total_tokens for r in prefill_batch + decode_batch)}/{total_budget}"
            )
            
            return prefill_batch, decode_batch
    
    async def _select_decode_batch(self, token_budget: int) -> List[BatchRequest]:
        """Select decode requests within token budget."""
        decode_batch = []
        tokens_used = 0
        
        # Sort by started time (FIFO for fairness)
        sorted_decode = sorted(
            self.decode_batch.values(),
            key=lambda r: r.started_at or r.created_at
        )
        
        for req in sorted_decode:
            if req.state != RequestState.DECODE:
                continue
                
            req_tokens = req.total_tokens
            if tokens_used + req_tokens <= token_budget:
                decode_batch.append(req)
                tokens_used += req_tokens
                
                if len(decode_batch) >= self.max_batch_size:
                    break
        
        return decode_batch
    
    async def _select_prefill_batch(self, token_budget: int) -> List[BatchRequest]:
        """Select prefill requests within token budget."""
        prefill_batch = []
        tokens_used = 0
        
        for req in self.prefill_queue[:]:
            req_tokens = len(req.prompt_tokens)
            if tokens_used + req_tokens <= token_budget:
                prefill_batch.append(req)
                tokens_used += req_tokens
                self.prefill_queue.remove(req)
                
                if len(prefill_batch) >= self.max_batch_size:
                    break
        
        return prefill_batch
    
    async def evict(self, num_requests: int = 1) -> List[str]:
        """Evict requests to free resources."""
        async with self._lock:
            evicted = []
            
            if self.eviction_policy == "oldest_first":
                candidates = sorted(
                    self.decode_batch.values(),
                    key=lambda r: r.started_at or r.created_at
                )
            elif self.eviction_policy == "lowest_priority":
                candidates = sorted(
                    self.decode_batch.values(),
                    key=lambda r: (r.priority, r.started_at or r.created_at)
                )
            elif self.eviction_policy == "longest_running":
                candidates = sorted(
                    self.decode_batch.values(),
                    key=lambda r: time.time() - (r.started_at or r.created_at),
                    reverse=True
                )
            else:
                candidates = list(self.decode_batch.values())
            
            for req in candidates[:num_requests]:
                req.state = RequestState.EVICTED
                req.completed_at = time.time()
                del self.decode_batch[req.request_id]
                evicted.append(req.request_id)
                self.total_evicted += 1
                
                logger.info(f"Evicted request {req.request_id}")
            
            return evicted
    
    async def cancel(self, request_id: str) -> bool:
        """Cancel a request."""
        async with self._lock:
            # Check prefill queue
            for req in self.prefill_queue:
                if req.request_id == request_id:
                    req.state = RequestState.CANCELLED
                    self.prefill_queue.remove(req)
                    self.total_cancelled += 1
                    logger.info(f"Cancelled queued request {request_id}")
                    return True
            
            # Check active batch
            if request_id in self.decode_batch:
                req = self.decode_batch[request_id]
                req.state = RequestState.CANCELLED
                req.completed_at = time.time()
                del self.decode_batch[request_id]
                self.total_cancelled += 1
                logger.info(f"Cancelled active request {request_id}")
                return True
            
            return False
    
    async def complete(self, request_id: str, tokens: List[int]) -> bool:
        """Mark request as complete."""
        async with self._lock:
            if request_id in self.decode_batch:
                req = self.decode_batch[request_id]
                req.generated_tokens = tokens
                req.state = RequestState.COMPLETE
                req.completed_at = time.time()
                del self.decode_batch[request_id]
                self.total_processed += 1
                
                logger.info(
                    f"Completed request {request_id}: "
                    f"{len(tokens)} tokens in {req.completed_at - req.started_at:.2f}s"
                )
                return True
            return False
    
    async def _check_timeouts(self):
        """Check and handle request timeouts."""
        now = time.time()
        
        # Check queued requests
        self.prefill_queue = [
            req for req in self.prefill_queue
            if now - req.created_at < req.timeout_seconds
        ]
        
        # Check active requests
        for req_id in list(self.decode_batch.keys()):
            req = self.decode_batch[req_id]
            if now - req.created_at > req.timeout_seconds:
                req.state = RequestState.TIMEOUT
                req.completed_at = now
                del self.decode_batch[req_id]
                self.total_timeout += 1
                logger.warning(f"Request {req_id} timed out")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive batching metrics."""
        # Calculate fill rate
        active_tokens = sum(r.total_tokens for r in self.decode_batch.values())
        batch_fill_rate = active_tokens / self.max_batch_tokens if self.max_batch_tokens > 0 else 0
        
        # Calculate prefill ratio
        prefill_count = sum(1 for r in self.decode_batch.values() if r.is_prefill)
        decode_count = sum(1 for r in self.decode_batch.values() if r.is_decode)
        total_active = prefill_count + decode_count
        prefill_ratio = prefill_count / max(total_active, 1)
        
        # Get cache stats from batch manager
        cache_stats = self.batch_manager.cache.get_stats() if hasattr(self.batch_manager, 'cache') else {}
        
        # Calculate tokens per second
        tokens_per_sec = 0
        if self.decode_batch:
            avg_gen_time = sum(
                (r.completed_at or time.time()) - r.started_at
                for r in self.decode_batch.values()
                if r.started_at
            ) / len(self.decode_batch)
            
            avg_tokens = sum(
                len(r.generated_tokens)
                for r in self.decode_batch.values()
            ) / max(len(self.decode_batch), 1)
            
            if avg_gen_time > 0:
                tokens_per_sec = avg_tokens / avg_gen_time
        
        return {
            "batch_fill_rate": batch_fill_rate,
            "active_requests": len(self.decode_batch),
            "queued_requests": len(self.prefill_queue),
            "prefill_ratio": prefill_ratio,
            "tokens_per_sec": tokens_per_sec,
            "total_processed": self.total_processed,
            "total_evicted": self.total_evicted,
            "total_timeout": self.total_timeout,
            "total_cancelled": self.total_cancelled,
            "cache_stats": cache_stats
        }
    
    async def shutdown(self):
        """Graceful shutdown."""
        async with self._lock:
            self._shutdown = True
            
            # Cancel all pending requests
            for req in self.prefill_queue:
                req.state = RequestState.CANCELLED
            
            for req in self.decode_batch.values():
                req.state = RequestState.CANCELLED
            
            self.prefill_queue.clear()
            self.decode_batch.clear()
            
            logger.info("Batcher shutdown complete")