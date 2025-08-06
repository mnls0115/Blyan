#!/usr/bin/env python3
"""Batch management for efficient inference with dynamic batching and caching."""

import asyncio
import time
import torch
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch management."""
    max_batch_size: int = 8
    max_wait_time_ms: int = 50          # Max time to wait for batch formation
    adaptive_batching: bool = True       # Adjust batch size based on load
    enable_padding_optimization: bool = True
    cache_size: int = 1000              # Number of cached results
    cache_ttl_seconds: int = 300        # Cache time-to-live
    enable_dynamic_batching: bool = True # Group by sequence length

@dataclass 
class InferenceRequest:
    """Individual inference request."""
    request_id: str
    prompt: str
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    future: asyncio.Future = field(default_factory=asyncio.Future)
    timestamp: float = field(default_factory=time.time)
    sequence_length: int = field(init=False)
    cache_key: str = field(init=False)
    
    def __post_init__(self):
        # Calculate sequence length (simplified - in real use would tokenize)
        self.sequence_length = len(self.prompt.split())
        
        # Generate cache key
        cache_data = f"{self.prompt}_{self.max_new_tokens}_{self.temperature}_{self.top_p}"
        self.cache_key = hashlib.md5(cache_data.encode()).hexdigest()

class ResultCache:
    """LRU cache for inference results with TTL."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[str, float]] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        
    def get(self, key: str) -> Optional[str]:
        """Get cached result if valid."""
        if key in self.cache:
            result, timestamp = self.cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[key]
                self.miss_count += 1
                return None
                
            # Move to end (LRU)
            self.cache.move_to_end(key)
            self.hit_count += 1
            return result
            
        self.miss_count += 1
        return None
        
    def put(self, key: str, result: str):
        """Store result in cache."""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
            
        self.cache[key] = (result, time.time())
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "capacity": self.max_size
        }

class BatchManager:
    """Manages dynamic batching and caching for inference requests."""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.pending_requests: List[InferenceRequest] = []
        self.request_lock = asyncio.Lock()
        self.batch_event = asyncio.Event()
        self.result_cache = ResultCache(
            max_size=self.config.cache_size,
            ttl_seconds=self.config.cache_ttl_seconds
        )
        
        # Metrics
        self.total_batches = 0
        self.total_requests = 0
        self.batch_sizes: List[int] = []
        self.wait_times: List[float] = []
        
        # Dynamic batching groups
        self.sequence_length_buckets: Dict[int, List[InferenceRequest]] = defaultdict(list)
        
    async def submit_request(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        timeout: Optional[float] = None
    ) -> str:
        """Submit inference request with batching and caching."""
        
        # Create request
        request_id = f"{time.time()}_{hash(prompt)}"
        request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        self.total_requests += 1
        
        # Check cache first
        cached_result = self.result_cache.get(request.cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for request {request_id}")
            return cached_result
            
        # Add to pending requests
        async with self.request_lock:
            self.pending_requests.append(request)
            
            # Add to sequence length bucket if dynamic batching enabled
            if self.config.enable_dynamic_batching:
                bucket_size = self._get_sequence_bucket(request.sequence_length)
                self.sequence_length_buckets[bucket_size].append(request)
                
            # Signal batch formation
            self.batch_event.set()
            
        # Wait for result
        try:
            if timeout:
                result = await asyncio.wait_for(request.future, timeout=timeout)
            else:
                result = await request.future
                
            # Cache result
            self.result_cache.put(request.cache_key, result)
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out")
            raise
            
    def _get_sequence_bucket(self, length: int) -> int:
        """Get sequence length bucket for dynamic batching."""
        # Bucket sizes: 0-16, 17-32, 33-64, 65-128, 129-256, 257+
        buckets = [16, 32, 64, 128, 256]
        
        for bucket in buckets:
            if length <= bucket:
                return bucket
                
        return 512  # Max bucket
        
    async def process_batches(self, inference_fn):
        """Process pending requests in batches."""
        
        while True:
            # Wait for requests
            await self.batch_event.wait()
            
            # Small delay to accumulate more requests
            await asyncio.sleep(self.config.max_wait_time_ms / 1000)
            
            # Form batches
            async with self.request_lock:
                if not self.pending_requests:
                    self.batch_event.clear()
                    continue
                    
                batches = self._form_batches()
                self.pending_requests.clear()
                self.sequence_length_buckets.clear()
                self.batch_event.clear()
                
            # Process each batch
            for batch in batches:
                asyncio.create_task(self._process_batch(batch, inference_fn))
                
    def _form_batches(self) -> List[List[InferenceRequest]]:
        """Form optimal batches from pending requests."""
        
        if not self.config.enable_dynamic_batching:
            # Simple batching by arrival order
            batches = []
            for i in range(0, len(self.pending_requests), self.config.max_batch_size):
                batch = self.pending_requests[i:i + self.config.max_batch_size]
                batches.append(batch)
            return batches
            
        # Dynamic batching by sequence length
        batches = []
        
        for bucket_size, requests in self.sequence_length_buckets.items():
            if not requests:
                continue
                
            # Sort by wait time (oldest first)
            requests.sort(key=lambda r: r.timestamp)
            
            # Create batches for this bucket
            for i in range(0, len(requests), self.config.max_batch_size):
                batch = requests[i:i + self.config.max_batch_size]
                batches.append(batch)
                
        return batches
        
    async def _process_batch(self, batch: List[InferenceRequest], inference_fn):
        """Process a single batch of requests."""
        
        batch_start_time = time.time()
        self.total_batches += 1
        self.batch_sizes.append(len(batch))
        
        try:
            # Calculate wait times
            for request in batch:
                wait_time = (batch_start_time - request.timestamp) * 1000
                self.wait_times.append(wait_time)
                
            # Prepare batch inputs
            if self.config.enable_padding_optimization:
                batch_inputs = self._prepare_optimized_batch(batch)
            else:
                batch_inputs = self._prepare_simple_batch(batch)
                
            # Run inference
            results = await inference_fn(batch_inputs)
            
            # Distribute results
            if isinstance(results, list):
                for request, result in zip(batch, results):
                    request.future.set_result(result)
            else:
                # Single result for whole batch
                for request in batch:
                    request.future.set_result(results)
                    
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
                    
    def _prepare_simple_batch(self, batch: List[InferenceRequest]) -> Dict[str, Any]:
        """Prepare simple batch inputs."""
        return {
            "prompts": [r.prompt for r in batch],
            "max_new_tokens": max(r.max_new_tokens for r in batch),
            "temperature": batch[0].temperature,  # Assume same for batch
            "top_p": batch[0].top_p,
            "batch_size": len(batch)
        }
        
    def _prepare_optimized_batch(self, batch: List[InferenceRequest]) -> Dict[str, Any]:
        """Prepare optimized batch with padding information."""
        
        # Group by generation parameters
        param_groups = defaultdict(list)
        for request in batch:
            key = (request.max_new_tokens, request.temperature, request.top_p)
            param_groups[key].append(request)
            
        # Create sub-batches with same parameters
        sub_batches = []
        for (max_tokens, temp, top_p), requests in param_groups.items():
            sub_batch = {
                "prompts": [r.prompt for r in requests],
                "max_new_tokens": max_tokens,
                "temperature": temp,
                "top_p": top_p,
                "sequence_lengths": [r.sequence_length for r in requests],
                "request_ids": [r.request_id for r in requests]
            }
            sub_batches.append(sub_batch)
            
        return {
            "sub_batches": sub_batches,
            "total_requests": len(batch),
            "optimized": True
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """Get batch manager statistics."""
        avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
        avg_wait_time = sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0
        
        return {
            "total_batches": self.total_batches,
            "total_requests": self.total_requests,
            "average_batch_size": avg_batch_size,
            "average_wait_time_ms": avg_wait_time,
            "cache_stats": self.result_cache.get_stats(),
            "pending_requests": len(self.pending_requests),
            "config": {
                "max_batch_size": self.config.max_batch_size,
                "max_wait_time_ms": self.config.max_wait_time_ms,
                "dynamic_batching": self.config.enable_dynamic_batching
            }
        }

# Integration with model inference
class BatchedInferenceEngine:
    """Engine that combines batch manager with actual inference."""
    
    def __init__(self, model, config: BatchConfig = None):
        self.model = model
        self.batch_manager = BatchManager(config)
        self.processing_task = None
        
    async def start(self):
        """Start batch processing."""
        self.processing_task = asyncio.create_task(
            self.batch_manager.process_batches(self._run_inference)
        )
        logger.info("Batched inference engine started")
        
    async def stop(self):
        """Stop batch processing."""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
                
    async def infer(self, prompt: str, **kwargs) -> str:
        """Submit inference request."""
        return await self.batch_manager.submit_request(prompt, **kwargs)
        
    async def _run_inference(self, batch_inputs: Dict[str, Any]) -> List[str]:
        """Run actual model inference on batch."""
        
        # Handle optimized batches
        if batch_inputs.get("optimized"):
            all_results = []
            for sub_batch in batch_inputs["sub_batches"]:
                results = await self._run_sub_batch(sub_batch)
                all_results.extend(results)
            return all_results
            
        # Simple batch
        prompts = batch_inputs["prompts"]
        
        # Mock inference for demo
        await asyncio.sleep(0.1)  # Simulate inference time
        
        results = [f"Response to: {prompt[:50]}..." for prompt in prompts]
        return results
        
    async def _run_sub_batch(self, sub_batch: Dict[str, Any]) -> List[str]:
        """Process optimized sub-batch."""
        # In real implementation, would handle padding and batching properly
        prompts = sub_batch["prompts"]
        max_tokens = sub_batch["max_new_tokens"]
        
        # Mock inference
        await asyncio.sleep(0.05 * len(prompts))
        
        results = [f"Optimized response (max_tokens={max_tokens}): {p[:30]}..." for p in prompts]
        return results

if __name__ == "__main__":
    # Basic test of batch manager functionality
    import asyncio
    
    async def test_batch_manager():
        config = BatchConfig(max_batch_size=4, max_wait_time_ms=50)
        manager = BatchManager(config)
        print("BatchManager initialized successfully")
        print(f"Config: max_batch_size={config.max_batch_size}, max_wait_time={config.max_wait_time_ms}ms")
        print(f"Cache size: {config.cache_size}, Dynamic batching: {config.enable_dynamic_batching}")
        
    asyncio.run(test_batch_manager())
    
# Production code - demo removed