#!/usr/bin/env python3
"""
KV-Cache Management System for Efficient Inference
Manages key-value caches across multiple requests for memory efficiency
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import logging
from collections import OrderedDict
import asyncio
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Single KV-cache entry for a sequence."""
    request_id: str
    key_cache: torch.Tensor  # [num_layers, num_heads, seq_len, head_dim]
    value_cache: torch.Tensor
    seq_length: int
    last_accessed: float
    memory_bytes: int
    is_active: bool = True
    shared_prefix_id: Optional[str] = None  # For prefix sharing

class KVCachePool:
    """
    Manages a pool of KV-caches with memory limits and eviction policies.
    """
    
    def __init__(
        self,
        max_memory_gb: float = 8.0,
        cache_dtype: torch.dtype = torch.float16,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128
    ):
        self.max_memory_bytes = int(max_memory_gb * 1e9)
        self.cache_dtype = cache_dtype
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Cache storage
        self.caches: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory_usage = 0
        
        # Prefix sharing for common prompts
        self.prefix_cache: Dict[str, CacheEntry] = {}
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        
    def allocate_cache(
        self,
        request_id: str,
        seq_length: int,
        device: torch.device = torch.device('cuda')
    ) -> CacheEntry:
        """
        Allocate a new KV-cache for a request.
        """
        # Calculate memory requirement
        cache_shape = (self.num_layers, self.num_heads, seq_length, self.head_dim)
        bytes_per_element = 2 if self.cache_dtype == torch.float16 else 4
        memory_required = 2 * np.prod(cache_shape) * bytes_per_element  # 2 for K and V
        
        # Check if eviction is needed
        while self.current_memory_usage + memory_required > self.max_memory_bytes:
            if not self._evict_lru():
                raise RuntimeError(f"Cannot allocate {memory_required / 1e6:.1f}MB cache, pool exhausted")
        
        # Allocate tensors
        key_cache = torch.zeros(cache_shape, dtype=self.cache_dtype, device=device)
        value_cache = torch.zeros(cache_shape, dtype=self.cache_dtype, device=device)
        
        # Create cache entry
        entry = CacheEntry(
            request_id=request_id,
            key_cache=key_cache,
            value_cache=value_cache,
            seq_length=seq_length,
            last_accessed=time.time(),
            memory_bytes=memory_required
        )
        
        # Store in pool
        self.caches[request_id] = entry
        self.current_memory_usage += memory_required
        
        logger.debug(f"Allocated {memory_required / 1e6:.1f}MB cache for request {request_id}")
        
        return entry
    
    def get_cache(self, request_id: str) -> Optional[CacheEntry]:
        """
        Retrieve cache for a request, updating access time.
        """
        if request_id in self.caches:
            entry = self.caches[request_id]
            entry.last_accessed = time.time()
            
            # Move to end (most recently used)
            self.caches.move_to_end(request_id)
            
            self.cache_hits += 1
            return entry
        
        self.cache_misses += 1
        return None
    
    def release_cache(self, request_id: str):
        """
        Mark cache as inactive (can be evicted).
        """
        if request_id in self.caches:
            self.caches[request_id].is_active = False
            logger.debug(f"Released cache for request {request_id}")
    
    def _evict_lru(self) -> bool:
        """
        Evict least recently used inactive cache.
        """
        # Find oldest inactive cache
        for request_id, entry in self.caches.items():
            if not entry.is_active:
                # Evict this entry
                self.current_memory_usage -= entry.memory_bytes
                del self.caches[request_id]
                self.evictions += 1
                
                logger.debug(f"Evicted cache for request {request_id}, freed {entry.memory_bytes / 1e6:.1f}MB")
                return True
        
        # No inactive caches to evict
        logger.warning("No inactive caches available for eviction")
        return False
    
    def share_prefix(self, request_id: str, prefix_text: str, prefix_length: int):
        """
        Enable prefix sharing for common prompts (e.g., system prompts).
        """
        prefix_hash = hash(prefix_text)
        
        if prefix_hash in self.prefix_cache:
            # Reuse existing prefix cache
            shared_entry = self.prefix_cache[prefix_hash]
            if request_id in self.caches:
                self.caches[request_id].shared_prefix_id = str(prefix_hash)
            
            logger.debug(f"Shared prefix cache for request {request_id}")
            return shared_entry
        
        # Create new prefix cache
        # (Implementation would copy first prefix_length tokens from request cache)
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache pool statistics."""
        active_caches = sum(1 for e in self.caches.values() if e.is_active)
        inactive_caches = len(self.caches) - active_caches
        
        return {
            "total_caches": len(self.caches),
            "active_caches": active_caches,
            "inactive_caches": inactive_caches,
            "memory_usage_gb": self.current_memory_usage / 1e9,
            "memory_limit_gb": self.max_memory_bytes / 1e9,
            "memory_utilization": self.current_memory_usage / self.max_memory_bytes,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            "evictions": self.evictions,
            "prefix_caches": len(self.prefix_cache)
        }

class PagedAttention:
    """
    Paged attention for efficient KV-cache management (inspired by vLLM).
    Stores KV-cache in fixed-size pages instead of contiguous memory.
    """
    
    def __init__(
        self,
        block_size: int = 16,  # Tokens per block
        num_blocks: int = 1024,  # Total blocks in pool
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        
        # Block pool (pre-allocated)
        self.key_blocks = torch.zeros(
            (num_blocks, num_layers, num_heads, block_size, head_dim),
            dtype=dtype,
            device='cuda'
        )
        self.value_blocks = torch.zeros(
            (num_blocks, num_layers, num_heads, block_size, head_dim),
            dtype=dtype,
            device='cuda'
        )
        
        # Block allocation table
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks: Dict[str, List[int]] = {}  # request_id -> block indices
        
    def allocate_blocks(self, request_id: str, num_tokens: int) -> List[int]:
        """
        Allocate blocks for a request.
        """
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError(f"Not enough free blocks: need {num_blocks_needed}, have {len(self.free_blocks)}")
        
        # Allocate blocks
        allocated = []
        for _ in range(num_blocks_needed):
            block_idx = self.free_blocks.pop(0)
            allocated.append(block_idx)
        
        self.allocated_blocks[request_id] = allocated
        
        logger.debug(f"Allocated {num_blocks_needed} blocks for request {request_id}")
        return allocated
    
    def free_blocks(self, request_id: str):
        """
        Free blocks allocated to a request.
        """
        if request_id in self.allocated_blocks:
            blocks = self.allocated_blocks.pop(request_id)
            self.free_blocks.extend(blocks)
            
            # Clear block contents (optional, for security)
            for block_idx in blocks:
                self.key_blocks[block_idx].zero_()
                self.value_blocks[block_idx].zero_()
            
            logger.debug(f"Freed {len(blocks)} blocks from request {request_id}")
    
    def read_cache(
        self,
        request_id: str,
        layer: int,
        head: int,
        token_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read KV-cache for specific token positions.
        """
        if request_id not in self.allocated_blocks:
            raise KeyError(f"No blocks allocated for request {request_id}")
        
        blocks = self.allocated_blocks[request_id]
        
        # Map token positions to blocks
        block_indices = token_positions // self.block_size
        block_offsets = token_positions % self.block_size
        
        # Gather from blocks
        keys = []
        values = []
        
        for block_idx, offset in zip(block_indices, block_offsets):
            if block_idx >= len(blocks):
                logger.warning(f"Token position out of allocated range")
                continue
            
            actual_block = blocks[block_idx]
            keys.append(self.key_blocks[actual_block, layer, head, offset])
            values.append(self.value_blocks[actual_block, layer, head, offset])
        
        return torch.stack(keys), torch.stack(values)
    
    def write_cache(
        self,
        request_id: str,
        layer: int,
        head: int,
        token_position: int,
        key: torch.Tensor,
        value: torch.Tensor
    ):
        """
        Write KV-cache for a specific token position.
        """
        if request_id not in self.allocated_blocks:
            raise KeyError(f"No blocks allocated for request {request_id}")
        
        blocks = self.allocated_blocks[request_id]
        
        # Map to block
        block_idx = token_position // self.block_size
        block_offset = token_position % self.block_size
        
        if block_idx >= len(blocks):
            raise IndexError(f"Token position {token_position} exceeds allocated blocks")
        
        actual_block = blocks[block_idx]
        
        # Write to block
        self.key_blocks[actual_block, layer, head, block_offset] = key
        self.value_blocks[actual_block, layer, head, block_offset] = value
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        bytes_per_element = 2 if self.dtype == torch.float16 else 4
        block_size_bytes = (
            self.block_size * self.num_layers * self.num_heads * self.head_dim * 
            2 * bytes_per_element  # 2 for K and V
        )
        
        return {
            "total_blocks": self.num_blocks,
            "free_blocks": len(self.free_blocks),
            "allocated_blocks": self.num_blocks - len(self.free_blocks),
            "block_size_mb": block_size_bytes / 1e6,
            "total_memory_gb": (self.num_blocks * block_size_bytes) / 1e9,
            "used_memory_gb": ((self.num_blocks - len(self.free_blocks)) * block_size_bytes) / 1e9,
            "requests_served": len(self.allocated_blocks)
        }

class ContinuousBatching:
    """
    Continuous batching for maximizing GPU utilization.
    Dynamically batches requests at different generation stages.
    """
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_sequence_length: int = 2048
    ):
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        
        # Request queues by priority
        self.high_priority_queue = asyncio.Queue()
        self.normal_priority_queue = asyncio.Queue()
        
        # Active batch
        self.active_batch: List[str] = []
        self.batch_positions: Dict[str, int] = {}  # request_id -> position in batch
        
    async def add_request(
        self,
        request_id: str,
        tokens: torch.Tensor,
        priority: str = "normal"
    ):
        """Add request to appropriate queue."""
        request_data = {
            "id": request_id,
            "tokens": tokens,
            "position": 0,
            "completed": False
        }
        
        if priority == "high":
            await self.high_priority_queue.put(request_data)
        else:
            await self.normal_priority_queue.put(request_data)
    
    async def get_next_batch(self) -> List[Dict[str, Any]]:
        """
        Get next batch of requests to process.
        Combines ongoing and new requests.
        """
        batch = []
        
        # Keep ongoing requests in batch
        ongoing = [req_id for req_id in self.active_batch if not self._is_completed(req_id)]
        batch.extend(ongoing)
        
        # Fill remaining slots with new requests
        remaining_slots = self.max_batch_size - len(batch)
        
        # Prioritize high priority queue
        while remaining_slots > 0 and not self.high_priority_queue.empty():
            try:
                request = await asyncio.wait_for(self.high_priority_queue.get(), timeout=0.001)
                batch.append(request)
                remaining_slots -= 1
            except asyncio.TimeoutError:
                break
        
        # Fill with normal priority
        while remaining_slots > 0 and not self.normal_priority_queue.empty():
            try:
                request = await asyncio.wait_for(self.normal_priority_queue.get(), timeout=0.001)
                batch.append(request)
                remaining_slots -= 1
            except asyncio.TimeoutError:
                break
        
        self.active_batch = [req["id"] if isinstance(req, dict) else req for req in batch]
        
        return batch
    
    def _is_completed(self, request_id: str) -> bool:
        """Check if request has completed generation."""
        # Implementation would check actual generation state
        return False
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        return {
            "active_batch_size": len(self.active_batch),
            "max_batch_size": self.max_batch_size,
            "utilization": len(self.active_batch) / self.max_batch_size,
            "high_priority_pending": self.high_priority_queue.qsize(),
            "normal_priority_pending": self.normal_priority_queue.qsize()
        }

# Singleton instances
_kv_cache_pool = None
_paged_attention = None
_continuous_batching = None

def get_kv_cache_pool(max_memory_gb: float = 8.0) -> KVCachePool:
    """Get or create KV-cache pool."""
    global _kv_cache_pool
    if _kv_cache_pool is None:
        _kv_cache_pool = KVCachePool(max_memory_gb=max_memory_gb)
    return _kv_cache_pool

def get_paged_attention() -> PagedAttention:
    """Get or create paged attention manager."""
    global _paged_attention
    if _paged_attention is None:
        _paged_attention = PagedAttention()
    return _paged_attention

def get_continuous_batching() -> ContinuousBatching:
    """Get or create continuous batching manager."""
    global _continuous_batching
    if _continuous_batching is None:
        _continuous_batching = ContinuousBatching()
    return _continuous_batching