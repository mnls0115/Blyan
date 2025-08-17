"""Expert weights cache API facade for MoE inference.

Integrates with backend/model/expert_cache.py and P2P hot cache.
"""

import asyncio
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import torch
import structlog

from backend.model.expert_cache import ExpertLRUCache
from backend.p2p.hot_expert_cache import HotExpertCache

logger = structlog.get_logger()


class ExpertCacheAPI:
    """Unified expert cache API with prewarming and prefetching."""
    
    def __init__(
        self,
        local_cache: Optional[ExpertLRUCache] = None,
        hot_cache: Optional[HotExpertCache] = None,
        max_memory_gb: float = 8.0,
        prefetch_queue_size: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        # Local LRU cache
        self.local_cache = local_cache or ExpertLRUCache(
            max_memory_gb=max_memory_gb,
            device=device
        )
        
        # P2P hot cache (optional)
        self.hot_cache = hot_cache
        
        # Prefetch management
        self.prefetch_queue: OrderedDict[str, asyncio.Task] = OrderedDict()
        self.prefetch_queue_size = prefetch_queue_size
        self.prefetch_lock = asyncio.Lock()
        
        # Prewarm tracking
        self.prewarmed_experts: Set[str] = set()
        
        # Metrics
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.prewarm_count = 0
        
    def get(self, expert_id: str) -> Optional[Dict[str, Any]]:
        """Get expert from cache (synchronous).
        
        Checks local cache first, then hot cache if available.
        """
        # Check local cache
        expert_data = self.local_cache.get(expert_id)
        if expert_data is not None:
            return expert_data
        
        # Check P2P hot cache if available
        if self.hot_cache:
            try:
                # Hot cache might be async, but we need sync interface
                # In production, would use sync wrapper or thread pool
                expert_data = self._get_from_hot_cache_sync(expert_id)
                if expert_data is not None:
                    # Add to local cache
                    self.local_cache.put(expert_id, expert_data)
                    return expert_data
            except Exception as e:
                logger.warning(f"Hot cache fetch failed for {expert_id}: {e}")
        
        return None
    
    async def get_async(self, expert_id: str) -> Optional[Dict[str, Any]]:
        """Get expert from cache (async version)."""
        # Check if being prefetched
        async with self.prefetch_lock:
            if expert_id in self.prefetch_queue:
                # Wait for prefetch to complete
                task = self.prefetch_queue[expert_id]
                try:
                    await task
                    self.prefetch_hits += 1
                except Exception as e:
                    logger.error(f"Prefetch failed for {expert_id}: {e}")
                    self.prefetch_misses += 1
        
        # Now check cache
        return self.get(expert_id)
    
    def prewarm(self, expert_ids: List[str]) -> int:
        """Prewarm cache with specified experts.
        
        Returns number of experts successfully prewarmed.
        """
        success_count = 0
        
        for expert_id in expert_ids:
            if expert_id in self.prewarmed_experts:
                continue  # Already prewarmed
            
            try:
                # Load expert if not in cache
                if not self.local_cache.get(expert_id):
                    expert_data = self._load_expert(expert_id)
                    if expert_data:
                        self.local_cache.put(expert_id, expert_data)
                        success_count += 1
                
                self.prewarmed_experts.add(expert_id)
                
            except Exception as e:
                logger.error(f"Failed to prewarm {expert_id}: {e}")
        
        self.prewarm_count += success_count
        logger.info(f"Prewarmed {success_count}/{len(expert_ids)} experts")
        
        return success_count
    
    async def prefetch_async(self, expert_ids: List[str]):
        """Asynchronously prefetch experts into cache."""
        async with self.prefetch_lock:
            for expert_id in expert_ids:
                # Skip if already cached or being fetched
                if self.local_cache.get(expert_id) or expert_id in self.prefetch_queue:
                    continue
                
                # Manage queue size
                while len(self.prefetch_queue) >= self.prefetch_queue_size:
                    # Remove oldest prefetch
                    oldest_id, oldest_task = self.prefetch_queue.popitem(last=False)
                    if not oldest_task.done():
                        oldest_task.cancel()
                
                # Start prefetch task
                task = asyncio.create_task(self._prefetch_expert(expert_id))
                self.prefetch_queue[expert_id] = task
    
    async def _prefetch_expert(self, expert_id: str):
        """Prefetch single expert."""
        try:
            # Simulate async loading (in production, would be actual I/O)
            await asyncio.sleep(0.1)  # Simulated load time
            
            expert_data = self._load_expert(expert_id)
            if expert_data:
                self.local_cache.put(expert_id, expert_data)
                logger.debug(f"Prefetched expert {expert_id}")
            
        except asyncio.CancelledError:
            logger.debug(f"Prefetch cancelled for {expert_id}")
            raise
        except Exception as e:
            logger.error(f"Prefetch failed for {expert_id}: {e}")
        finally:
            # Remove from queue
            async with self.prefetch_lock:
                self.prefetch_queue.pop(expert_id, None)
    
    def _load_expert(self, expert_id: str) -> Optional[Dict[str, Any]]:
        """Load expert from disk or blockchain.
        
        This is a placeholder - integrate with your actual loading logic.
        """
        # In production, load from:
        # 1. Local model files
        # 2. Blockchain data
        # 3. P2P network
        
        # For now, return mock data
        return {
            "expert_id": expert_id,
            "weights": torch.randn(1024, 1024, device=self.local_cache.device),
            "loaded_at": time.time()
        }
    
    def _get_from_hot_cache_sync(self, expert_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous wrapper for hot cache access."""
        # In production, use proper async/sync bridge
        # For now, return None (hot cache is async-only)
        return None
    
    def evict(self, expert_id: Optional[str] = None) -> bool:
        """Evict expert from cache.
        
        If expert_id is None, evicts LRU expert.
        """
        if expert_id:
            # Evict specific expert
            success = self.local_cache.evict_specific(expert_id)
            if success and expert_id in self.prewarmed_experts:
                self.prewarmed_experts.remove(expert_id)
            return success
        else:
            # Evict LRU
            evicted_id = self.local_cache._evict_lru()
            if evicted_id and evicted_id in self.prewarmed_experts:
                self.prewarmed_experts.remove(evicted_id)
            return evicted_id is not None
    
    def clear(self):
        """Clear entire cache."""
        self.local_cache.clear()
        self.prewarmed_experts.clear()
        
        # Cancel all prefetch tasks
        for task in self.prefetch_queue.values():
            if not task.done():
                task.cancel()
        self.prefetch_queue.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        local_stats = self.local_cache.get_stats()
        
        return {
            "local_cache": local_stats,
            "prewarmed_count": len(self.prewarmed_experts),
            "prefetch_queue_size": len(self.prefetch_queue),
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "total_prewarms": self.prewarm_count,
            "memory_usage_gb": local_stats.get("current_memory_gb", 0)
        }
    
    def suggest_experts_to_cache(self, routing_history: List[List[str]], limit: int = 10) -> List[str]:
        """Suggest experts to cache based on routing history.
        
        Args:
            routing_history: List of expert ID lists from recent requests
            limit: Maximum number of suggestions
            
        Returns:
            List of expert IDs to cache
        """
        # Count expert frequency
        expert_counts = {}
        for expert_list in routing_history:
            for expert_id in expert_list:
                expert_counts[expert_id] = expert_counts.get(expert_id, 0) + 1
        
        # Sort by frequency
        sorted_experts = sorted(
            expert_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Filter out already cached
        suggestions = []
        for expert_id, count in sorted_experts:
            if not self.local_cache.get(expert_id):
                suggestions.append(expert_id)
                if len(suggestions) >= limit:
                    break
        
        return suggestions