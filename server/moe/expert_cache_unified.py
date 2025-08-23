"""
Unified Expert Cache API using Block Runtime

This module replaces the old expert_cache.py with a unified implementation
that leverages the block runtime for consistency across all nodes.
"""

import asyncio
from typing import Dict, List, Optional, Any
import torch

from backend.runtime.block import BlockRuntime, RequestSpec, RuntimeConfig, CacheConfig
from backend.runtime.block.expert_store import ExpertStore
from backend.runtime.block.config import get_feature_flags


class UnifiedExpertCache:
    """
    Unified expert cache that uses block runtime for all operations.
    
    Provides both sync and async interfaces for compatibility.
    """
    
    def __init__(self, runtime: Optional[BlockRuntime] = None):
        """Initialize with optional runtime instance."""
        if runtime is None:
            # Create default runtime
            flags = get_feature_flags()
            config = RuntimeConfig(
                cache_config=CacheConfig(
                    memory_cache_size_mb=flags.block_runtime_memory_cache_mb,
                    disk_cache_size_mb=flags.block_runtime_disk_cache_mb,
                    ttl_seconds=flags.block_runtime_cache_ttl_seconds
                ),
                enable_metrics=True
            )
            self.runtime = BlockRuntime(config=config)
        else:
            self.runtime = runtime
        
        # Direct access to expert store for cache operations
        self.store = self.runtime.expert_store
        
        # Metrics
        self.prefetch_hits = 0
        self.prefetch_misses = 0
    
    async def get_expert_async(
        self, 
        layer_id: int, 
        expert_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get expert asynchronously."""
        try:
            expert_data = await self.store.get_expert(layer_id, expert_id)
            if expert_data:
                return {
                    "weights": expert_data.weights,
                    "verified": expert_data.verified,
                    "cache_hit": expert_data.cache_hit,
                    "fetch_latency_ms": expert_data.fetch_latency_ms
                }
            return None
        except Exception as e:
            print(f"Error getting expert L{layer_id}_E{expert_id}: {e}")
            return None
    
    def get_expert_sync(
        self, 
        layer_id: int, 
        expert_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get expert synchronously."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.get_expert_async(layer_id, expert_id)
        )
    
    async def prefetch_experts(
        self, 
        layer_experts: Dict[int, List[int]]
    ) -> None:
        """Prefetch multiple experts."""
        await self.store.prefetch_experts(layer_experts)
        self.prefetch_hits += 1
    
    async def prewarm_experts(
        self, 
        expert_list: List[tuple[int, int]]
    ) -> int:
        """
        Prewarm cache with specified experts.
        
        Args:
            expert_list: List of (layer_id, expert_id) tuples
            
        Returns:
            Number of experts successfully prewarmed
        """
        layer_experts = {}
        for layer_id, expert_id in expert_list:
            if layer_id not in layer_experts:
                layer_experts[layer_id] = []
            layer_experts[layer_id].append(expert_id)
        
        await self.prefetch_experts(layer_experts)
        return len(expert_list)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        store_metrics = self.store.get_metrics()
        runtime_metrics = self.runtime.get_metrics()
        
        return {
            "cache_hit_ratio": store_metrics["cache_hit_ratio"],
            "total_requests": store_metrics["total_requests"],
            "cache_hits": store_metrics["cache_hits"],
            "cache_misses": store_metrics["cache_misses"],
            "memory_cache_size": store_metrics["memory_cache_size"],
            "disk_cache_size": store_metrics["disk_cache_size"],
            "avg_fetch_time_ms": store_metrics["avg_fetch_time_ms"],
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "runtime_metrics": runtime_metrics
        }
    
    def suggest_experts_to_cache(
        self, 
        usage_history: List[Dict[int, List[int]]], 
        limit: int = 10
    ) -> List[tuple[int, int]]:
        """
        Suggest experts to cache based on usage history.
        
        Args:
            usage_history: List of layer_experts dicts from recent requests
            limit: Maximum number of suggestions
            
        Returns:
            List of (layer_id, expert_id) tuples to cache
        """
        # Count expert usage frequency
        expert_counts = {}
        for layer_experts in usage_history:
            for layer_id, expert_ids in layer_experts.items():
                for expert_id in expert_ids:
                    key = (layer_id, expert_id)
                    expert_counts[key] = expert_counts.get(key, 0) + 1
        
        # Sort by frequency
        sorted_experts = sorted(
            expert_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top suggestions
        suggestions = []
        for (layer_id, expert_id), count in sorted_experts:
            # Check if not already cached (would need async check)
            suggestions.append((layer_id, expert_id))
            if len(suggestions) >= limit:
                break
        
        return suggestions
    
    async def clear_cache(self):
        """Clear all caches."""
        # Clear memory cache
        self.store.memory_cache.clear()
        
        # Clear disk cache
        import shutil
        if self.store.disk_cache_path.exists():
            shutil.rmtree(self.store.disk_cache_path)
            self.store.disk_cache_path.mkdir(parents=True, exist_ok=True)
        
        # Reset metrics
        self.store.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "fetch_successes": 0,
            "fetch_failures": 0,
            "verification_successes": 0,
            "verification_failures": 0,
            "total_fetch_time_ms": 0
        }
    
    async def shutdown(self):
        """Shutdown the cache and runtime."""
        await self.runtime.shutdown()


# Global instance for backward compatibility
_global_cache: Optional[UnifiedExpertCache] = None


def get_expert_cache() -> UnifiedExpertCache:
    """Get or create global expert cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = UnifiedExpertCache()
    return _global_cache


# Compatibility exports
ExpertCacheAPI = UnifiedExpertCache  # Alias for old name