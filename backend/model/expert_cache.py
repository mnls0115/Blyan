"""
Expert Cache - Unified interface using Block Runtime

This module provides backward compatibility while using the new block runtime
for all expert caching operations.
"""

import torch
from typing import Dict, Any, Optional
from pathlib import Path

from backend.runtime.block import ExpertStore, CacheConfig
from backend.runtime.block.config import get_feature_flags


class ExpertLRUCache:
    """
    Backward-compatible expert cache using the new block runtime.
    
    This class wraps the new ExpertStore to provide the same interface
    as the old ExpertLRUCache for smooth migration.
    """
    
    def __init__(self, max_memory_gb: float = 8.0, device: str = 'cuda'):
        """Initialize expert cache using block runtime."""
        self.device = device
        self.max_memory_gb = max_memory_gb
        
        # Get feature flags to determine cache sizes
        flags = get_feature_flags()
        
        # Create cache config
        self.cache_config = CacheConfig(
            memory_cache_size_mb=int(max_memory_gb * 1024),
            disk_cache_size_mb=flags.block_runtime_disk_cache_mb,
            ttl_seconds=flags.block_runtime_cache_ttl_seconds,
            eviction_policy="lru"
        )
        
        # Initialize the new expert store
        self.store = ExpertStore(
            cache_config=self.cache_config,
            manifest_path=Path("./data/expert_manifest.json")
        )
        
        # Compatibility attributes
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, expert_key: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get expert from cache (synchronous)."""
        # Parse expert key (assumed format: "L{layer}_E{expert}")
        parts = expert_key.split("_")
        if len(parts) != 2:
            return None
        
        try:
            layer_id = int(parts[0][1:])  # Remove 'L' prefix
            expert_id = int(parts[1][1:])  # Remove 'E' prefix
        except (ValueError, IndexError):
            return None
        
        # Use async wrapper to call the store
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            expert_data = loop.run_until_complete(
                self.store.get_expert(layer_id, expert_id)
            )
            
            if expert_data:
                self.hits += 1
                return {"weights": expert_data.weights}
            else:
                self.misses += 1
                return None
                
        except Exception as e:
            self.misses += 1
            return None
    
    def put(self, expert_key: str, expert_data: Dict[str, torch.Tensor]) -> bool:
        """Add expert to cache."""
        # The new store handles caching automatically during get_expert
        # This is here for compatibility but doesn't need to do anything
        # as the store manages its own cache
        return True
    
    def clear(self):
        """Clear the cache."""
        # Reset metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Clear underlying store caches
        self.store.memory_cache.clear()
        self.store.disk_cache_lru.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        store_metrics = self.store.get_metrics()
        
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'current_size': store_metrics.get('memory_cache_size', 0),
            'memory_usage_mb': store_metrics.get('memory_cache_size', 0) * 0.1,  # Estimate
            'memory_limit_mb': self.max_memory_gb * 1024,
            'memory_utilization': store_metrics.get('cache_hit_ratio', 0),
            # Additional metrics from new store
            'disk_cache_size': store_metrics.get('disk_cache_size', 0),
            'avg_fetch_time_ms': store_metrics.get('avg_fetch_time_ms', 0)
        }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        info = {
            'cache_memory_mb': self.max_memory_gb * 1024,
            'cache_limit_mb': self.max_memory_gb * 1024,
        }
        
        if self.device == 'cuda' and torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_reserved = torch.cuda.memory_reserved(0)
            
            info.update({
                'gpu_total_mb': gpu_mem / 1024 / 1024,
                'gpu_allocated_mb': gpu_allocated / 1024 / 1024,
                'gpu_reserved_mb': gpu_reserved / 1024 / 1024,
                'gpu_free_mb': (gpu_mem - gpu_reserved) / 1024 / 1024
            })
        
        return info