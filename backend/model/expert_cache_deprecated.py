"""
DEPRECATED: This module is replaced by backend.runtime.block.expert_store

The new BlockRuntime provides unified expert caching with:
- Multi-level caching (memory + disk)
- CID verification
- Hedged fetching
- Better metrics

To migrate:
    from backend.runtime.block import ExpertStore, CacheConfig
    
    config = CacheConfig(
        memory_cache_size_mb=8192,
        disk_cache_size_mb=20480,
        ttl_seconds=3600
    )
    store = ExpertStore(config)

This file is kept for reference during migration.
"""

# Import from new location for compatibility
from backend.runtime.block.expert_store import ExpertStore as ExpertLRUCache

__all__ = ['ExpertLRUCache']