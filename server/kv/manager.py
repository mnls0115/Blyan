"""Modular KV-Cache manager facade with model/tokenizer versioning.

Wraps backend/optimization/kv_cache_manager.py with enhanced keying.
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
import structlog

from backend.optimization.kv_cache_manager import KVCachePool, CacheEntry

logger = structlog.get_logger()


@dataclass
class KVCacheKey:
    """Enhanced cache key with model/tokenizer versioning."""
    model_version: str
    model_hash: str
    tokenizer_hash: str
    system_prompt_id: Optional[str]
    prefix_fingerprint: str
    
    def to_string(self) -> str:
        """Convert to string key."""
        components = [
            self.model_version,
            self.model_hash[:8],  # Short hash
            self.tokenizer_hash[:8],
            self.system_prompt_id or "none",
            self.prefix_fingerprint
        ]
        return ":".join(components)
    
    @classmethod
    def from_string(cls, key_str: str) -> 'KVCacheKey':
        """Parse from string key."""
        parts = key_str.split(":")
        return cls(
            model_version=parts[0],
            model_hash=parts[1],
            tokenizer_hash=parts[2],
            system_prompt_id=parts[3] if parts[3] != "none" else None,
            prefix_fingerprint=parts[4]
        )


class KVCacheManager:
    """Enhanced KV-Cache manager with versioning and fingerprinting."""
    
    def __init__(
        self,
        kv_pool: Optional[KVCachePool] = None,
        max_memory_gb: float = 8.0,
        admission_policy: str = "always",
        eviction_policy: str = "LRU",
        prefix_cache_enabled: bool = True,
        fingerprint_length: int = 64
    ):
        # Underlying cache pool
        self.kv_pool = kv_pool or KVCachePool(
            max_memory_gb=max_memory_gb,
            cache_dtype=torch.float16
        )
        
        # Policies
        self.admission_policy = admission_policy
        self.eviction_policy = eviction_policy
        self.prefix_cache_enabled = prefix_cache_enabled
        self.fingerprint_length = fingerprint_length
        
        # Enhanced key mapping
        self.key_mapping: Dict[str, str] = {}  # KVCacheKey.to_string() -> request_id
        self.reverse_mapping: Dict[str, KVCacheKey] = {}  # request_id -> KVCacheKey
        
        # Prefix deduplication
        self.prefix_groups: Dict[str, List[str]] = {}  # prefix_fingerprint -> [request_ids]
        
        # Metrics
        self.versioned_hits = 0
        self.versioned_misses = 0
        self.prefix_reuse_count = 0
    
    @staticmethod
    def compute_prefix_fingerprint(tokens: List[int], max_tokens: int = 64) -> str:
        """Compute deterministic fingerprint for token prefix."""
        prefix = tokens[:max_tokens]
        
        # Use SHA256 for deterministic hashing
        hasher = hashlib.sha256()
        for token in prefix:
            hasher.update(token.to_bytes(4, byteorder='little'))
        
        return hasher.hexdigest()[:16]  # 16 hex chars
    
    def make_key(
        self,
        model_version: str,
        model_hash: str,
        tokenizer_hash: str,
        tokens: List[int],
        system_prompt_id: Optional[str] = None
    ) -> KVCacheKey:
        """Create enhanced cache key."""
        return KVCacheKey(
            model_version=model_version,
            model_hash=model_hash,
            tokenizer_hash=tokenizer_hash,
            system_prompt_id=system_prompt_id,
            prefix_fingerprint=self.compute_prefix_fingerprint(tokens, self.fingerprint_length)
        )
    
    def admit(
        self,
        key: KVCacheKey,
        kv_tensors: Tuple[torch.Tensor, torch.Tensor],
        request_id: Optional[str] = None
    ) -> bool:
        """Add KV cache entry with admission control.
        
        Args:
            key: Enhanced cache key
            kv_tensors: Tuple of (key_cache, value_cache) tensors
            request_id: Optional request ID (generated if not provided)
            
        Returns:
            True if admitted, False otherwise
        """
        # Check admission policy
        if not self._should_admit(key, kv_tensors):
            logger.debug(f"Admission denied for key {key.to_string()}")
            return False
        
        # Generate request ID if needed
        if request_id is None:
            request_id = f"kv_{hash(key.to_string())}_{time.time()}"
        
        # Check for prefix sharing opportunity
        if self.prefix_cache_enabled and key.prefix_fingerprint in self.prefix_groups:
            # Reuse existing prefix cache
            existing_requests = self.prefix_groups[key.prefix_fingerprint]
            if existing_requests:
                logger.info(f"Reusing prefix cache from {existing_requests[0]}")
                self.prefix_reuse_count += 1
        
        # Store in underlying pool
        try:
            # Create cache entry
            entry = CacheEntry(
                request_id=request_id,
                key_cache=kv_tensors[0],
                value_cache=kv_tensors[1],
                seq_length=kv_tensors[0].shape[2],  # seq_len dimension
                last_accessed=time.time(),
                memory_bytes=kv_tensors[0].nbytes + kv_tensors[1].nbytes,
                shared_prefix_id=key.prefix_fingerprint if self.prefix_cache_enabled else None
            )
            
            # Add to pool
            self.kv_pool.caches[request_id] = entry
            self.kv_pool.current_memory_usage += entry.memory_bytes
            
            # Update mappings
            key_str = key.to_string()
            self.key_mapping[key_str] = request_id
            self.reverse_mapping[request_id] = key
            
            # Update prefix groups
            if self.prefix_cache_enabled:
                if key.prefix_fingerprint not in self.prefix_groups:
                    self.prefix_groups[key.prefix_fingerprint] = []
                self.prefix_groups[key.prefix_fingerprint].append(request_id)
            
            logger.debug(f"Admitted cache entry: {key_str} -> {request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to admit cache entry: {e}")
            return False
    
    def get(self, key: KVCacheKey) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve KV cache by enhanced key.
        
        Returns:
            Tuple of (key_cache, value_cache) or None if not found
        """
        key_str = key.to_string()
        
        # Check if key exists
        if key_str not in self.key_mapping:
            self.versioned_misses += 1
            
            # Check for compatible versions (relaxed matching)
            compatible_key = self._find_compatible_key(key)
            if compatible_key:
                logger.info(f"Using compatible cache: {compatible_key}")
                key_str = compatible_key
            else:
                return None
        
        # Get request ID
        request_id = self.key_mapping[key_str]
        
        # Get from pool
        entry = self.kv_pool.get_cache(request_id)
        if entry:
            self.versioned_hits += 1
            return (entry.key_cache, entry.value_cache)
        
        self.versioned_misses += 1
        return None
    
    def evict(self, policy: Optional[str] = None) -> bool:
        """Force eviction using specified policy.
        
        Args:
            policy: Eviction policy ("LRU", "LFU", "largest")
            
        Returns:
            True if eviction successful
        """
        policy = policy or self.eviction_policy
        
        if policy == "LRU":
            success = self.kv_pool._evict_lru()
        elif policy == "largest":
            # Find and evict largest entry
            if not self.kv_pool.caches:
                return False
            
            largest_id = max(
                self.kv_pool.caches.keys(),
                key=lambda k: self.kv_pool.caches[k].memory_bytes
            )
            entry = self.kv_pool.caches.pop(largest_id)
            self.kv_pool.current_memory_usage -= entry.memory_bytes
            self.kv_pool.evictions += 1
            
            # Clean up mappings
            self._cleanup_mappings(largest_id)
            success = True
        else:
            # Default to LRU
            success = self.kv_pool._evict_lru()
        
        if success:
            logger.debug(f"Evicted cache using {policy} policy")
        
        return success
    
    def _should_admit(self, key: KVCacheKey, kv_tensors: Tuple[torch.Tensor, torch.Tensor]) -> bool:
        """Check admission policy."""
        if self.admission_policy == "always":
            return True
        
        elif self.admission_policy == "size_threshold":
            # Admit only if size is reasonable
            size_bytes = kv_tensors[0].nbytes + kv_tensors[1].nbytes
            max_single_cache = self.kv_pool.max_memory_bytes * 0.1  # 10% of total
            return size_bytes <= max_single_cache
        
        elif self.admission_policy == "no_system_prompt":
            # Don't cache system prompts (they're usually static)
            return key.system_prompt_id is None
        
        elif self.admission_policy == "versioned_only":
            # Only cache if we have proper version info
            return key.model_hash != "unknown" and key.tokenizer_hash != "unknown"
        
        return True
    
    def _find_compatible_key(self, key: KVCacheKey) -> Optional[str]:
        """Find compatible cache key with relaxed matching."""
        # Try to find cache with same prefix but different version
        for cached_key_str in self.key_mapping.keys():
            cached_key = KVCacheKey.from_string(cached_key_str)
            
            # Check if prefix matches (most important for KV reuse)
            if cached_key.prefix_fingerprint == key.prefix_fingerprint:
                # Check if model is compatible (same architecture)
                if cached_key.model_version.split(".")[0] == key.model_version.split(".")[0]:
                    return cached_key_str
        
        return None
    
    def _cleanup_mappings(self, request_id: str):
        """Clean up mappings when entry is evicted."""
        if request_id in self.reverse_mapping:
            key = self.reverse_mapping[request_id]
            key_str = key.to_string()
            
            # Remove from mappings
            self.key_mapping.pop(key_str, None)
            del self.reverse_mapping[request_id]
            
            # Remove from prefix groups
            if self.prefix_cache_enabled and key.prefix_fingerprint in self.prefix_groups:
                self.prefix_groups[key.prefix_fingerprint].remove(request_id)
                if not self.prefix_groups[key.prefix_fingerprint]:
                    del self.prefix_groups[key.prefix_fingerprint]
    
    def stats(self) -> Dict:
        """Get enhanced cache statistics."""
        base_stats = self.kv_pool.get_stats()
        
        # Calculate hit rate
        total_access = self.versioned_hits + self.versioned_misses
        hit_rate = self.versioned_hits / max(total_access, 1)
        
        return {
            **base_stats,
            "versioned_hits": self.versioned_hits,
            "versioned_misses": self.versioned_misses,
            "versioned_hit_rate": hit_rate,
            "unique_keys": len(self.key_mapping),
            "prefix_groups": len(self.prefix_groups),
            "prefix_reuse_count": self.prefix_reuse_count,
            "admission_policy": self.admission_policy,
            "eviction_policy": self.eviction_policy
        }
    
    def clear(self):
        """Clear all caches and mappings."""
        self.kv_pool.clear()
        self.key_mapping.clear()
        self.reverse_mapping.clear()
        self.prefix_groups.clear()
        
        # Reset metrics
        self.versioned_hits = 0
        self.versioned_misses = 0
        self.prefix_reuse_count = 0