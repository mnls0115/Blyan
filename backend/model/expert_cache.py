"""
Expert LRU Cache with CUDA memory management.
Prevents OOM by properly managing GPU memory with eviction and cleanup.
"""
import torch
import gc
import psutil
import logging
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple
from threading import Lock
import time

logger = logging.getLogger(__name__)


class ExpertLRUCache:
    """
    LRU cache for Expert weights with proper CUDA memory management.
    
    Features:
    - Size-based eviction (memory limit)
    - Proper CUDA memory cleanup
    - Thread-safe operations
    - Memory pressure monitoring
    - Cache hit/miss statistics
    """
    
    def __init__(self, max_memory_gb: float = 8.0, device: str = 'cuda'):
        """
        Initialize Expert LRU cache.
        
        Args:
            max_memory_gb: Maximum GPU memory to use for caching (in GB)
            device: Target device ('cuda' or 'cpu')
        """
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.device = device
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.lock = Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_load_time = 0.0
        
        # Memory tracking
        self.current_memory_usage = 0
        
        logger.info(f"Initialized ExpertLRUCache with {max_memory_gb:.1f}GB limit on {device}")
    
    def _get_tensor_memory_size(self, tensor: torch.Tensor) -> int:
        """Calculate memory size of a tensor in bytes."""
        return tensor.element_size() * tensor.nelement()
    
    def _get_expert_memory_size(self, expert_data: Dict[str, Any]) -> int:
        """Calculate total memory size of an expert."""
        total_size = 0
        for key, value in expert_data.items():
            if isinstance(value, torch.Tensor):
                total_size += self._get_tensor_memory_size(value)
            elif isinstance(value, dict):
                # Recursively handle nested dictionaries
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        total_size += self._get_tensor_memory_size(v)
        return total_size
    
    def _evict_lru(self) -> Optional[str]:
        """
        Evict least recently used expert and properly clean up CUDA memory.
        
        Returns:
            Expert key that was evicted, or None if cache is empty
        """
        if not self.cache:
            return None
        
        # Get LRU expert (first item in OrderedDict)
        expert_key, expert_data = self.cache.popitem(last=False)
        expert_size = expert_data['size']
        
        # Properly delete tensors and free CUDA memory
        self._cleanup_expert(expert_data['data'])
        
        self.current_memory_usage -= expert_size
        self.evictions += 1
        
        logger.debug(f"Evicted expert {expert_key}, freed {expert_size / 1024 / 1024:.1f}MB")
        
        return expert_key
    
    def _cleanup_expert(self, expert_data: Dict[str, Any]):
        """
        Properly clean up expert tensors and free CUDA memory.
        
        CRITICAL: This ensures CUDA memory is actually freed, not just Python references.
        """
        # Delete all tensor references
        for key, value in list(expert_data.items()):
            if isinstance(value, torch.Tensor):
                del expert_data[key]
            elif isinstance(value, dict):
                for k, v in list(value.items()):
                    if isinstance(v, torch.Tensor):
                        del value[k]
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache to actually free memory
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations complete
    
    def _make_room_for(self, required_size: int) -> bool:
        """
        Evict experts until there's room for the required size.
        
        Returns:
            True if enough room was made, False otherwise
        """
        available_space = self.max_memory_bytes - self.current_memory_usage
        
        while available_space < required_size and self.cache:
            self._evict_lru()
            available_space = self.max_memory_bytes - self.current_memory_usage
        
        return available_space >= required_size
    
    def get(self, expert_key: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get expert from cache, updating LRU order.
        
        Args:
            expert_key: Unique identifier for the expert
            
        Returns:
            Expert state dict or None if not in cache
        """
        with self.lock:
            if expert_key not in self.cache:
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(expert_key)
            self.hits += 1
            
            return self.cache[expert_key]['data']
    
    def put(self, expert_key: str, expert_data: Dict[str, torch.Tensor]) -> bool:
        """
        Add expert to cache, evicting LRU if necessary.
        
        Args:
            expert_key: Unique identifier for the expert
            expert_data: Expert state dict with tensors
            
        Returns:
            True if successfully cached, False if expert is too large
        """
        with self.lock:
            # Calculate expert size
            expert_size = self._get_expert_memory_size(expert_data)
            
            # Check if expert is too large for cache
            if expert_size > self.max_memory_bytes:
                logger.warning(f"Expert {expert_key} ({expert_size / 1024 / 1024:.1f}MB) "
                             f"exceeds cache limit ({self.max_memory_bytes / 1024 / 1024:.1f}MB)")
                return False
            
            # If expert already in cache, remove old version
            if expert_key in self.cache:
                old_size = self.cache[expert_key]['size']
                self.current_memory_usage -= old_size
                self._cleanup_expert(self.cache[expert_key]['data'])
            
            # Make room for new expert
            if not self._make_room_for(expert_size):
                return False
            
            # Add to cache
            self.cache[expert_key] = {
                'data': expert_data,
                'size': expert_size,
                'timestamp': time.time()
            }
            self.cache.move_to_end(expert_key)  # Mark as most recently used
            self.current_memory_usage += expert_size
            
            logger.debug(f"Cached expert {expert_key} ({expert_size / 1024 / 1024:.1f}MB)")
            
            return True
    
    def clear(self):
        """Clear entire cache and free all GPU memory."""
        with self.lock:
            for expert_data in self.cache.values():
                self._cleanup_expert(expert_data['data'])
            
            self.cache.clear()
            self.current_memory_usage = 0
            
            # Final cleanup
            gc.collect()
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate,
                'current_size': len(self.cache),
                'memory_usage_mb': self.current_memory_usage / 1024 / 1024,
                'memory_limit_mb': self.max_memory_bytes / 1024 / 1024,
                'memory_utilization': self.current_memory_usage / self.max_memory_bytes
            }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory information."""
        info = {
            'cache_memory_mb': self.current_memory_usage / 1024 / 1024,
            'cache_limit_mb': self.max_memory_bytes / 1024 / 1024,
        }
        
        if self.device == 'cuda' and torch.cuda.is_available():
            # GPU memory info
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_reserved = torch.cuda.memory_reserved(0)
            
            info.update({
                'gpu_total_mb': gpu_mem / 1024 / 1024,
                'gpu_allocated_mb': gpu_allocated / 1024 / 1024,
                'gpu_reserved_mb': gpu_reserved / 1024 / 1024,
                'gpu_free_mb': (gpu_mem - gpu_reserved) / 1024 / 1024
            })
        else:
            # CPU memory info
            vm = psutil.virtual_memory()
            info.update({
                'cpu_total_mb': vm.total / 1024 / 1024,
                'cpu_available_mb': vm.available / 1024 / 1024,
                'cpu_used_percent': vm.percent
            })
        
        return info
    
    def __del__(self):
        """Ensure proper cleanup on deletion."""
        self.clear()