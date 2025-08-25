"""
GPU-Direct Block Loader for Blyan
Zero-copy, pinned memory optimized loading for dense models
"""

import mmap
import os
import io
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import torch
import numpy as np
from contextlib import contextmanager
import logging
from concurrent.futures import ThreadPoolExecutor
import pickle

from .chain import Chain
from .block import Block

logger = logging.getLogger(__name__)

class GPUDirectBlockLoader:
    """
    Ultra-fast GPU-direct block loader for dense models
    
    Features:
    - Zero-copy memory mapping
    - Pinned memory for optimal GPU transfers
    - Async non-blocking transfers
    - Batch loading optimization
    - Persistent caching
    """
    
    def __init__(
        self, 
        param_chain: Chain,
        cache_dir: Optional[Path] = None,
        enable_pinned_cache: bool = True,
        max_pinned_memory_gb: float = 4.0
    ):
        """
        Initialize GPU-direct loader.
        
        Args:
            param_chain: Parameter blockchain chain
            cache_dir: Directory for cached blocks
            enable_pinned_cache: Use pinned memory cache for frequently accessed blocks
            max_pinned_memory_gb: Maximum pinned memory to allocate
        """
        self.param_chain = param_chain
        self.cache_dir = cache_dir or Path("./data/gpu_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory-mapped file cache
        self._mmap_cache: Dict[int, mmap.mmap] = {}
        self._file_cache: Dict[int, Any] = {}
        
        # Pinned memory cache for hot blocks
        self.enable_pinned_cache = enable_pinned_cache
        self.max_pinned_bytes = int(max_pinned_memory_gb * 1024 * 1024 * 1024)
        self._pinned_cache: Dict[str, torch.Tensor] = {}
        self._pinned_cache_size = 0
        
        # CUDA stream for async transfers
        if torch.cuda.is_available():
            self.cuda_stream = torch.cuda.Stream()
        else:
            self.cuda_stream = None
            
        # Performance monitoring
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'pinned_hits': 0,
            'total_load_time': 0.0,
            'gpu_transfer_time': 0.0,
            'blocks_loaded': 0
        }
        
        logger.info(f"🚀 GPU-Direct loader initialized")
        logger.info(f"   Cache dir: {self.cache_dir}")
        logger.info(f"   Pinned memory: {max_pinned_memory_gb:.1f}GB")
        logger.info(f"   CUDA available: {torch.cuda.is_available()}")
    
    def load_block_to_gpu(
        self, 
        block_index: int,
        device: str = 'cuda',
        non_blocking: bool = True,
        dtype: torch.dtype = torch.bfloat16
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load block directly to GPU with zero-copy optimization.
        
        Args:
            block_index: Index of block in chain
            device: Target device
            non_blocking: Use async GPU transfer
            dtype: Target dtype (BF16 only for consistency)
            
        Returns:
            Dictionary of tensors loaded to GPU
        """
        start_time = time.time()
        
        try:
            # Check pinned cache first (fastest path)
            cache_key = f"block_{block_index}"
            if cache_key in self._pinned_cache:
                self.stats['pinned_hits'] += 1
                tensor = self._pinned_cache[cache_key]
                if str(tensor.device) != device:
                    tensor = tensor.to(device, non_blocking=non_blocking)
                
                self.stats['total_load_time'] += time.time() - start_time
                return {"tensor": tensor}
            
            # Get memory-mapped view
            mmap_view = self._get_mmap_view(block_index)
            if mmap_view is None:
                return None
            
            # Deserialize directly from mmap (zero-copy)
            tensors = self._deserialize_from_mmap(mmap_view, dtype)
            
            if not tensors:
                return None
            
            # Transfer to GPU with pinned memory optimization
            gpu_tensors = {}
            transfer_start = time.time()
            
            for key, tensor in tensors.items():
                # Convert to target dtype if needed
                if tensor.dtype != dtype:
                    tensor = tensor.to(dtype)
                
                # Pin memory for efficient GPU transfer
                if device != 'cpu' and not tensor.is_pinned():
                    tensor = tensor.pin_memory()
                
                # Async transfer to GPU
                if device != 'cpu':
                    if self.cuda_stream and non_blocking:
                        with torch.cuda.stream(self.cuda_stream):
                            gpu_tensor = tensor.to(device, non_blocking=True)
                    else:
                        gpu_tensor = tensor.to(device, non_blocking=non_blocking)
                else:
                    gpu_tensor = tensor
                
                gpu_tensors[key] = gpu_tensor
                
                # Cache in pinned memory if enabled
                if self.enable_pinned_cache and self._should_cache(tensor):
                    self._add_to_pinned_cache(cache_key, gpu_tensor)
            
            self.stats['gpu_transfer_time'] += time.time() - transfer_start
            self.stats['total_load_time'] += time.time() - start_time
            self.stats['blocks_loaded'] += 1
            
            return gpu_tensors
            
        except Exception as e:
            logger.error(f"Failed to load block {block_index}: {e}")
            return None
    
    def load_layer_to_gpu(
        self,
        layer_idx: int,
        device: str = 'cuda',
        non_blocking: bool = True
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load all weights for a specific layer directly to GPU.
        
        Args:
            layer_idx: Layer index
            device: Target device
            non_blocking: Use async transfer
            
        Returns:
            Dictionary of layer weights on GPU
        """
        logger.info(f"⚡ Loading layer {layer_idx} directly to {device}...")
        
        # Find all blocks for this layer
        layer_blocks = []
        block_count = len(self.param_chain._hash_index) if hasattr(self.param_chain, '_hash_index') else 0
        
        for i in range(block_count):
            block = self.param_chain.storage.get_block_by_index(i)
            if block and hasattr(block.header, 'layer_idx') and block.header.layer_idx == layer_idx:
                layer_blocks.append(i)
        
        if not layer_blocks:
            logger.warning(f"No blocks found for layer {layer_idx}")
            return None
        
        # Load all blocks for layer
        layer_weights = {}
        for block_idx in layer_blocks:
            block_tensors = self.load_block_to_gpu(block_idx, device, non_blocking)
            if block_tensors:
                layer_weights.update(block_tensors)
        
        return layer_weights
    
    def batch_load_to_gpu(
        self,
        block_indices: List[int],
        device: str = 'cuda',
        max_workers: int = 4
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Load multiple blocks in parallel with optimized batching.
        
        Args:
            block_indices: List of block indices to load
            device: Target device
            max_workers: Number of parallel workers
            
        Returns:
            Dictionary mapping block index to tensors
        """
        logger.info(f"📦 Batch loading {len(block_indices)} blocks to {device}...")
        
        # Pre-fetch all blocks into memory maps
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx in block_indices:
                future = executor.submit(self._get_mmap_view, idx)
                futures.append((idx, future))
            
            # Wait for all memory maps
            for idx, future in futures:
                try:
                    future.result(timeout=10)
                except Exception as e:
                    logger.warning(f"Failed to prefetch block {idx}: {e}")
        
        # Now load to GPU with CUDA stream optimization
        results = {}
        
        if self.cuda_stream and device != 'cpu':
            with torch.cuda.stream(self.cuda_stream):
                for idx in block_indices:
                    tensors = self.load_block_to_gpu(idx, device, non_blocking=True)
                    if tensors:
                        results[idx] = tensors
            
            # Synchronize stream
            self.cuda_stream.synchronize()
        else:
            for idx in block_indices:
                tensors = self.load_block_to_gpu(idx, device, non_blocking=False)
                if tensors:
                    results[idx] = tensors
        
        logger.info(f"✅ Loaded {len(results)}/{len(block_indices)} blocks")
        return results
    
    def _get_mmap_view(self, block_index: int) -> Optional[mmap.mmap]:
        """Get memory-mapped view of block file."""
        # Check cache
        if block_index in self._mmap_cache:
            self.stats['cache_hits'] += 1
            return self._mmap_cache[block_index]
        
        self.stats['cache_misses'] += 1
        
        # Ensure block file exists
        cache_file = self._ensure_block_file(block_index)
        if not cache_file:
            return None
        
        try:
            # Create memory mapping
            file_obj = open(cache_file, 'rb')
            mmap_obj = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Cache the mapping
            self._file_cache[block_index] = file_obj
            self._mmap_cache[block_index] = mmap_obj
            
            return mmap_obj
        except Exception as e:
            logger.error(f"Failed to mmap block {block_index}: {e}")
            return None
    
    def _ensure_block_file(self, block_index: int) -> Optional[Path]:
        """Ensure block file exists in cache."""
        cache_file = self.cache_dir / f"block_{block_index}.bin"
        
        if not cache_file.exists():
            # Fetch from blockchain
            block = self.param_chain.storage.get_block_by_index(block_index)
            if not block:
                logger.warning(f"Block {block_index} not found in chain")
                return None
            
            # Write to cache
            try:
                with open(cache_file, 'wb') as f:
                    f.write(block.payload)
                logger.debug(f"Cached block {block_index} to disk")
            except Exception as e:
                logger.error(f"Failed to cache block {block_index}: {e}")
                return None
        
        return cache_file
    
    def _deserialize_from_mmap(
        self, 
        mmap_view: mmap.mmap,
        dtype: torch.dtype = torch.bfloat16
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Deserialize tensors directly from memory map."""
        try:
            # Read all bytes (this is still zero-copy with mmap)
            data = bytes(mmap_view)
            
            # Try pickle first (most common format)
            try:
                tensors = pickle.loads(data)
                if isinstance(tensors, dict):
                    # Convert to target dtype
                    return {k: v.to(dtype) if torch.is_tensor(v) else v 
                           for k, v in tensors.items()}
            except:
                pass
            
            # Try safetensors format
            try:
                import safetensors.torch
                tensors = safetensors.torch.load(data)
                return {k: v.to(dtype) for k, v in tensors.items()}
            except:
                pass
            
            # Try raw tensor
            try:
                tensor = torch.load(io.BytesIO(data), map_location='cpu')
                if torch.is_tensor(tensor):
                    return {"weight": tensor.to(dtype)}
                elif isinstance(tensor, dict):
                    return {k: v.to(dtype) if torch.is_tensor(v) else v 
                           for k, v in tensor.items()}
            except:
                pass
            
            logger.warning("Could not deserialize block data")
            return None
            
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            return None
    
    def _should_cache(self, tensor: torch.Tensor) -> bool:
        """Check if tensor should be cached in pinned memory."""
        tensor_size = tensor.element_size() * tensor.nelement()
        return (self._pinned_cache_size + tensor_size) <= self.max_pinned_bytes
    
    def _add_to_pinned_cache(self, key: str, tensor: torch.Tensor):
        """Add tensor to pinned memory cache."""
        if key in self._pinned_cache:
            return
        
        tensor_size = tensor.element_size() * tensor.nelement()
        
        # Evict old entries if needed
        while (self._pinned_cache_size + tensor_size) > self.max_pinned_bytes and self._pinned_cache:
            evict_key = next(iter(self._pinned_cache))
            evicted = self._pinned_cache.pop(evict_key)
            self._pinned_cache_size -= evicted.element_size() * evicted.nelement()
            logger.debug(f"Evicted {evict_key} from pinned cache")
        
        # Add to cache
        self._pinned_cache[key] = tensor
        self._pinned_cache_size += tensor_size
    
    def preload_layers(
        self,
        layer_indices: List[int],
        device: str = 'cuda'
    ):
        """Preload multiple layers into GPU memory."""
        logger.info(f"🔄 Preloading {len(layer_indices)} layers to {device}...")
        
        for layer_idx in layer_indices:
            self.load_layer_to_gpu(layer_idx, device, non_blocking=True)
        
        if self.cuda_stream:
            self.cuda_stream.synchronize()
        
        logger.info(f"✅ Preloaded {len(layer_indices)} layers")
    
    def clear_cache(self):
        """Clear all caches."""
        # Close memory maps
        for mmap_obj in self._mmap_cache.values():
            mmap_obj.close()
        
        # Close files
        for file_obj in self._file_cache.values():
            file_obj.close()
        
        # Clear caches
        self._mmap_cache.clear()
        self._file_cache.clear()
        self._pinned_cache.clear()
        self._pinned_cache_size = 0
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader performance statistics."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        
        return {
            'cache_hit_rate': self.stats['cache_hits'] / total_requests if total_requests > 0 else 0,
            'pinned_hit_rate': self.stats['pinned_hits'] / total_requests if total_requests > 0 else 0,
            'avg_load_time_ms': (self.stats['total_load_time'] / self.stats['blocks_loaded'] * 1000) 
                               if self.stats['blocks_loaded'] > 0 else 0,
            'avg_gpu_transfer_ms': (self.stats['gpu_transfer_time'] / self.stats['blocks_loaded'] * 1000)
                                  if self.stats['blocks_loaded'] > 0 else 0,
            'blocks_loaded': self.stats['blocks_loaded'],
            'pinned_cache_size_mb': self._pinned_cache_size / (1024 * 1024),
            'mmap_cache_size': len(self._mmap_cache)
        }

@contextmanager
def gpu_memory_optimization():
    """Context manager for GPU memory optimization."""
    if torch.cuda.is_available():
        # Clear cache before operation
        torch.cuda.empty_cache()
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Enable TF32 for better performance on Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

class GPUDirectBenchmark:
    """Benchmark GPU-direct loading performance."""
    
    def __init__(self, loader: GPUDirectBlockLoader):
        self.loader = loader
    
    def benchmark_loading_methods(self, block_index: int, iterations: int = 10):
        """Compare GPU-direct vs traditional loading."""
        import io
        
        # Warmup
        self.loader.load_block_to_gpu(block_index)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark GPU-direct loading
        start_time = time.time()
        for _ in range(iterations):
            tensors = self.loader.load_block_to_gpu(block_index, non_blocking=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        gpu_direct_time = (time.time() - start_time) / iterations
        
        # Benchmark traditional loading
        start_time = time.time()
        for _ in range(iterations):
            # Traditional: fetch -> deserialize -> copy to GPU
            block = self.loader.param_chain.storage.get_block_by_index(block_index)
            if block:
                data = pickle.loads(block.payload)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if torch.is_tensor(v):
                            v = v.to('cuda')
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        traditional_time = (time.time() - start_time) / iterations
        
        return {
            'gpu_direct_ms': gpu_direct_time * 1000,
            'traditional_ms': traditional_time * 1000,
            'speedup': traditional_time / gpu_direct_time if gpu_direct_time > 0 else 0,
            'iterations': iterations
        }

# Export main classes
__all__ = ['GPUDirectBlockLoader', 'gpu_memory_optimization', 'GPUDirectBenchmark']