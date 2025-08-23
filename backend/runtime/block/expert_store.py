"""
Expert Store Implementation

Manages expert loading, caching, and verification.
"""

import asyncio
import hashlib
import time
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import torch
import aiohttp
from cachetools import TTLCache, LRUCache
import logging

from .types import ExpertMetadata, ExpertData, FetchStrategy, CacheConfig
from .errors import (
    ExpertNotFoundError, 
    ExpertVerificationError,
    FetchTimeoutError,
    CacheError
)
from .blockchain_fetcher import BlockchainExpertFetcher

logger = logging.getLogger(__name__)


class ExpertStore:
    """
    Manages expert storage with CID verification and multi-level caching.
    """
    
    def __init__(
        self,
        cache_config: CacheConfig,
        manifest_path: Optional[Path] = None,
        peers: Optional[List[str]] = None,
        fetch_timeout_ms: int = 5000,
        enable_verification: bool = True,
        chain_b = None,
        fetch_strategy: FetchStrategy = FetchStrategy.STANDARD,
        max_concurrent_fetches: int = 3,
        hedged_delay_ms: int = 50
    ):
        self.cache_config = cache_config
        self.manifest_path = manifest_path or Path("./data/expert_manifest.json")
        self.peers = peers or ["http://localhost:8001", "http://localhost:8002"]
        self.fetch_timeout_ms = fetch_timeout_ms
        self.enable_verification = enable_verification
        self.chain_b = chain_b
        self.fetch_strategy = fetch_strategy
        self.max_concurrent_fetches = max_concurrent_fetches
        self.hedged_delay_ms = hedged_delay_ms
        
        # Initialize blockchain fetcher if chain provided
        self.blockchain_fetcher = BlockchainExpertFetcher(chain_b) if chain_b else None
        
        # Memory cache (hot experts)
        cache_size = int(cache_config.memory_cache_size_mb * 1024 * 1024 / (100 * 1024))  # Estimate 100KB per expert
        self.memory_cache: TTLCache = TTLCache(
            maxsize=cache_size,
            ttl=cache_config.ttl_seconds
        )
        
        # Disk cache path
        self.disk_cache_path = Path("./cache/experts")
        self.disk_cache_path.mkdir(parents=True, exist_ok=True)
        
        # LRU for disk cache tracking
        disk_cache_size = int(cache_config.disk_cache_size_mb * 1024 * 1024 / (100 * 1024))
        self.disk_cache_lru: LRUCache = LRUCache(maxsize=disk_cache_size)
        
        # Metrics
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "fetch_successes": 0,
            "fetch_failures": 0,
            "verification_successes": 0,
            "verification_failures": 0,
            "total_fetch_time_ms": 0
        }
        
        # In-flight fetches to prevent duplicate requests
        self.in_flight: Dict[str, asyncio.Future] = {}
    
    def _expert_key(self, layer_id: int, expert_id: int) -> str:
        """Generate cache key for an expert."""
        return f"L{layer_id}_E{expert_id}"
    
    async def get_expert(
        self, 
        layer_id: int, 
        expert_id: int,
        metadata: Optional[ExpertMetadata] = None
    ) -> ExpertData:
        """
        Get an expert, loading from cache or fetching if needed.
        """
        key = self._expert_key(layer_id, expert_id)
        start_time = time.time()
        
        # Check memory cache
        if key in self.memory_cache:
            self.metrics["cache_hits"] += 1
            cached_data = self.memory_cache[key]
            return ExpertData(
                metadata=cached_data["metadata"],
                weights=cached_data["weights"],
                verified=True,
                cache_hit=True,
                fetch_latency_ms=0
            )
        
        # Check if already being fetched
        if key in self.in_flight:
            future = self.in_flight[key]
            return await future
        
        # Create future for this fetch
        future = asyncio.Future()
        self.in_flight[key] = future
        
        try:
            # Check disk cache
            disk_data = await self._load_from_disk(key)
            if disk_data:
                self.metrics["cache_hits"] += 1
                # Promote to memory cache
                self.memory_cache[key] = disk_data
                result = ExpertData(
                    metadata=disk_data["metadata"],
                    weights=disk_data["weights"],
                    verified=True,
                    cache_hit=True,
                    fetch_latency_ms=(time.time() - start_time) * 1000
                )
                future.set_result(result)
                return result
            
            # Cache miss - fetch from network
            self.metrics["cache_misses"] += 1
            
            if not metadata:
                metadata = await self._lookup_metadata(layer_id, expert_id)
            
            expert_data = await self._fetch_from_network(metadata)
            
            # Verify
            if not await self._verify_expert(expert_data, metadata):
                raise ExpertVerificationError(
                    metadata.cid, 
                    "CID verification failed"
                )
            
            # Cache
            await self._cache_expert(key, expert_data, metadata)
            
            fetch_time_ms = (time.time() - start_time) * 1000
            self.metrics["total_fetch_time_ms"] += fetch_time_ms
            self.metrics["fetch_successes"] += 1
            
            result = ExpertData(
                metadata=metadata,
                weights=expert_data,
                verified=True,
                cache_hit=False,
                fetch_latency_ms=fetch_time_ms
            )
            
            future.set_result(result)
            return result
            
        except Exception as e:
            self.metrics["fetch_failures"] += 1
            future.set_exception(e)
            raise
        finally:
            # Clean up in-flight tracker
            if key in self.in_flight:
                del self.in_flight[key]
    
    async def _load_from_disk(self, key: str) -> Optional[Dict]:
        """Load expert from disk cache."""
        cache_file = self.disk_cache_path / f"{key}.pt"
        
        if not cache_file.exists():
            return None
        
        try:
            # Track in LRU
            self.disk_cache_lru[key] = time.time()
            
            # Load from disk
            data = torch.load(cache_file, map_location="cpu")
            return data
        except Exception as e:
            logger.error(f"Failed to load from disk cache: {e}")
            return None
    
    async def _lookup_metadata(self, layer_id: int, expert_id: int) -> ExpertMetadata:
        """Look up expert metadata from blockchain chain."""
        expert_name = f"layer{layer_id}.expert{expert_id}"
        
        # Try to get metadata from blockchain
        if self.chain_b:
            try:
                # Search for the expert block
                if hasattr(self.chain_b, 'get_blocks_by_type'):
                    expert_blocks = self.chain_b.get_blocks_by_type('expert')
                    for block in expert_blocks:
                        if hasattr(block.header, 'expert_name') and block.header.expert_name == expert_name:
                            # Found the expert, extract metadata
                            block_hash = block.compute_hash()
                            
                            return ExpertMetadata(
                                layer_id=layer_id,
                                expert_id=expert_id,
                                cid=f"Qm{block_hash[:44]}",  # Use block hash as CID
                                shard_id=getattr(block.header, 'shard_id', 'default'),
                                offset=getattr(block.header, 'offset', 0),
                                length=len(block.payload) if block.payload else 0,
                                merkle_root=block_hash,
                                compression=getattr(block.header, 'compression', None),
                                precision=getattr(block.header, 'precision', 'fp16')
                            )
            except Exception as e:
                logger.warning(f"Failed to lookup metadata from chain: {e}")
        
        # Try manifest file if available
        if self.manifest_path and self.manifest_path.exists():
            try:
                import json
                with open(self.manifest_path, 'r') as f:
                    manifest = json.load(f)
                    
                if expert_name in manifest:
                    entry = manifest[expert_name]
                    return ExpertMetadata(
                        layer_id=layer_id,
                        expert_id=expert_id,
                        cid=entry.get('cid', ''),
                        shard_id=entry.get('shard_id', 'default'),
                        offset=entry.get('offset', 0),
                        length=entry.get('length', 0),
                        merkle_root=entry.get('merkle_root', ''),
                        merkle_proof=entry.get('merkle_proof'),  # Load proof from manifest
                        compression=entry.get('compression'),
                        precision=entry.get('precision', 'fp16')
                    )
            except Exception as e:
                logger.debug(f"Manifest lookup failed: {e}")
        
        # Fallback: generate metadata from expert name or blockchain
        expert_hash = hashlib.sha256(expert_name.encode()).hexdigest()
        
        # Try to get merkle proof from blockchain if available
        merkle_proof = None
        if self.chain_b:
            try:
                # Look for expert block in chain
                expert_blocks = self.chain_b.get_blocks_by_type('expert')
                for block in expert_blocks:
                    if hasattr(block, 'header') and block.header.expert_name == expert_name:
                        # Extract merkle proof if available
                        if hasattr(block, 'merkle_proof'):
                            merkle_proof = block.merkle_proof
                        break
            except Exception as e:
                logger.debug(f"Failed to get merkle proof from chain: {e}")
        
        return ExpertMetadata(
            layer_id=layer_id,
            expert_id=expert_id,
            cid=expert_hash,  # Use raw hash instead of fake IPFS CID
            shard_id="computed",
            offset=0,
            length=0,  # Unknown
            merkle_root=expert_hash,
            merkle_proof=merkle_proof,  # Include proof if found
            compression=None,
            precision="fp16"
        )
    
    async def _fetch_from_network(self, metadata: ExpertMetadata) -> torch.Tensor:
        """Fetch expert from network peers or blockchain with hedged requests."""
        
        # First try blockchain fetcher if available
        if self.blockchain_fetcher:
            try:
                expert_data = await asyncio.to_thread(
                    self.blockchain_fetcher.fetch_expert,
                    metadata.layer_id,
                    metadata.expert_id
                )
                
                if expert_data:
                    # Return the first tensor (simplified for now)
                    for key, tensor in expert_data.items():
                        return tensor
            except Exception as e:
                logger.warning(f"Blockchain fetch failed: {e}")
        
        # Hedged fetch from network peers
        if self.fetch_strategy == FetchStrategy.HEDGED and len(self.peers) > 1:
            return await self._hedged_fetch_from_peers(metadata)
        else:
            return await self._standard_fetch_from_peers(metadata)
    
    async def _standard_fetch_from_peers(self, metadata: ExpertMetadata) -> torch.Tensor:
        """Standard sequential fetch from peers."""
        errors = []
        for peer in self.peers:
            try:
                result = await self._fetch_from_single_peer(peer, metadata)
                if result is not None:
                    return result
            except Exception as e:
                errors.append(f"Error fetching from {peer}: {e}")
        
        raise FetchTimeoutError(
            f"L{metadata.layer_id}_E{metadata.expert_id}",
            self.fetch_timeout_ms
        )
    
    async def _hedged_fetch_from_peers(self, metadata: ExpertMetadata) -> torch.Tensor:
        """Hedged parallel fetch from multiple peers with delay."""
        # Create tasks for hedged requests
        tasks = []
        errors = []
        
        async def fetch_with_delay(peer: str, delay_ms: int = 0):
            """Fetch from a peer with optional delay."""
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)
            try:
                return await self._fetch_from_single_peer(peer, metadata)
            except Exception as e:
                errors.append(f"Error from {peer}: {e}")
                raise
        
        # Start first request immediately
        primary_task = asyncio.create_task(fetch_with_delay(self.peers[0], 0))
        tasks.append(primary_task)
        
        # Start hedged requests with delay
        hedged_delay_ms = getattr(self, 'hedged_delay_ms', 50)  # Default 50ms
        for i, peer in enumerate(self.peers[1:], 1):
            if i >= getattr(self, 'max_concurrent_fetches', 3):
                break  # Limit concurrent fetches
            hedged_task = asyncio.create_task(
                fetch_with_delay(peer, hedged_delay_ms * i)
            )
            tasks.append(hedged_task)
        
        # Wait for first successful result
        try:
            # Use asyncio.as_completed to get results as they complete
            for completed_task in asyncio.as_completed(tasks):
                try:
                    result = await completed_task
                    if result is not None:
                        # Cancel remaining tasks
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        return result
                except Exception:
                    continue  # Try next task
        except Exception as e:
            logger.error(f"All hedged fetches failed: {errors}")
        
        # All tasks failed
        raise FetchTimeoutError(
            f"L{metadata.layer_id}_E{metadata.expert_id}",
            self.fetch_timeout_ms
        )
    
    async def _fetch_from_single_peer(self, peer: str, metadata: ExpertMetadata) -> Optional[torch.Tensor]:
        """Fetch expert from a single peer."""
        url = f"{peer}/experts/{metadata.cid}"
        
        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(
                total=self.fetch_timeout_ms / 1000
            )
            
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.read()
                    # Try to deserialize as pickle
                    try:
                        import pickle
                        expert_dict = pickle.loads(data)
                        # Return first tensor
                        for key, tensor in expert_dict.items():
                            if isinstance(tensor, torch.Tensor):
                                return tensor
                    except:
                        # Fallback to raw tensor
                        return torch.frombuffer(data, dtype=torch.float16).reshape(-1, 768)
        
        return None
    
    async def _verify_expert(
        self, 
        expert_data: torch.Tensor, 
        metadata: ExpertMetadata
    ) -> bool:
        """Verify expert data against CID and merkle root."""
        if not self.enable_verification:
            return True
        
        try:
            # Import hashing utilities from crypto module
            from backend.crypto.hashing import merkle_root, verify_merkle_proof
            import pickle
            
            # Serialize expert data for hashing
            if isinstance(expert_data, torch.Tensor):
                # Convert tensor to bytes for consistent hashing
                data_bytes = pickle.dumps(expert_data.cpu().numpy())
            elif isinstance(expert_data, dict):
                # Handle dict of tensors
                data_to_hash = {}
                for key, tensor in expert_data.items():
                    if isinstance(tensor, torch.Tensor):
                        data_to_hash[key] = tensor.cpu().numpy()
                    else:
                        data_to_hash[key] = tensor
                data_bytes = pickle.dumps(data_to_hash, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                data_bytes = pickle.dumps(expert_data)
            
            # Compute SHA256 hash
            computed_hash = hashlib.sha256(data_bytes).hexdigest()
            
            # Verify against merkle root if available
            if metadata.merkle_root:
                # If we have merkle proof data, verify it
                if hasattr(metadata, 'merkle_proof') and metadata.merkle_proof:
                    is_valid = verify_merkle_proof(
                        computed_hash,
                        metadata.merkle_proof,
                        metadata.merkle_root
                    )
                    if is_valid:
                        self.metrics["verification_successes"] += 1
                        logger.debug(f"Merkle proof verification successful for L{metadata.layer_id}_E{metadata.expert_id}")
                        return True
                    else:
                        self.metrics["verification_failures"] += 1
                        logger.warning(f"Merkle proof verification failed for L{metadata.layer_id}_E{metadata.expert_id}")
                        return False
                
                # Direct hash comparison with merkle root (single item case)
                if computed_hash == metadata.merkle_root:
                    self.metrics["verification_successes"] += 1
                    return True
            
            # Verify against CID if available
            if metadata.cid:
                # Import CID utilities for strict validation
                from backend.crypto.cid_utils import verify_cid, is_valid_cid_format
                
                # Check if CID is a raw hash (64 hex chars)
                if len(metadata.cid) == 64 and metadata.cid == computed_hash:
                    self.metrics["verification_successes"] += 1
                    logger.debug(f"Direct hash CID verification successful: {metadata.cid[:16]}...")
                    return True
                
                # Strict IPFS CID verification
                if metadata.cid.startswith("Qm") or metadata.cid.startswith("b"):
                    # Verify data against CID
                    if verify_cid(data_bytes, metadata.cid):
                        self.metrics["verification_successes"] += 1
                        logger.debug(f"IPFS CID verification successful: {metadata.cid}")
                        return True
                    else:
                        self.metrics["verification_failures"] += 1
                        logger.error(f"IPFS CID verification failed for {metadata.cid}")
                        return False
                
                # Check if it's at least a valid format
                if not is_valid_cid_format(metadata.cid):
                    self.metrics["verification_failures"] += 1
                    logger.error(f"Invalid CID format: {metadata.cid}")
                    return False
            
            # If no verification data available but verification is enabled
            if self.enable_verification:
                # Check if strict mode is enabled (default: true for production)
                import os
                strict_mode = os.environ.get('BLOCK_RUNTIME_STRICT_VERIFICATION', 'true').lower() == 'true'
                
                if strict_mode:
                    self.metrics["verification_failures"] += 1
                    logger.error(
                        f"No verification data (CID/merkle) for L{metadata.layer_id}_E{metadata.expert_id} "
                        f"in strict mode - rejecting"
                    )
                    return False
                else:
                    logger.warning(
                        f"No verification data (CID/merkle) for L{metadata.layer_id}_E{metadata.expert_id}, "
                        f"accepting expert data (non-strict mode)"
                    )
                    self.metrics["verification_successes"] += 1
                    return True
            
            # Verification disabled, accept
            self.metrics["verification_successes"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            self.metrics["verification_failures"] += 1
            return False
    
    async def _cache_expert(
        self, 
        key: str, 
        expert_data: torch.Tensor,
        metadata: ExpertMetadata
    ) -> None:
        """Cache expert in memory and disk."""
        cache_entry = {
            "metadata": metadata,
            "weights": expert_data
        }
        
        # Add to memory cache
        self.memory_cache[key] = cache_entry
        
        # Save to disk cache
        try:
            cache_file = self.disk_cache_path / f"{key}.pt"
            torch.save(cache_entry, cache_file)
            self.disk_cache_lru[key] = time.time()
            
            # Evict old entries if needed
            await self._evict_disk_cache()
            
        except Exception as e:
            logger.error(f"Failed to save to disk cache: {e}")
    
    async def _evict_disk_cache(self) -> None:
        """Evict old entries from disk cache if needed."""
        # Simple size-based eviction
        # In production, use more sophisticated policies
        max_entries = int(
            self.cache_config.disk_cache_size_mb * 1024 * 1024 / (100 * 1024)
        )
        
        if len(self.disk_cache_lru) > max_entries:
            # Remove oldest entries
            to_remove = len(self.disk_cache_lru) - max_entries
            for key, _ in list(self.disk_cache_lru.items())[:to_remove]:
                cache_file = self.disk_cache_path / f"{key}.pt"
                if cache_file.exists():
                    cache_file.unlink()
                del self.disk_cache_lru[key]
    
    async def prefetch_experts(
        self, 
        layer_experts: Dict[int, List[int]]
    ) -> None:
        """Prefetch experts for early layers."""
        tasks = []
        for layer_id, expert_ids in layer_experts.items():
            for expert_id in expert_ids:
                key = self._expert_key(layer_id, expert_id)
                if key not in self.memory_cache:
                    tasks.append(self.get_expert(layer_id, expert_id))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get store metrics."""
        total_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        hit_ratio = (
            self.metrics["cache_hits"] / total_requests 
            if total_requests > 0 else 0
        )
        
        avg_fetch_time = (
            self.metrics["total_fetch_time_ms"] / self.metrics["fetch_successes"]
            if self.metrics["fetch_successes"] > 0 else 0
        )
        
        return {
            "cache_hit_ratio": hit_ratio,
            "total_requests": total_requests,
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "fetch_successes": self.metrics["fetch_successes"],
            "fetch_failures": self.metrics["fetch_failures"],
            "avg_fetch_time_ms": avg_fetch_time,
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size": len(self.disk_cache_lru)
        }