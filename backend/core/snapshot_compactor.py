"""
Snapshot Compactor for Blyan Tile-Based Learning
Automatic delta chain compression and periodic snapshots
"""

import asyncio
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import threading
import torch

from .tile_blockchain_integration import TileBlockchainManager
from .delta_compression import DeltaBase, DeltaCompressor
from .tile_block import TileBlock

@dataclass
class SnapshotMetrics:
    """Metrics for snapshot compaction performance"""
    tiles_compacted: int = 0
    deltas_compressed: int = 0
    space_saved_mb: float = 0.0
    compression_time_ms: float = 0.0
    snapshot_size_mb: float = 0.0
    original_chain_length: int = 0
    compressed_chain_length: int = 0
    generation_id: int = 0
    epoch_boundary: bool = False

@dataclass
class GenerationMetadata:
    """Metadata for a generation/epoch boundary."""
    generation_id: int
    start_block_hash: str
    end_block_hash: str
    snapshot_hash: str
    creation_timestamp: float
    total_deltas: int
    performance_improvement: float
    validation_score: float
    epoch_summary: Dict[str, Any]

class SnapshotCompactor:
    """
    Automatic snapshot compaction system
    
    Features:
    - Periodic delta chain compression
    - Smart compaction triggers
    - Space-efficient snapshots
    - Background processing
    - Configurable retention policies
    """
    
    def __init__(self, 
                 blockchain_manager: TileBlockchainManager,
                 snapshot_dir: Optional[Path] = None,
                 compaction_interval: float = 3600.0,  # 1 hour
                 min_deltas_for_compaction: int = 10,
                 max_chain_length: int = 100):
        
        self.blockchain_manager = blockchain_manager
        self.snapshot_dir = snapshot_dir or Path("./data/snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Compaction configuration
        self.compaction_interval = compaction_interval
        self.min_deltas_for_compaction = min_deltas_for_compaction
        self.max_chain_length = max_chain_length
        
        # Background processing
        self.compaction_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Compaction metrics
        self.metrics = SnapshotMetrics()
        self.compaction_history: List[Dict] = []
        
        # Generation management
        self.current_generation_id: int = 0
        self.generation_metadata: Dict[int, GenerationMetadata] = {}
        self.generation_boundaries: Dict[str, int] = {}  # tile_id -> generation_id
        self.epoch_thresholds = {
            'min_deltas_per_epoch': 50,
            'min_time_per_epoch': 3600.0,  # 1 hour
            'performance_threshold': 0.05,  # 5% improvement
            'validation_threshold': 0.9     # 90% validation success
        }
        
        # Delta compressor for snapshot optimization
        self.delta_compressor = DeltaCompressor(
            int8_enabled=True,
            sparse_enabled=True,
            lora_enabled=True,
            sparsity_threshold=0.9  # Very aggressive compression for snapshots
        )
        
        print(f"📦 SnapshotCompactor initialized")
        print(f"   Interval: {compaction_interval}s")
        print(f"   Min deltas: {min_deltas_for_compaction}")
        print(f"   Max chain length: {max_chain_length}")
    
    def start_background_compaction(self):
        """Start background compaction thread"""
        if self.running:
            return
        
        self.running = True
        self.compaction_thread = threading.Thread(target=self._compaction_loop)
        self.compaction_thread.daemon = True
        self.compaction_thread.start()
        
        print("🔄 Background compaction started")
    
    def stop_background_compaction(self):
        """Stop background compaction"""
        self.running = False
        if self.compaction_thread:
            self.compaction_thread.join(timeout=10.0)
        
        print("⏹️ Background compaction stopped")
    
    def _compaction_loop(self):
        """Main compaction loop"""
        while self.running:
            try:
                # Check all tiles for compaction opportunities
                tiles_to_compact = self._identify_compaction_candidates()
                
                if tiles_to_compact:
                    print(f"🔍 Found {len(tiles_to_compact)} tiles needing compaction")
                    
                    for tile_id in tiles_to_compact:
                        if not self.running:
                            break
                        
                        try:
                            self.compact_tile_chain(tile_id)
                        except Exception as e:
                            print(f"❌ Failed to compact tile {tile_id}: {e}")
                
                # Sleep until next compaction cycle
                time.sleep(self.compaction_interval)
                
            except Exception as e:
                print(f"💥 Error in compaction loop: {e}")
                time.sleep(60)  # Brief pause on error
    
    def _identify_compaction_candidates(self) -> List[str]:
        """Identify tiles that need compaction"""
        candidates = []
        
        # Check each tile's version history
        for tile_id, versions in self.blockchain_manager.tile_versions.items():
            # Skip if not enough versions
            if len(versions) < self.min_deltas_for_compaction:
                continue
            
            # Check if chain is too long
            if len(versions) > self.max_chain_length:
                candidates.append(tile_id)
                continue
            
            # Check time since last compaction
            metadata = self.blockchain_manager.tile_metadata.get(tile_id, {})
            last_compaction = metadata.get('last_compaction', 0)
            
            if time.time() - last_compaction > self.compaction_interval:
                candidates.append(tile_id)
        
        return candidates
    
    def compact_tile_chain(self, tile_id: str, 
                          snapshot_interval: int = 10) -> SnapshotMetrics:
        """
        Compact a tile's version chain by creating periodic snapshots
        
        Args:
            tile_id: Tile to compact
            snapshot_interval: Create snapshot every N versions
            
        Returns:
            Compaction metrics
        """
        start_time = time.time()
        
        try:
            if tile_id not in self.blockchain_manager.tile_versions:
                raise ValueError(f"Tile {tile_id} not found")
            
            versions = self.blockchain_manager.tile_versions[tile_id]
            if len(versions) < self.min_deltas_for_compaction:
                print(f"⚠️ Tile {tile_id} has only {len(versions)} versions, skipping compaction")
                return SnapshotMetrics()
            
            print(f"📦 Compacting tile {tile_id} ({len(versions)} versions)")
            
            # Calculate original size
            original_size = self._calculate_chain_size(tile_id)
            
            # Create snapshots at regular intervals
            snapshots_created = 0
            deltas_removed = 0
            
            # Keep track of which versions to snapshot
            snapshot_points = list(range(0, len(versions), snapshot_interval))
            if (len(versions) - 1) not in snapshot_points:
                snapshot_points.append(len(versions) - 1)  # Always keep latest
            
            # Create snapshots
            new_versions = []
            for snapshot_idx in snapshot_points:
                if snapshot_idx >= len(versions):
                    continue
                
                block_hash = versions[snapshot_idx]
                snapshot_hash = self._create_snapshot(tile_id, block_hash, snapshot_idx)
                
                if snapshot_hash:
                    new_versions.append(snapshot_hash)
                    snapshots_created += 1
                else:
                    # Keep original if snapshot failed
                    new_versions.append(block_hash)
            
            # Calculate deltas removed
            deltas_removed = len(versions) - len(new_versions)
            
            # Update tile versions (keep only snapshots)
            self.blockchain_manager.tile_versions[tile_id] = new_versions
            
            # Calculate new size
            compressed_size = self._calculate_chain_size(tile_id)
            space_saved = (original_size - compressed_size) / (1024 * 1024)  # MB
            
            # Update metadata
            self.blockchain_manager.tile_metadata[tile_id]['last_compaction'] = time.time()
            self.blockchain_manager.tile_metadata[tile_id]['compaction_count'] = \
                self.blockchain_manager.tile_metadata[tile_id].get('compaction_count', 0) + 1
            
            # Create metrics
            compaction_time = (time.time() - start_time) * 1000  # ms
            metrics = SnapshotMetrics(
                tiles_compacted=1,
                deltas_compressed=deltas_removed,
                space_saved_mb=space_saved,
                compression_time_ms=compaction_time,
                snapshot_size_mb=compressed_size / (1024 * 1024),
                original_chain_length=len(versions),
                compressed_chain_length=len(new_versions)
            )
            
            # Update global metrics
            self.metrics.tiles_compacted += 1
            self.metrics.deltas_compressed += deltas_removed
            self.metrics.space_saved_mb += space_saved
            
            # Record compaction
            self.compaction_history.append({
                'tile_id': tile_id,
                'timestamp': time.time(),
                'original_length': len(versions),
                'compressed_length': len(new_versions),
                'space_saved_mb': space_saved,
                'snapshots_created': snapshots_created
            })
            
            print(f"✅ Compacted {tile_id}: {len(versions)} → {len(new_versions)} versions")
            print(f"   Space saved: {space_saved:.1f} MB")
            print(f"   Snapshots created: {snapshots_created}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Failed to compact tile {tile_id}: {e}")
            raise
    
    def _create_snapshot(self, tile_id: str, block_hash: str, 
                        version_idx: int) -> Optional[str]:
        """
        Create a compressed snapshot of a tile version
        
        Returns:
            Block hash of snapshot, or None if failed
        """
        try:
            # Load tile tensor
            tensor = self.blockchain_manager.zero_copy_loader.load_tile(block_hash, device='cpu')
            
            # Create optimized tile with compression
            compressed_tile = self._create_compressed_tile(tile_id, tensor, version_idx)
            
            # Store snapshot on blockchain
            metadata = self.blockchain_manager.tile_metadata.get(tile_id, {})
            snapshot_hash = self.blockchain_manager.create_tile_block(
                compressed_tile,
                expert_name=f"{metadata.get('expert_name', '')}_snapshot",
                layer_id=metadata.get('layer_id', '')
            )
            
            # Save snapshot metadata
            snapshot_metadata = {
                'original_block': block_hash,
                'version_index': version_idx,
                'compression_method': 'snapshot_optimized',
                'created_at': time.time(),
                'tile_id': tile_id
            }
            
            snapshot_file = self.snapshot_dir / f"{tile_id}_v{version_idx}_{snapshot_hash[:8]}.json"
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot_metadata, f, indent=2)
            
            return snapshot_hash
            
        except Exception as e:
            print(f"❌ Failed to create snapshot for {tile_id} version {version_idx}: {e}")
            return None
    
    def _create_compressed_tile(self, tile_id: str, tensor: torch.Tensor, 
                              version_idx: int) -> TileBlock:
        """Create a space-optimized tile"""
        # Apply aggressive compression if beneficial
        if tensor.dtype == torch.float32:
            # Convert to fp16 for better compression
            tensor = tensor.to(torch.float16)
        
        # Create tile with smaller subtile size for better granularity
        compressed_tile = TileBlock(
            tile_id=f"{tile_id}_snapshot_v{version_idx}",
            tensor_data=tensor,
            subtile_size=128 * 1024  # 128KB subtiles for better compression
        )
        
        return compressed_tile
    
    def _calculate_chain_size(self, tile_id: str) -> int:
        """Calculate total size of tile chain in bytes"""
        if tile_id not in self.blockchain_manager.tile_versions:
            return 0
        
        total_size = 0
        for block_hash in self.blockchain_manager.tile_versions[tile_id]:
            block = self.blockchain_manager.tile_chain.get_block_by_hash(block_hash)
            if block:
                total_size += block.header.payload_size
        
        return total_size
    
    def compact_all_tiles(self, max_concurrent: int = 5) -> Dict[str, SnapshotMetrics]:
        """Compact all tiles that need compaction"""
        candidates = self._identify_compaction_candidates()
        
        if not candidates:
            print("✅ No tiles need compaction")
            return {}
        
        print(f"📦 Compacting {len(candidates)} tiles...")
        
        results = {}
        
        # Process in batches to limit resource usage
        for i in range(0, len(candidates), max_concurrent):
            batch = candidates[i:i + max_concurrent]
            
            # Process batch
            for tile_id in batch:
                if not self.running:
                    break
                
                try:
                    metrics = self.compact_tile_chain(tile_id)
                    results[tile_id] = metrics
                except Exception as e:
                    print(f"❌ Failed to compact {tile_id}: {e}")
                    results[tile_id] = SnapshotMetrics()
        
        return results
    
    def get_compaction_stats(self) -> Dict[str, Any]:
        """Get compaction performance statistics"""
        total_space_saved = sum(entry['space_saved_mb'] for entry in self.compaction_history)
        
        return {
            'total_compactions': len(self.compaction_history),
            'tiles_compacted': self.metrics.tiles_compacted,
            'deltas_compressed': self.metrics.deltas_compressed,
            'total_space_saved_mb': total_space_saved,
            'avg_compression_ratio': (
                self.metrics.space_saved_mb / max(1, self.metrics.snapshot_size_mb)
            ),
            'running': self.running,
            'last_compaction': (
                max(entry['timestamp'] for entry in self.compaction_history)
                if self.compaction_history else 0
            ),
            'recent_compactions': self.compaction_history[-10:]  # Last 10
        }
    
    def force_compaction(self, tile_id: str) -> SnapshotMetrics:
        """Force immediate compaction of a specific tile"""
        print(f"🔧 Force compacting tile {tile_id}")
        return self.compact_tile_chain(tile_id)
    
    def should_create_generation_boundary(self, tile_id: str) -> bool:
        """Determine if it's time to create a new generation boundary."""
        if tile_id not in self.blockchain_manager.tile_versions:
            return False
        
        versions = self.blockchain_manager.tile_versions[tile_id]
        current_generation = self.generation_boundaries.get(tile_id, 0)
        
        # Get last generation boundary
        last_boundary_idx = 0
        for i, version_hash in enumerate(versions):
            block = self.blockchain_manager.tile_chain.get_block_by_hash(version_hash)
            if block and hasattr(block.header, 'evolution_metadata'):
                metadata = block.header.evolution_metadata or {}
                if metadata.get('generation_id') == current_generation:
                    last_boundary_idx = i
                    break
        
        # Count deltas since last boundary
        deltas_since_boundary = len(versions) - last_boundary_idx - 1
        
        # Time since last boundary
        if versions:
            latest_block = self.blockchain_manager.tile_chain.get_block_by_hash(versions[-1])
            first_block = self.blockchain_manager.tile_chain.get_block_by_hash(versions[last_boundary_idx])
            
            if latest_block and first_block:
                time_since_boundary = latest_block.header.timestamp - first_block.header.timestamp
            else:
                time_since_boundary = 0
        else:
            time_since_boundary = 0
        
        # Check thresholds
        should_create = (
            deltas_since_boundary >= self.epoch_thresholds['min_deltas_per_epoch'] or
            time_since_boundary >= self.epoch_thresholds['min_time_per_epoch']
        )
        
        return should_create
    
    def create_generation_boundary(self, tile_id: str, 
                                 performance_improvement: float = 0.0,
                                 validation_score: float = 0.0) -> int:
        """Create a new generation boundary for a tile."""
        if tile_id not in self.blockchain_manager.tile_versions:
            raise ValueError(f"Tile {tile_id} not found")
        
        versions = self.blockchain_manager.tile_versions[tile_id]
        if not versions:
            raise ValueError(f"No versions found for tile {tile_id}")
        
        # Increment generation ID
        new_generation_id = self.current_generation_id + 1
        self.current_generation_id = new_generation_id
        
        # Get boundary details
        current_generation_start = self.generation_boundaries.get(tile_id, 0)
        start_idx = 0
        
        for i, version_hash in enumerate(versions):
            block = self.blockchain_manager.tile_chain.get_block_by_hash(version_hash)
            if block and hasattr(block.header, 'evolution_metadata'):
                metadata = block.header.evolution_metadata or {}
                if metadata.get('generation_id') == current_generation_start:
                    start_idx = i
                    break
        
        start_block_hash = versions[start_idx] if start_idx < len(versions) else versions[0]
        end_block_hash = versions[-1]
        
        # Create generation snapshot
        snapshot_hash = self._create_generation_snapshot(tile_id, new_generation_id)
        
        # Create generation metadata
        generation_metadata = GenerationMetadata(
            generation_id=new_generation_id,
            start_block_hash=start_block_hash,
            end_block_hash=end_block_hash,
            snapshot_hash=snapshot_hash,
            creation_timestamp=time.time(),
            total_deltas=len(versions) - start_idx,
            performance_improvement=performance_improvement,
            validation_score=validation_score,
            epoch_summary={
                'tile_id': tile_id,
                'start_generation': current_generation_start,
                'versions_processed': len(versions) - start_idx,
                'creation_method': 'automatic_threshold'
            }
        )
        
        # Store metadata
        self.generation_metadata[new_generation_id] = generation_metadata
        self.generation_boundaries[tile_id] = new_generation_id
        
        # Update blockchain metadata
        self._update_blockchain_generation_metadata(tile_id, new_generation_id)
        
        print(f"🧬 Created generation {new_generation_id} for tile {tile_id}")
        print(f"   Deltas processed: {generation_metadata.total_deltas}")
        print(f"   Performance improvement: {performance_improvement:.3f}")
        print(f"   Validation score: {validation_score:.3f}")
        
        return new_generation_id
    
    def _create_generation_snapshot(self, tile_id: str, generation_id: int) -> str:
        """Create a special generation snapshot."""
        if tile_id not in self.blockchain_manager.tile_versions:
            raise ValueError(f"Tile {tile_id} not found")
        
        versions = self.blockchain_manager.tile_versions[tile_id]
        latest_version_hash = versions[-1]
        
        # Load latest tile
        latest_tensor = self.blockchain_manager.zero_copy_loader.load_tile(latest_version_hash, device='cpu')
        
        # Create generation snapshot tile
        generation_tile = TileBlock(
            tile_id=f"{tile_id}_gen_{generation_id}",
            tensor_data=latest_tensor,
            subtile_size=256 * 1024  # 256KB subtiles
        )
        
        # Store generation snapshot
        metadata = self.blockchain_manager.tile_metadata.get(tile_id, {})
        snapshot_hash = self.blockchain_manager.create_tile_block(
            generation_tile,
            expert_name=f"{metadata.get('expert_name', '')}_gen_{generation_id}",
            layer_id=metadata.get('layer_id', '')
        )
        
        # Save generation metadata to file
        generation_file = self.snapshot_dir / f"generation_{generation_id}_{tile_id}.json"
        generation_data = {
            'generation_id': generation_id,
            'tile_id': tile_id,
            'snapshot_hash': snapshot_hash,
            'creation_timestamp': time.time(),
            'original_versions': len(versions),
            'generation_boundary': True
        }
        
        with open(generation_file, 'w') as f:
            json.dump(generation_data, f, indent=2)
        
        return snapshot_hash
    
    def _update_blockchain_generation_metadata(self, tile_id: str, generation_id: int):
        """Update blockchain block metadata with generation information."""
        if tile_id not in self.blockchain_manager.tile_versions:
            return
        
        # Update metadata for the tile
        if tile_id in self.blockchain_manager.tile_metadata:
            self.blockchain_manager.tile_metadata[tile_id].update({
                'current_generation': generation_id,
                'last_generation_boundary': time.time(),
                'generation_count': self.blockchain_manager.tile_metadata[tile_id].get('generation_count', 0) + 1
            })
    
    def get_tile_generation_history(self, tile_id: str) -> List[Dict[str, Any]]:
        """Get generation history for a tile."""
        history = []
        
        for gen_id, metadata in self.generation_metadata.items():
            if metadata.epoch_summary.get('tile_id') == tile_id:
                history.append({
                    'generation_id': gen_id,
                    'creation_timestamp': metadata.creation_timestamp,
                    'total_deltas': metadata.total_deltas,
                    'performance_improvement': metadata.performance_improvement,
                    'validation_score': metadata.validation_score,
                    'snapshot_hash': metadata.snapshot_hash,
                    'start_block': metadata.start_block_hash,
                    'end_block': metadata.end_block_hash
                })
        
        return sorted(history, key=lambda x: x['generation_id'])
    
    def validate_generation_integrity(self, generation_id: int) -> bool:
        """Validate integrity of a generation."""
        if generation_id not in self.generation_metadata:
            return False
        
        metadata = self.generation_metadata[generation_id]
        
        try:
            # Check if snapshot exists
            snapshot_block = self.blockchain_manager.tile_chain.get_block_by_hash(metadata.snapshot_hash)
            if not snapshot_block:
                return False
            
            # Check if start and end blocks exist
            start_block = self.blockchain_manager.tile_chain.get_block_by_hash(metadata.start_block_hash)
            end_block = self.blockchain_manager.tile_chain.get_block_by_hash(metadata.end_block_hash)
            
            if not start_block or not end_block:
                return False
            
            # Validate timestamp ordering
            if start_block.header.timestamp > end_block.header.timestamp:
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Generation {generation_id} validation failed: {e}")
            return False
    
    def get_generation_report(self) -> Dict[str, Any]:
        """Get comprehensive generation management report."""
        valid_generations = sum(1 for gen_id in self.generation_metadata.keys() 
                              if self.validate_generation_integrity(gen_id))
        
        generation_summary = {}
        for gen_id, metadata in self.generation_metadata.items():
            generation_summary[gen_id] = {
                'tile_id': metadata.epoch_summary.get('tile_id'),
                'creation_time': metadata.creation_timestamp,
                'deltas_count': metadata.total_deltas,
                'performance_improvement': metadata.performance_improvement,
                'validation_score': metadata.validation_score,
                'is_valid': self.validate_generation_integrity(gen_id)
            }
        
        return {
            'current_generation_id': self.current_generation_id,
            'total_generations': len(self.generation_metadata),
            'valid_generations': valid_generations,
            'generation_boundaries': self.generation_boundaries.copy(),
            'epoch_thresholds': self.epoch_thresholds.copy(),
            'generation_summary': generation_summary,
            'tiles_with_generations': len(self.generation_boundaries)
        }
    
    def restore_from_snapshot(self, tile_id: str, snapshot_hash: str) -> bool:
        """Restore a tile from a snapshot (recovery feature)"""
        try:
            # Load snapshot
            tensor = self.blockchain_manager.zero_copy_loader.load_tile(snapshot_hash, device='cpu')
            
            # Create new tile
            restored_tile = TileBlock(tile_id, tensor)
            
            # Store as new version
            metadata = self.blockchain_manager.tile_metadata.get(tile_id, {})
            new_hash = self.blockchain_manager.create_tile_block(
                restored_tile,
                expert_name=metadata.get('expert_name', ''),
                layer_id=metadata.get('layer_id', '')
            )
            
            print(f"🔄 Restored tile {tile_id} from snapshot {snapshot_hash[:8]}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to restore tile {tile_id}: {e}")
            return False

# Export main class
__all__ = ['SnapshotCompactor', 'SnapshotMetrics']