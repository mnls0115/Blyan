"""
Tile-Blockchain Integration Layer for Blyan
Connects tile-based distributed learning with Blyan's custom blockchain
"""

import asyncio
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
import torch

from .chain import Chain
from .block import Block, BlockHeader
from .tile_block import TileBlock, TileBlockFactory
from .zero_copy_loader import ZeroCopyTileLoader
from .delta_compression import DeltaBase, DeltaCompressor
from .tile_ownership import TileOwnershipRegistry
from .edge_aggregator import EdgeAggregator, DeltaSubmission

class TileBlockchainManager:
    """
    Integration manager between tile-based learning and Blyan blockchain
    
    Features:
    - Store tiles as blockchain blocks
    - Apply deltas to create new tile versions
    - Maintain tile ownership and versioning
    - Zero-copy loading from blockchain
    """
    
    def __init__(self, 
                 data_dir: Path = Path("./data"),
                 tile_chain_id: str = "T",  # New chain for tiles
                 difficulty: int = 2):  # Lower difficulty for tiles
        
        self.data_dir = data_dir
        self.tile_chain_id = tile_chain_id
        
        # Initialize Blyan blockchain for tiles
        self.tile_chain = Chain(
            root_dir=data_dir,
            chain_id=tile_chain_id,
            difficulty=difficulty,
            skip_pow=False  # Enable PoW for security
        )
        
        # Tile management
        self.zero_copy_loader = ZeroCopyTileLoader(self.tile_chain)
        self.delta_compressor = DeltaCompressor()
        self.ownership_registry = TileOwnershipRegistry()
        
        # Tile versioning and metadata
        self.tile_versions: Dict[str, List[str]] = {}  # tile_id -> [block_hashes]
        self.tile_metadata: Dict[str, Dict] = {}  # tile_id -> metadata
        
        # Performance stats
        self.stats = {
            'tiles_created': 0,
            'deltas_applied': 0,
            'blocks_mined': 0,
            'total_tile_size_mb': 0.0
        }
        
        print(f"🔗 TileBlockchainManager initialized with chain {tile_chain_id}")
    
    def create_tile_block(self, tile: TileBlock, expert_name: str = "", 
                         layer_id: str = "") -> str:
        """
        Store a tile as a block in Blyan blockchain
        
        Returns:
            Block hash of created tile block
        """
        try:
            # Serialize tile to bytes
            tile_bytes = tile.to_bytes()
            
            # Create block header with tile metadata
            header = BlockHeader(
                index=self.tile_chain.storage.get_block_count(),
                timestamp=time.time(),
                prev_hash=self.tile_chain._latest().compute_hash() if self.tile_chain._latest() else "0",
                chain_id=self.tile_chain_id,
                points_to=None,
                payload_hash=hashlib.sha256(tile_bytes).hexdigest(),
                payload_size=len(tile_bytes),
                
                # Tile-specific metadata
                block_type='expert',  # Tiles are expert data
                expert_name=expert_name or tile.tile_id,
                layer_id=layer_id,
                
                # TensorBlock format info
                payload_type="tile_stream",
                tensor_dtype=self._get_tensor_dtype_string(tile.tensor_data.dtype),
                tensor_shape=list(tile.tensor_data.shape),
                tensor_layout="row_major",
                quantization_method="none",
                merkle_root=self._compute_tile_merkle_root(tile),
                
                # Versioning
                version="1.0.0",
                evolution_type="creation"
            )
            
            # Create block
            block = Block(header=header, payload=tile_bytes)
            
            # Add to blockchain (will mine PoW)
            success = self.tile_chain.add_block(tile_bytes, 
                                              block_type='expert',
                                              expert_name=expert_name or tile.tile_id,
                                              layer_id=layer_id)
            
            if success:
                block_hash = block.compute_hash()
                
                # Update tile tracking
                if tile.tile_id not in self.tile_versions:
                    self.tile_versions[tile.tile_id] = []
                self.tile_versions[tile.tile_id].append(block_hash)
                
                self.tile_metadata[tile.tile_id] = {
                    'expert_name': expert_name,
                    'layer_id': layer_id,
                    'shape': list(tile.tensor_data.shape),
                    'dtype': str(tile.tensor_data.dtype),
                    'size_mb': len(tile_bytes) / (1024 * 1024),
                    'created_at': time.time(),
                    'subtile_count': len(tile.subtiles)
                }
                
                # Update stats
                self.stats['tiles_created'] += 1
                self.stats['blocks_mined'] += 1
                self.stats['total_tile_size_mb'] += len(tile_bytes) / (1024 * 1024)
                
                print(f"✅ Tile {tile.tile_id} stored as block {block_hash[:8]}...")
                return block_hash
            else:
                raise RuntimeError("Failed to add tile block to blockchain")
                
        except Exception as e:
            print(f"❌ Failed to create tile block: {e}")
            raise
    
    def apply_delta_to_tile(self, tile_id: str, delta: DeltaBase, 
                           node_id: str = "system") -> str:
        """
        Apply delta to tile and create new version on blockchain
        
        Returns:
            Block hash of new tile version
        """
        try:
            # Get latest tile version
            if tile_id not in self.tile_versions or not self.tile_versions[tile_id]:
                raise ValueError(f"Tile {tile_id} not found")
            
            latest_block_hash = self.tile_versions[tile_id][-1]
            
            # Load current tile using zero-copy loader
            current_tensor = self.zero_copy_loader.load_tile(latest_block_hash, device='cpu')
            
            # Apply delta
            new_tensor = delta.apply_to_tensor(current_tensor)
            
            # Create new tile
            new_tile = TileBlock(tile_id, new_tensor)
            
            # Get original metadata
            metadata = self.tile_metadata.get(tile_id, {})
            
            # Store new version with incremented version
            old_version = metadata.get('version', '1.0.0')
            new_version = self._increment_version(old_version)
            
            # Create delta block first (stores the delta)
            delta_block_hash = self._create_delta_block(tile_id, delta, node_id, 
                                                      latest_block_hash, new_version)
            
            # Create new tile block
            new_block_hash = self.create_tile_block(
                new_tile, 
                expert_name=metadata.get('expert_name', ''),
                layer_id=metadata.get('layer_id', '')
            )
            
            # Update metadata
            self.tile_metadata[tile_id].update({
                'version': new_version,
                'last_updated': time.time(),
                'delta_applied_by': node_id,
                'delta_block': delta_block_hash
            })
            
            self.stats['deltas_applied'] += 1
            
            print(f"✅ Applied delta to {tile_id}, new version: {new_block_hash[:8]}...")
            return new_block_hash
            
        except Exception as e:
            print(f"❌ Failed to apply delta to tile: {e}")
            raise
    
    def _create_delta_block(self, tile_id: str, delta: DeltaBase, node_id: str,
                           parent_hash: str, version: str) -> str:
        """Create a separate block storing the delta operation"""
        try:
            # Serialize delta
            delta_bytes = delta.to_bytes()
            
            # Create delta metadata
            delta_metadata = {
                'tile_id': tile_id,
                'delta_type': type(delta).__name__,
                'compression_ratio': delta.get_compression_ratio(),
                'applied_by': node_id,
                'parent_tile': parent_hash,
                'timestamp': time.time()
            }
            
            # Combine delta and metadata
            delta_payload = {
                'metadata': delta_metadata,
                'delta_data': delta_bytes.hex()  # Store as hex string
            }
            
            payload_bytes = json.dumps(delta_payload).encode()
            
            # Create block header
            header = BlockHeader(
                index=self.tile_chain.storage.get_block_count(),
                timestamp=time.time(),
                prev_hash=self.tile_chain._latest().compute_hash() if self.tile_chain._latest() else "0",
                chain_id=self.tile_chain_id,
                points_to=parent_hash,  # Points to parent tile
                payload_hash=hashlib.sha256(payload_bytes).hexdigest(),
                payload_size=len(payload_bytes),
                
                # Delta-specific metadata
                block_type='router',  # Use router type for delta blocks
                expert_name=tile_id,
                
                # Evolution metadata
                version=version,
                parent_hash=parent_hash,
                evolution_type="delta_application",
                
                # Payload info
                payload_type="json"
            )
            
            # Add to blockchain
            block = Block(header=header, payload=payload_bytes)
            success = self.tile_chain.add_block(payload_bytes,
                                              block_type='router',
                                              expert_name=tile_id)
            
            if success:
                self.stats['blocks_mined'] += 1
                return block.compute_hash()
            else:
                raise RuntimeError("Failed to add delta block")
                
        except Exception as e:
            print(f"❌ Failed to create delta block: {e}")
            raise
    
    def load_tile_zero_copy(self, tile_id: str, version: str = "latest", 
                           device: str = 'cuda') -> torch.Tensor:
        """Load tile using zero-copy optimization"""
        try:
            if tile_id not in self.tile_versions:
                raise ValueError(f"Tile {tile_id} not found")
            
            # Get block hash for requested version
            if version == "latest":
                block_hash = self.tile_versions[tile_id][-1]
            else:
                # Find version by index or hash
                try:
                    version_index = int(version)
                    block_hash = self.tile_versions[tile_id][version_index]
                except (ValueError, IndexError):
                    # Assume it's a hash
                    if version in self.tile_versions[tile_id]:
                        block_hash = version
                    else:
                        raise ValueError(f"Version {version} not found for tile {tile_id}")
            
            # Load using zero-copy loader
            return self.zero_copy_loader.load_tile(block_hash, device=device)
            
        except Exception as e:
            print(f"❌ Failed to load tile {tile_id}: {e}")
            raise
    
    def get_tile_history(self, tile_id: str) -> List[Dict]:
        """Get complete history of tile versions"""
        if tile_id not in self.tile_versions:
            return []
        
        history = []
        for i, block_hash in enumerate(self.tile_versions[tile_id]):
            # Get block from blockchain
            block = self.tile_chain.get_block_by_hash(block_hash)
            if block:
                history.append({
                    'version': i,
                    'block_hash': block_hash,
                    'timestamp': block.header.timestamp,
                    'size_bytes': block.header.payload_size,
                    'evolution_type': block.header.evolution_type,
                    'applied_by': block.header.expert_name
                })
        
        return history
    
    def create_tiles_from_expert(self, expert_name: str, weights: Dict[str, torch.Tensor],
                                layer_id: str = "") -> List[str]:
        """Create multiple tiles from expert weights"""
        try:
            # Use factory to create tiles
            tiles = TileBlockFactory.from_expert_weights(expert_name, weights)
            
            # Store each tile on blockchain
            block_hashes = []
            for tile in tiles:
                block_hash = self.create_tile_block(tile, expert_name, layer_id)
                block_hashes.append(block_hash)
            
            print(f"✅ Created {len(tiles)} tiles for expert {expert_name}")
            return block_hashes
            
        except Exception as e:
            print(f"❌ Failed to create tiles from expert: {e}")
            raise
    
    def get_expert_tiles(self, expert_name: str) -> Dict[str, str]:
        """Get all tiles belonging to an expert"""
        expert_tiles = {}
        
        for tile_id, versions in self.tile_versions.items():
            metadata = self.tile_metadata.get(tile_id, {})
            if metadata.get('expert_name') == expert_name:
                expert_tiles[tile_id] = versions[-1]  # Latest version
        
        return expert_tiles
    
    def compact_tile_chain(self, tile_id: str, keep_last_n: int = 5):
        """Compact tile chain by removing old versions (keep only last N)"""
        if tile_id not in self.tile_versions:
            return
        
        versions = self.tile_versions[tile_id]
        if len(versions) <= keep_last_n:
            return
        
        # Keep only last N versions
        old_versions = versions[:-keep_last_n]
        self.tile_versions[tile_id] = versions[-keep_last_n:]
        
        # Note: In production, old blocks would be archived, not deleted
        print(f"📦 Compacted {tile_id}: removed {len(old_versions)} old versions")
        
        return old_versions
    
    def _get_tensor_dtype_string(self, dtype: torch.dtype) -> str:
        """Convert torch dtype to string"""
        dtype_map = {
            torch.float16: "fp16",
            torch.int8: "int8",
            torch.float32: "fp32"
        }
        return dtype_map.get(dtype, "fp16")
    
    def _compute_tile_merkle_root(self, tile: TileBlock) -> str:
        """Compute merkle root for tile integrity"""
        if not tile.subtiles:
            return ""
        
        # Simple merkle root from subtile hashes
        hashes = [subtile.hash.hex() for subtile in tile.subtiles]
        while len(hashes) > 1:
            new_hashes = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    combined = hashes[i] + hashes[i + 1]
                else:
                    combined = hashes[i] + hashes[i]
                new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = new_hashes
        
        return hashes[0] if hashes else ""
    
    def _increment_version(self, version: str) -> str:
        """Increment semantic version"""
        try:
            parts = version.split('.')
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{major}.{minor}.{patch + 1}"
        except:
            return "1.0.1"
    
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain and tile statistics"""
        chain_stats = {
            'chain_id': self.tile_chain_id,
            'total_blocks': self.tile_chain.storage.get_block_count(),
            'latest_block_hash': self.tile_chain._latest().compute_hash() if self.tile_chain._latest() else None,
            'difficulty': self.tile_chain.difficulty
        }
        
        tile_stats = {
            'total_tiles': len(self.tile_versions),
            'total_versions': sum(len(versions) for versions in self.tile_versions.values()),
            'tiles_created': self.stats['tiles_created'],
            'deltas_applied': self.stats['deltas_applied'],
            'blocks_mined': self.stats['blocks_mined'],
            'total_size_mb': self.stats['total_tile_size_mb']
        }
        
        return {
            'blockchain': chain_stats,
            'tiles': tile_stats,
            'cache_stats': self.zero_copy_loader.get_cache_stats()
        }

class DistributedTileLearning:
    """
    Complete distributed tile-based learning system
    Combines EdgeAggregator + TileBlockchainManager + OwnershipRegistry
    """
    
    def __init__(self, region: str, data_dir: Path = Path("./data")):
        self.region = region
        
        # Initialize core components
        self.blockchain_manager = TileBlockchainManager(data_dir)
        self.ownership_registry = self.blockchain_manager.ownership_registry
        self.edge_aggregator = EdgeAggregator(
            region=region,
            ownership_registry=self.ownership_registry
        )
        
        print(f"🌐 DistributedTileLearning initialized for region {region}")
    
    async def start_learning_network(self, port: int = 8080):
        """Start the complete distributed learning network"""
        try:
            # Start edge aggregator
            await self.edge_aggregator.start_server()
            
            print(f"🚀 Distributed tile learning network started on port {port}")
            print(f"📊 Region: {self.region}")
            print(f"⛓️  Blockchain: {self.blockchain_manager.tile_chain_id}")
            
        except Exception as e:
            print(f"❌ Failed to start learning network: {e}")
            raise
    
    async def submit_expert_for_learning(self, expert_name: str, 
                                       weights: Dict[str, torch.Tensor]) -> List[str]:
        """Submit expert weights for distributed learning"""
        # Create tiles from expert
        block_hashes = self.blockchain_manager.create_tiles_from_expert(expert_name, weights)
        
        # Register tiles with ownership system
        for i, block_hash in enumerate(block_hashes):
            tile_id = f"{expert_name}.tile_{i}"
            self.ownership_registry.elect_primary(tile_id)
        
        return block_hashes
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning system statistics"""
        return {
            'blockchain': self.blockchain_manager.get_blockchain_stats(),
            'aggregation': self.edge_aggregator.get_stats(),
            'ownership': self.ownership_registry.get_node_stats()
        }

# Export main classes
__all__ = ['TileBlockchainManager', 'DistributedTileLearning']