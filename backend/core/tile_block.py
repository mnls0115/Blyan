"""
Blyan Tile-Based Block System
Revolutionary zero-copy tile format for distributed AI learning
"""

import struct
import hashlib
import mmap
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import torch
import numpy as np

# Tile format constants
TILE_MAGIC = b"TBLOCK01"
TILE_HEADER_SIZE = 128
DEFAULT_TILE_SIZE = 4 * 1024 * 1024  # 4MB
DEFAULT_SUBTILE_SIZE = 256 * 1024    # 256KB

@dataclass
class TileHeader:
    """Tile block header with metadata for zero-copy loading"""
    magic: bytes = TILE_MAGIC
    version: int = 1
    dtype: int = 1  # 1=fp16, 2=int8, 3=fp8
    shape: Tuple[int, ...] = ()
    layout: int = 0  # 0=row_major, 1=col_major
    quantization_method: int = 0  # 0=none, 1=per_tensor, 2=per_channel
    scale_offset: int = 0
    data_offset: int = TILE_HEADER_SIZE
    merkle_root_offset: int = 0
    subtile_count: int = 0
    tile_size: int = DEFAULT_TILE_SIZE
    
    def to_bytes(self) -> bytes:
        """Serialize header to bytes"""
        # Pack shape as variable length (max 8 dimensions)
        shape_bytes = struct.pack(f'<{len(self.shape)}I', *self.shape) if self.shape else b''
        shape_bytes = shape_bytes.ljust(32, b'\x00')  # Pad to 32 bytes
        
        header_data = struct.pack(
            '<8sIIII32sIIIIII',
            self.magic,
            self.version,
            self.dtype,
            len(self.shape),
            self.layout,
            shape_bytes,
            self.quantization_method,
            self.scale_offset,
            self.data_offset,
            self.merkle_root_offset,
            self.subtile_count,
            self.tile_size
        )
        
        # Pad to TILE_HEADER_SIZE
        return header_data.ljust(TILE_HEADER_SIZE, b'\x00')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'TileHeader':
        """Deserialize header from bytes"""
        if len(data) < TILE_HEADER_SIZE:
            raise ValueError(f"Header data too small: {len(data)} < {TILE_HEADER_SIZE}")
        
        # Unpack fixed fields
        unpacked = struct.unpack('<8sIIII32sIIIIII', data[:80])
        magic, version, dtype, shape_len, layout = unpacked[:5]
        shape_bytes = unpacked[5]
        quantization_method, scale_offset, data_offset, merkle_root_offset, subtile_count, tile_size = unpacked[6:]
        
        # Decode shape
        if shape_len > 0:
            shape = struct.unpack(f'<{shape_len}I', shape_bytes[:shape_len * 4])
        else:
            shape = ()
        
        return cls(
            magic=magic,
            version=version,
            dtype=dtype,
            shape=shape,
            layout=layout,
            quantization_method=quantization_method,
            scale_offset=scale_offset,
            data_offset=data_offset,
            merkle_root_offset=merkle_root_offset,
            subtile_count=subtile_count,
            tile_size=tile_size
        )
    
    def get_torch_dtype(self) -> torch.dtype:
        """Convert dtype code to torch dtype"""
        dtype_map = {
            1: torch.float16,
            2: torch.int8,
            3: torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.float16
        }
        return dtype_map.get(self.dtype, torch.float16)
    
    def get_element_count(self) -> int:
        """Get total number of elements in tensor"""
        if not self.shape:
            return 0
        count = 1
        for dim in self.shape:
            count *= dim
        return count

@dataclass
class SubTileInfo:
    """Information about a subtile within a tile"""
    subtile_id: int
    offset: int
    size: int
    hash: bytes
    is_modified: bool = False

class TileBlock:
    """
    Revolutionary tile-based block for zero-copy distributed learning
    
    Format:
    [Header: 128 bytes]
    [Quantization Metadata: Variable]
    [Tensor Data: tile_size bytes]
    [SubTile Index: Variable]
    [Merkle Tree: Variable]
    """
    
    def __init__(self, tile_id: str, tensor_data: torch.Tensor, subtile_size: int = DEFAULT_SUBTILE_SIZE):
        self.tile_id = tile_id
        self.tensor_data = tensor_data
        self.subtile_size = subtile_size
        self.header = self._create_header()
        self.subtiles: List[SubTileInfo] = []
        self._create_subtile_index()
    
    def _create_header(self) -> TileHeader:
        """Create header from tensor data"""
        dtype_map = {
            torch.float16: 1,
            torch.int8: 2,
            torch.float8_e4m3fn: 3 if hasattr(torch, 'float8_e4m3fn') else 1
        }
        
        return TileHeader(
            dtype=dtype_map.get(self.tensor_data.dtype, 1),
            shape=tuple(self.tensor_data.shape),
            data_offset=TILE_HEADER_SIZE,
            tile_size=self.tensor_data.numel() * self.tensor_data.element_size()
        )
    
    def _create_subtile_index(self):
        """Divide tile into subtiles for efficient delta operations"""
        tensor_bytes = self.tensor_data.numel() * self.tensor_data.element_size()
        subtile_count = (tensor_bytes + self.subtile_size - 1) // self.subtile_size
        
        self.subtiles = []
        for i in range(subtile_count):
            offset = i * self.subtile_size
            size = min(self.subtile_size, tensor_bytes - offset)
            
            # Calculate hash of subtile data
            start_elem = offset // self.tensor_data.element_size()
            end_elem = (offset + size) // self.tensor_data.element_size()
            subtile_data = self.tensor_data.flatten()[start_elem:end_elem]
            subtile_hash = hashlib.sha256(subtile_data.cpu().numpy().tobytes()).digest()
            
            self.subtiles.append(SubTileInfo(
                subtile_id=i,
                offset=offset,
                size=size,
                hash=subtile_hash
            ))
        
        self.header.subtile_count = len(self.subtiles)
    
    def to_bytes(self) -> bytes:
        """Serialize entire tile to bytes"""
        # Header
        header_bytes = self.header.to_bytes()
        
        # Tensor data (contiguous, aligned)
        if not self.tensor_data.is_contiguous():
            self.tensor_data = self.tensor_data.contiguous()
        
        tensor_bytes = self.tensor_data.cpu().numpy().tobytes()
        
        # SubTile index
        subtile_index = b''
        for subtile in self.subtiles:
            subtile_index += struct.pack('<III32s?', 
                subtile.subtile_id,
                subtile.offset, 
                subtile.size, 
                subtile.hash,
                subtile.is_modified
            )
        
        # Calculate offsets
        merkle_offset = len(header_bytes) + len(tensor_bytes) + len(subtile_index)
        
        # Simple merkle tree (just subtile hashes for now)
        merkle_tree = b''.join(subtile.hash for subtile in self.subtiles)
        
        # Update header with correct offsets
        self.header.merkle_root_offset = merkle_offset
        header_bytes = self.header.to_bytes()
        
        return header_bytes + tensor_bytes + subtile_index + merkle_tree
    
    @classmethod
    def from_bytes(cls, data: bytes, tile_id: str) -> 'TileBlock':
        """Deserialize tile from bytes"""
        # Parse header
        header = TileHeader.from_bytes(data[:TILE_HEADER_SIZE])
        
        # Extract tensor data
        tensor_start = header.data_offset
        element_size = 2 if header.dtype == 1 else 1  # fp16=2, int8=1
        tensor_size = header.get_element_count() * element_size
        
        tensor_bytes = data[tensor_start:tensor_start + tensor_size]
        
        # Convert to numpy then torch
        np_dtype = np.float16 if header.dtype == 1 else np.int8
        np_array = np.frombuffer(tensor_bytes, dtype=np_dtype)
        tensor_data = torch.from_numpy(np_array).view(header.shape)
        
        # Create tile instance
        tile = cls(tile_id, tensor_data)
        tile.header = header
        
        # Parse subtile index if present
        if header.subtile_count > 0:
            subtile_start = tensor_start + tensor_size
            tile.subtiles = []
            
            for i in range(header.subtile_count):
                offset = subtile_start + i * 41  # struct size
                subtile_data = struct.unpack('<III32s?', data[offset:offset + 41])
                
                tile.subtiles.append(SubTileInfo(
                    subtile_id=subtile_data[0],
                    offset=subtile_data[1],
                    size=subtile_data[2],
                    hash=subtile_data[3],
                    is_modified=subtile_data[4]
                ))
        
        return tile
    
    def get_subtile_tensor(self, subtile_id: int) -> torch.Tensor:
        """Get tensor data for a specific subtile"""
        if subtile_id >= len(self.subtiles):
            raise ValueError(f"SubTile {subtile_id} not found")
        
        subtile = self.subtiles[subtile_id]
        element_size = self.tensor_data.element_size()
        start_elem = subtile.offset // element_size
        end_elem = (subtile.offset + subtile.size) // element_size
        
        return self.tensor_data.flatten()[start_elem:end_elem]
    
    def compute_tile_hash(self) -> str:
        """Compute hash of entire tile for blockchain storage"""
        tile_bytes = self.to_bytes()
        return hashlib.sha256(tile_bytes).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify tile integrity using subtile hashes"""
        for subtile in self.subtiles:
            subtile_tensor = self.get_subtile_tensor(subtile.subtile_id)
            computed_hash = hashlib.sha256(subtile_tensor.cpu().numpy().tobytes()).digest()
            if computed_hash != subtile.hash:
                return False
        return True

class TileBlockFactory:
    """Factory for creating different types of tile blocks"""
    
    @staticmethod
    def from_expert_weights(expert_name: str, weights: Dict[str, torch.Tensor], 
                          tile_size: int = DEFAULT_TILE_SIZE) -> List[TileBlock]:
        """Convert expert weights dict to multiple tiles"""
        tiles = []
        
        for weight_name, tensor in weights.items():
            # Flatten large tensors into multiple tiles if needed
            if tensor.numel() * tensor.element_size() > tile_size:
                # Split into multiple tiles
                flat_tensor = tensor.flatten()
                elements_per_tile = tile_size // tensor.element_size()
                
                for i in range(0, flat_tensor.numel(), elements_per_tile):
                    end_idx = min(i + elements_per_tile, flat_tensor.numel())
                    tile_tensor = flat_tensor[i:end_idx]
                    
                    tile_id = f"{expert_name}.{weight_name}.tile_{i//elements_per_tile}"
                    tiles.append(TileBlock(tile_id, tile_tensor))
            else:
                # Single tile
                tile_id = f"{expert_name}.{weight_name}"
                tiles.append(TileBlock(tile_id, tensor))
        
        return tiles
    
    @staticmethod
    def create_empty_tile(tile_id: str, shape: Tuple[int, ...], 
                         dtype: torch.dtype = torch.float16) -> TileBlock:
        """Create empty tile with specified shape and dtype"""
        tensor = torch.zeros(shape, dtype=dtype)
        return TileBlock(tile_id, tensor)

# Export main classes
__all__ = ['TileBlock', 'TileHeader', 'SubTileInfo', 'TileBlockFactory']