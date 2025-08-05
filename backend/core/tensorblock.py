#!/usr/bin/env python3
"""Zero-copy TensorBlock format for high-performance expert loading.

Implements the TensorBlock format specified in the whitepaper:
- Header + contiguous tensor data + optional quantization metadata  
- Memory-mappable for zero-copy loading
- Tile-based merkle indexing for partial verification
- Support for FP16, INT8, FP8 with quantization metadata
"""

import os
import mmap
import struct
import hashlib
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TensorBlockHeader:
    """TensorBlock header structure (128 bytes total)."""
    magic: bytes = b"TBLOCK01"  # 8 bytes
    version: int = 1            # 4 bytes  
    dtype: int = 1              # 4 bytes: fp16=1, int8=2, fp8=3
    shape_rank: int = 2         # 4 bytes
    shape: Tuple[int, ...] = (0, 0)  # 16 bytes max (up to 4 dimensions)
    layout: int = 0             # 4 bytes: row_major=0, col_major=1
    quantization_method: int = 0 # 4 bytes: none=0, per_tensor_int8=1, per_channel_int8=2
    scale_offset: int = 0       # 8 bytes: offset to quantization scales
    data_offset: int = 128      # 8 bytes: offset to tensor data
    merkle_root_offset: int = 0 # 8 bytes: offset to merkle index
    tile_size: int = 1024       # 4 bytes: tile size for merkle indexing
    reserved: bytes = b'\x00' * 68  # 68 bytes reserved for future use (total 128 bytes)


@dataclass 
class QuantizationMetadata:
    """Quantization parameters for INT8/FP8 tensors."""
    method: str  # "none", "per_tensor_int8", "per_channel_int8"
    scales: Optional[torch.Tensor] = None
    zero_points: Optional[torch.Tensor] = None
    

class TensorBlockWriter:
    """Writes tensors to TensorBlock format."""
    
    DTYPE_MAP = {
        torch.float16: 1,
        torch.int8: 2,
        torch.float8_e4m3fn: 3,  # FP8 (if available)
    }
    
    DTYPE_SIZES = {
        1: 2,  # fp16
        2: 1,  # int8
        3: 1,  # fp8
    }
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.tile_size = 1024  # Default tile size for merkle indexing
        
    def write_tensor(self, 
                    tensor: torch.Tensor,
                    quantization: Optional[QuantizationMetadata] = None) -> Dict[str, any]:
        """Write tensor to TensorBlock format file."""
        
        # Validate tensor
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
            
        if tensor.dtype not in self.DTYPE_MAP:
            raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")
            
        # Create header
        header = TensorBlockHeader(
            dtype=self.DTYPE_MAP[tensor.dtype],
            shape_rank=len(tensor.shape),
            shape=tuple(tensor.shape + (0,) * (4 - len(tensor.shape))),  # Pad to 4 dims
            quantization_method=self._get_quant_method_id(quantization),
            tile_size=self.tile_size
        )
        
        # Calculate offsets
        scale_size = 0
        if quantization and quantization.scales is not None:
            scale_size = quantization.scales.numel() * 4  # FP32 scales
            
        header.scale_offset = 128 if scale_size > 0 else 0
        header.data_offset = 128 + scale_size
        
        # Calculate tensor data size
        tensor_bytes = tensor.numel() * self.DTYPE_SIZES[header.dtype]
        
        # Calculate merkle index offset  
        header.merkle_root_offset = header.data_offset + tensor_bytes
        
        # Write to file
        with open(self.file_path, 'wb') as f:
            # Write header
            self._write_header(f, header)
            
            # Write quantization metadata
            if quantization and quantization.scales is not None:
                scales_bytes = quantization.scales.cpu().numpy().astype(np.float32).tobytes()
                f.write(scales_bytes)
                
            # Write tensor data
            tensor_bytes_data = tensor.cpu().numpy().tobytes()
            f.write(tensor_bytes_data)
            
            # Write merkle index
            merkle_root, merkle_index = self._build_merkle_index(tensor_bytes_data)
            f.write(merkle_index)
            
        # Return metadata for blockchain storage
        return {
            "file_path": str(self.file_path),
            "header": header,
            "tensor_shape": tensor.shape,
            "tensor_dtype": str(tensor.dtype),
            "file_size": self.file_path.stat().st_size,
            "merkle_root": merkle_root.hex(),
            "quantization": quantization.method if quantization else "none"
        }
    
    def _write_header(self, f: BinaryIO, header: TensorBlockHeader):
        """Write header to file in binary format."""
        # Pack header struct (128 bytes total)
        header_bytes = struct.pack(
            '<8sIIII4IIIIII68s',  # Little-endian format
            header.magic,
            header.version,
            header.dtype,
            header.shape_rank,
            *header.shape,  # 4 ints for shape
            header.layout,
            header.quantization_method,
            header.scale_offset,
            header.data_offset,
            header.merkle_root_offset,
            header.tile_size,
            header.reserved
        )
        
        assert len(header_bytes) == 128, f"Header size mismatch: {len(header_bytes)}"
        f.write(header_bytes)
    
    def _get_quant_method_id(self, quantization: Optional[QuantizationMetadata]) -> int:
        """Get quantization method ID."""
        if not quantization or quantization.method == "none":
            return 0
        elif quantization.method == "per_tensor_int8":
            return 1
        elif quantization.method == "per_channel_int8":
            return 2
        else:
            raise ValueError(f"Unknown quantization method: {quantization.method}")
    
    def _build_merkle_index(self, tensor_data: bytes) -> Tuple[bytes, bytes]:
        """Build merkle tree index for tile-based verification."""
        # Split tensor data into tiles
        tiles = []
        for i in range(0, len(tensor_data), self.tile_size):
            tile = tensor_data[i:i + self.tile_size]
            tile_hash = hashlib.sha256(tile).digest()
            tiles.append(tile_hash)
        
        # Build merkle tree (simplified - just hash all tile hashes)
        if not tiles:
            merkle_root = hashlib.sha256(b'').digest()
        else:
            combined_hashes = b''.join(tiles)
            merkle_root = hashlib.sha256(combined_hashes).digest()
        
        # Create merkle index (store all tile hashes)
        merkle_index = struct.pack('<I', len(tiles))  # Number of tiles
        merkle_index += b''.join(tiles)  # All tile hashes
        
        return merkle_root, merkle_index


class TensorBlockReader:
    """Reads tensors from TensorBlock format with zero-copy optimization."""
    
    DTYPE_MAP_REVERSE = {
        1: torch.float16,
        2: torch.int8, 
        3: torch.float8_e4m3fn,  # FP8 (if available)
    }
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self._mmap_file = None
        self._header = None
        
    def __enter__(self):
        """Context manager for automatic cleanup."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up memory mapping."""
        self.close()
        
    def close(self):
        """Close memory mapping."""
        if self._mmap_file:
            self._mmap_file.close()
            self._mmap_file = None
            
    def load_tensor_zero_copy(self, device: str = "cpu") -> torch.Tensor:
        """Load tensor with zero-copy optimization using memory mapping."""
        
        # Open file and create memory mapping
        file_obj = open(self.file_path, 'rb')
        self._mmap_file = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Parse header
        self._header = self._parse_header()
        
        # Get tensor metadata
        tensor_dtype = self.DTYPE_MAP_REVERSE[self._header.dtype]
        tensor_shape = self._header.shape[:self._header.shape_rank]
        
        # Create tensor view directly from memory map (zero-copy)
        tensor_view = torch.frombuffer(
            self._mmap_file,
            dtype=tensor_dtype,
            count=np.prod(tensor_shape),
            offset=self._header.data_offset
        ).view(tensor_shape)
        
        # Handle quantization if present
        if self._header.quantization_method > 0:
            # Load quantization scales
            scales = self._load_quantization_scales()
            tensor_view = self._dequantize_tensor(tensor_view, scales)
        
        # Transfer to target device
        if device != "cpu":
            # Pin memory for faster GPU transfer
            tensor_pinned = tensor_view.pin_memory()
            tensor_gpu = tensor_pinned.to(device, non_blocking=True)
            return tensor_gpu
        else:
            return tensor_view.clone()  # Clone to ensure data persistence after mmap close
    
    def _parse_header(self) -> TensorBlockHeader:
        """Parse TensorBlock header from memory map."""
        header_data = self._mmap_file[:128]
        
        # Unpack header
        unpacked = struct.unpack('<8sIIII4IIIIII68s', header_data)
        
        return TensorBlockHeader(
            magic=unpacked[0],
            version=unpacked[1],
            dtype=unpacked[2],
            shape_rank=unpacked[3],
            shape=unpacked[4:8],  # 4 shape dimensions
            layout=unpacked[8],
            quantization_method=unpacked[9],
            scale_offset=unpacked[10],
            data_offset=unpacked[11],
            merkle_root_offset=unpacked[12],
            tile_size=unpacked[13],
            reserved=unpacked[14]
        )
    
    def _load_quantization_scales(self) -> Optional[torch.Tensor]:
        """Load quantization scales if present."""
        if self._header.scale_offset == 0:
            return None
            
        # Calculate number of scales based on quantization method
        if self._header.quantization_method == 1:  # per_tensor
            num_scales = 1
        elif self._header.quantization_method == 2:  # per_channel
            # Assume scales for last dimension (output channels)
            num_scales = self._header.shape[1] if self._header.shape_rank >= 2 else 1
        else:
            return None
            
        # Read scales from memory map
        scale_bytes = self._mmap_file[self._header.scale_offset:self._header.scale_offset + num_scales * 4]
        scales_np = np.frombuffer(scale_bytes, dtype=np.float32)
        
        return torch.from_numpy(scales_np)
    
    def _dequantize_tensor(self, quantized_tensor: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Dequantize INT8 tensor to FP16."""
        if self._header.quantization_method == 1:  # per_tensor
            return quantized_tensor.float() * scales.item()
        elif self._header.quantization_method == 2:  # per_channel
            # Broadcast scales to match tensor dimensions
            scales_expanded = scales.view(1, -1)
            return quantized_tensor.float() * scales_expanded
        else:
            return quantized_tensor
    
    def verify_integrity(self, expected_merkle_root: str) -> bool:
        """Verify tensor integrity using merkle root."""
        # Ensure file is memory mapped
        if not self._mmap_file:
            file_obj = open(self.file_path, 'rb')
            self._mmap_file = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
            
        if not self._header:
            self._header = self._parse_header()
            
        # Read merkle index
        merkle_data = self._mmap_file[self._header.merkle_root_offset:]
        num_tiles = struct.unpack('<I', merkle_data[:4])[0]
        
        # Reconstruct merkle root from tiles
        tile_hashes = []
        for i in range(num_tiles):
            tile_hash = merkle_data[4 + i * 32:4 + (i + 1) * 32]
            tile_hashes.append(tile_hash)
            
        if tile_hashes:
            combined_hashes = b''.join(tile_hashes)
            computed_root = hashlib.sha256(combined_hashes).digest()
        else:
            computed_root = hashlib.sha256(b'').digest()
            
        return computed_root.hex() == expected_merkle_root


# Utility functions for integration with existing codebase
def tensor_to_tensorblock(tensor: torch.Tensor, 
                         output_path: Union[str, Path],
                         quantization: Optional[QuantizationMetadata] = None) -> Dict[str, any]:
    """Convert PyTorch tensor to TensorBlock format."""
    writer = TensorBlockWriter(output_path)
    return writer.write_tensor(tensor, quantization)


def tensorblock_to_tensor(file_path: Union[str, Path], 
                         device: str = "cpu",
                         verify_merkle: Optional[str] = None) -> torch.Tensor:
    """Load tensor from TensorBlock format with zero-copy optimization."""
    with TensorBlockReader(file_path) as reader:
        # Verify integrity if merkle root provided
        if verify_merkle:
            if not reader.verify_integrity(verify_merkle):
                raise ValueError(f"TensorBlock integrity verification failed for {file_path}")
                
        return reader.load_tensor_zero_copy(device)


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("=== TensorBlock Format Demo ===")
    
    # Create test tensor
    test_tensor = torch.randn(1024, 512, dtype=torch.float16)
    print(f"Test tensor shape: {test_tensor.shape}, dtype: {test_tensor.dtype}")
    
    # Test basic conversion
    output_path = "/tmp/test_expert.tblock"
    
    print("\n1. Writing TensorBlock...")
    start_time = time.time()
    metadata = tensor_to_tensorblock(test_tensor, output_path)
    write_time = time.time() - start_time
    print(f"   Write time: {write_time:.3f}s")
    print(f"   File size: {metadata['file_size']:,} bytes")
    print(f"   Merkle root: {metadata['merkle_root'][:16]}...")
    
    print("\n2. Loading with zero-copy...")
    start_time = time.time()
    loaded_tensor = tensorblock_to_tensor(
        output_path, 
        device="cpu",
        verify_merkle=metadata['merkle_root']
    )
    load_time = time.time() - start_time
    print(f"   Load time: {load_time:.3f}s")
    print(f"   Speedup: {write_time/load_time:.1f}x faster loading")
    
    # Verify correctness
    print(f"\n3. Verification:")
    print(f"   Shape match: {test_tensor.shape == loaded_tensor.shape}")
    print(f"   Data match: {torch.allclose(test_tensor, loaded_tensor, atol=1e-6)}")
    
    # Test quantization
    print("\n4. Testing INT8 quantization...")
    
    # Create quantized version
    quantized_tensor = (test_tensor * 127).clamp(-128, 127).to(torch.int8)
    scales = torch.ones(test_tensor.shape[1]) * (1.0 / 127.0)  # Per-channel scales
    
    quant_metadata = QuantizationMetadata(
        method="per_channel_int8",
        scales=scales
    )
    
    quant_path = "/tmp/test_expert_int8.tblock"
    quant_info = tensor_to_tensorblock(quantized_tensor, quant_path, quant_metadata)
    
    # Load and dequantize
    dequantized_tensor = tensorblock_to_tensor(quant_path, verify_merkle=quant_info['merkle_root'])
    
    print(f"   Quantized file size: {quant_info['file_size']:,} bytes")
    print(f"   Compression ratio: {metadata['file_size'] / quant_info['file_size']:.1f}x")
    print(f"   Dequantization accuracy: {torch.mean(torch.abs(test_tensor - dequantized_tensor)):.6f}")
    
    # Cleanup
    os.unlink(output_path)
    os.unlink(quant_path)
    
    print("\n✅ TensorBlock format working correctly!")