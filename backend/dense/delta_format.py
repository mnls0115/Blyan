"""
Universal Delta Format Specification
=====================================
Model-agnostic delta format for storing parameter updates on blockchain.
Supports dense, LoRA, QLoRA, and sparse updates across any model size.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Literal
from enum import Enum
import torch
import numpy as np
import hashlib
import json
import struct
from datetime import datetime
import zlib
import pickle


class DeltaType(Enum):
    """Types of parameter deltas supported."""
    DENSE = "dense"  # Full parameter update
    LORA = "lora"  # Low-rank adaptation
    QLORA = "qlora"  # Quantized LoRA
    SPARSE = "sparse"  # Sparse updates (top-k values)
    STRUCTURED_SPARSE = "structured_sparse"  # Structured sparsity patterns
    DIFFERENTIAL = "differential"  # Diff from base parameters
    ADAPTER = "adapter"  # Adapter modules
    PREFIX = "prefix"  # Prefix tuning
    

class CompressionType(Enum):
    """Compression methods for delta storage."""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    QUANTIZED = "quantized"  # Quantization-based compression
    ENTROPY = "entropy"  # Entropy coding
    

class QuantizationType(Enum):
    """Quantization methods for deltas."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"  # 4-bit NormalFloat
    FP8 = "fp8"  # 8-bit floating point
    BINARY = "binary"  # Binary weights
    TERNARY = "ternary"  # {-1, 0, 1}


@dataclass
class DeltaMetadata:
    """Metadata for a delta block."""
    # Identification
    delta_id: str
    layer_name: str
    model_name: str
    model_version: str
    
    # Delta properties
    delta_type: DeltaType
    compression_type: CompressionType
    quantization_type: QuantizationType
    
    # Training context
    training_round_id: str
    base_hash: str  # Hash of base parameters
    parent_delta_hash: Optional[str] = None  # For delta chains
    
    # Size and efficiency
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 0.0
    sparsity_ratio: float = 0.0  # Percentage of zeros
    
    # LoRA-specific
    lora_rank: Optional[int] = None
    lora_alpha: Optional[float] = None
    lora_dropout: Optional[float] = None
    target_modules: Optional[List[str]] = None
    
    # Quality metrics
    loss_before: Optional[float] = None
    loss_after: Optional[float] = None
    perplexity_delta: Optional[float] = None
    validation_score: Optional[float] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    validated_at: Optional[datetime] = None
    
    # Signatures
    trainer_signature: Optional[str] = None
    validator_signatures: List[str] = field(default_factory=list)
    
    # Additional metadata
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeltaTensor:
    """Container for delta tensor data with metadata."""
    # Tensor properties
    shape: List[int]
    dtype: str  # "float32", "float16", "int8", etc.
    device: str = "cpu"
    
    # Data storage
    data: Optional[Union[bytes, np.ndarray, torch.Tensor]] = None
    indices: Optional[Union[bytes, np.ndarray]] = None  # For sparse
    values: Optional[Union[bytes, np.ndarray]] = None  # For sparse
    
    # LoRA components
    lora_A: Optional[Union[bytes, np.ndarray, torch.Tensor]] = None
    lora_B: Optional[Union[bytes, np.ndarray, torch.Tensor]] = None
    
    # Quantization parameters
    scale: Optional[float] = None
    zero_point: Optional[float] = None
    quantization_bits: Optional[int] = None
    
    # Compression info
    is_compressed: bool = False
    compression_method: Optional[str] = None
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor."""
        if isinstance(self.data, torch.Tensor):
            return self.data
        elif isinstance(self.data, np.ndarray):
            return torch.from_numpy(self.data)
        elif isinstance(self.data, bytes):
            # Decompress if needed
            if self.is_compressed:
                data = self._decompress(self.data)
            else:
                data = self.data
            # Deserialize
            arr = np.frombuffer(data, dtype=self.dtype).reshape(self.shape)
            return torch.from_numpy(arr)
        else:
            raise ValueError(f"Cannot convert {type(self.data)} to tensor")
    
    def _decompress(self, data: bytes) -> bytes:
        """Decompress data based on method."""
        if self.compression_method == "zlib":
            return zlib.decompress(data)
        elif self.compression_method == "gzip":
            import gzip
            return gzip.decompress(data)
        else:
            return data


@dataclass
class UniversalDelta:
    """Universal delta format for any model architecture."""
    # Core metadata
    metadata: DeltaMetadata
    
    # Delta tensors by name
    tensors: Dict[str, DeltaTensor] = field(default_factory=dict)
    
    # Validation
    checksum: Optional[str] = None
    signature: Optional[bytes] = None
    
    def add_dense_delta(self, name: str, delta: torch.Tensor):
        """Add a dense parameter delta."""
        self.tensors[name] = DeltaTensor(
            shape=list(delta.shape),
            dtype=str(delta.dtype).replace("torch.", ""),
            data=delta.cpu().numpy().tobytes()
        )
        self.metadata.delta_type = DeltaType.DENSE
    
    def add_lora_delta(self, name: str, lora_A: torch.Tensor, lora_B: torch.Tensor):
        """Add a LoRA delta."""
        self.tensors[name] = DeltaTensor(
            shape=list(lora_B.shape[0:1] + lora_A.shape[1:2]),  # Output shape
            dtype=str(lora_A.dtype).replace("torch.", ""),
            lora_A=lora_A.cpu().numpy().tobytes(),
            lora_B=lora_B.cpu().numpy().tobytes()
        )
        self.metadata.delta_type = DeltaType.LORA
        self.metadata.lora_rank = lora_A.shape[0]
    
    def add_sparse_delta(self, name: str, indices: torch.Tensor, values: torch.Tensor, shape: List[int]):
        """Add a sparse parameter delta."""
        self.tensors[name] = DeltaTensor(
            shape=shape,
            dtype=str(values.dtype).replace("torch.", ""),
            indices=indices.cpu().numpy().tobytes(),
            values=values.cpu().numpy().tobytes()
        )
        self.metadata.delta_type = DeltaType.SPARSE
        self.metadata.sparsity_ratio = 1.0 - (len(values) / np.prod(shape))
    
    def compress(self, method: CompressionType = CompressionType.ZLIB):
        """Compress all tensors."""
        original_size = 0
        compressed_size = 0
        
        for name, tensor in self.tensors.items():
            if tensor.data is not None and not tensor.is_compressed:
                original = tensor.data if isinstance(tensor.data, bytes) else tensor.data.tobytes()
                original_size += len(original)
                
                if method == CompressionType.ZLIB:
                    compressed = zlib.compress(original, level=6)
                    tensor.data = compressed
                    tensor.is_compressed = True
                    tensor.compression_method = "zlib"
                    compressed_size += len(compressed)
                elif method == CompressionType.QUANTIZED:
                    # Implement quantization compression
                    pass
        
        self.metadata.compression_type = method
        self.metadata.original_size_bytes = original_size
        self.metadata.compressed_size_bytes = compressed_size
        self.metadata.compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    def compute_checksum(self) -> str:
        """Compute SHA256 checksum of delta."""
        hasher = hashlib.sha256()
        
        # Hash metadata
        meta_json = json.dumps({
            "delta_id": self.metadata.delta_id,
            "layer_name": self.metadata.layer_name,
            "model_name": self.metadata.model_name,
            "delta_type": self.metadata.delta_type.value,
            "base_hash": self.metadata.base_hash
        }, sort_keys=True)
        hasher.update(meta_json.encode())
        
        # Hash tensors
        for name in sorted(self.tensors.keys()):
            tensor = self.tensors[name]
            hasher.update(name.encode())
            
            if tensor.data is not None:
                data = tensor.data if isinstance(tensor.data, bytes) else tensor.data.tobytes()
                hasher.update(data)
            
            if tensor.lora_A is not None:
                data = tensor.lora_A if isinstance(tensor.lora_A, bytes) else tensor.lora_A.tobytes()
                hasher.update(data)
            
            if tensor.lora_B is not None:
                data = tensor.lora_B if isinstance(tensor.lora_B, bytes) else tensor.lora_B.tobytes()
                hasher.update(data)
        
        self.checksum = hasher.hexdigest()
        return self.checksum
    
    def to_bytes(self) -> bytes:
        """Serialize delta to bytes for blockchain storage."""
        return pickle.dumps(self)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'UniversalDelta':
        """Deserialize delta from bytes."""
        return pickle.loads(data)
    
    def apply_to_parameters(self, base_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply delta to base parameters."""
        updated = base_params.copy()
        
        for name, tensor in self.tensors.items():
            if name not in base_params:
                continue
            
            base = base_params[name]
            
            if self.metadata.delta_type == DeltaType.DENSE:
                # Direct addition
                delta = tensor.to_tensor()
                updated[name] = base + delta
                
            elif self.metadata.delta_type == DeltaType.LORA:
                # LoRA update: W' = W + BA
                if tensor.lora_A is not None and tensor.lora_B is not None:
                    A = torch.from_numpy(np.frombuffer(tensor.lora_A, dtype=tensor.dtype).reshape(-1, base.shape[1]))
                    B = torch.from_numpy(np.frombuffer(tensor.lora_B, dtype=tensor.dtype).reshape(base.shape[0], -1))
                    lora_delta = B @ A * (self.metadata.lora_alpha / self.metadata.lora_rank)
                    updated[name] = base + lora_delta
                    
            elif self.metadata.delta_type == DeltaType.SPARSE:
                # Sparse update
                if tensor.indices is not None and tensor.values is not None:
                    indices = np.frombuffer(tensor.indices, dtype=np.int64)
                    values = np.frombuffer(tensor.values, dtype=tensor.dtype)
                    
                    # Apply sparse updates
                    flat_base = base.flatten()
                    flat_base[indices] += torch.from_numpy(values)
                    updated[name] = flat_base.reshape(base.shape)
        
        return updated
    
    def validate(self) -> bool:
        """Validate delta integrity."""
        # Check checksum
        if self.checksum:
            computed = self.compute_checksum()
            if computed != self.checksum:
                return False
        
        # Validate tensor shapes
        for name, tensor in self.tensors.items():
            if self.metadata.delta_type == DeltaType.LORA:
                if tensor.lora_A is None or tensor.lora_B is None:
                    return False
            elif self.metadata.delta_type == DeltaType.SPARSE:
                if tensor.indices is None or tensor.values is None:
                    return False
            elif tensor.data is None:
                return False
        
        return True


class DeltaFormatManager:
    """Manager for creating and handling universal deltas."""
    
    @staticmethod
    def create_delta(
        layer_name: str,
        model_name: str,
        base_hash: str,
        training_round_id: str,
        delta_type: DeltaType = DeltaType.DENSE
    ) -> UniversalDelta:
        """Create a new universal delta."""
        import uuid
        
        metadata = DeltaMetadata(
            delta_id=str(uuid.uuid4()),
            layer_name=layer_name,
            model_name=model_name,
            model_version="latest",
            delta_type=delta_type,
            compression_type=CompressionType.NONE,
            quantization_type=QuantizationType.NONE,
            training_round_id=training_round_id,
            base_hash=base_hash
        )
        
        return UniversalDelta(metadata=metadata)
    
    @staticmethod
    def optimize_for_blockchain(delta: UniversalDelta, max_size_mb: float = 10) -> UniversalDelta:
        """Optimize delta for blockchain storage."""
        current_size = sum(
            len(t.data) if t.data and isinstance(t.data, bytes) else 0
            for t in delta.tensors.values()
        )
        
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if current_size > max_size_bytes:
            # Try compression first
            delta.compress(CompressionType.ZLIB)
            
            # If still too large, consider quantization or sparsification
            if delta.metadata.compressed_size_bytes > max_size_bytes:
                # Could implement automatic sparsification here
                pass
        
        # Compute checksum for integrity
        delta.compute_checksum()
        
        return delta
    
    @staticmethod
    def merge_deltas(deltas: List[UniversalDelta]) -> UniversalDelta:
        """Merge multiple deltas into one."""
        if not deltas:
            raise ValueError("No deltas to merge")
        
        # Use first delta as base
        merged = deltas[0]
        
        # Merge subsequent deltas
        for delta in deltas[1:]:
            for name, tensor in delta.tensors.items():
                if name in merged.tensors:
                    # Combine tensors (implementation depends on delta type)
                    if merged.metadata.delta_type == DeltaType.DENSE:
                        # Add dense deltas
                        base_data = merged.tensors[name].to_tensor()
                        delta_data = tensor.to_tensor()
                        combined = base_data + delta_data
                        merged.tensors[name].data = combined.cpu().numpy().tobytes()
                else:
                    merged.tensors[name] = tensor
        
        # Update metadata
        merged.metadata.delta_id = str(uuid.uuid4())
        merged.metadata.parent_delta_hash = deltas[-1].checksum
        merged.compute_checksum()
        
        return merged