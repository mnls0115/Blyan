"""
Blyan Evolution System - Pluggable Block Loader System
Handles different payload types with extensible plugin architecture
"""

from __future__ import annotations
import json
import pickle
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, Protocol, runtime_checkable
from abc import ABC, abstractmethod

from .block import Block, BlockHeader
from .meta_v2 import MetaSpecV2


@runtime_checkable
class PayloadLoader(Protocol):
    """Protocol for payload loader plugins"""
    
    def can_load(self, block: Block) -> bool:
        """Check if this loader can handle the block"""
        ...
    
    def load(self, block: Block) -> Any:
        """Load the payload from the block"""
        ...
    
    def get_supported_types(self) -> list[str]:
        """Get list of supported payload types"""
        ...


class BasePayloadLoader(ABC):
    """Base class for payload loaders"""
    
    @abstractmethod
    def can_load(self, block: Block) -> bool:
        """Check if this loader can handle the block"""
        pass
    
    @abstractmethod
    def load(self, block: Block) -> Any:
        """Load the payload from the block"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> list[str]:
        """Get list of supported payload types"""
        pass


class PicklePayloadLoader(BasePayloadLoader):
    """Legacy pickle payload loader"""
    
    def can_load(self, block: Block) -> bool:
        return block.header.payload_type in [None, "pickle", "state_dict"]
    
    def load(self, block: Block) -> Any:
        try:
            return pickle.loads(block.payload)
        except Exception as e:
            raise ValueError(f"Failed to load pickle payload: {e}")
    
    def get_supported_types(self) -> list[str]:
        return ["pickle", "state_dict"]


class JSONPayloadLoader(BasePayloadLoader):
    """JSON payload loader for metadata blocks"""
    
    def can_load(self, block: Block) -> bool:
        return block.header.payload_type == "json"
    
    def load(self, block: Block) -> Dict[str, Any]:
        try:
            return json.loads(block.payload.decode('utf-8'))
        except Exception as e:
            raise ValueError(f"Failed to load JSON payload: {e}")
    
    def get_supported_types(self) -> list[str]:
        return ["json"]


class TensorBlockLoader(BasePayloadLoader):
    """Zero-copy TensorBlock loader"""
    
    def can_load(self, block: Block) -> bool:
        return block.header.payload_type == "tensorblock"
    
    def load(self, block: Block) -> torch.Tensor:
        """Load tensor using zero-copy TensorBlock format."""
        import tempfile
        from backend.core.tensorblock import TensorBlockReader
        
        try:
            # Write block payload to temporary file for memory mapping
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tblock') as tmp:
                tmp.write(block.payload)
                tmp_path = tmp.name
            
            # Use TensorBlockReader for zero-copy loading
            with TensorBlockReader(tmp_path) as reader:
                tensor = reader.load_tensor_zero_copy(device="cpu")
            
            # Clean up temp file
            Path(tmp_path).unlink()
            
            return tensor
        except Exception as e:
            raise ValueError(f"Failed to load TensorBlock payload: {e}")
    
    def get_supported_types(self) -> list[str]:
        return ["tensorblock"]


class EEBLoader(BasePayloadLoader):
    """Executable Expert Block loader for TensorRT/ONNX engines"""
    
    def can_load(self, block: Block) -> bool:
        return block.header.payload_type == "eeb"
    
    def load(self, block: Block) -> Any:
        # TODO: Implement EEB engine loading
        # This would deserialize TensorRT or ONNX Runtime engines
        try:
            # Placeholder implementation
            # In practice, would use:
            # - tensorrt.Runtime().deserialize_cuda_engine() for TensorRT
            # - onnxruntime.InferenceSession() for ONNX
            engine_data = {
                "type": "tensorrt_engine",
                "architecture": block.header.architecture,
                "engine_size": len(block.payload),
                "ready": True
            }
            return engine_data
        except Exception as e:
            raise ValueError(f"Failed to load EEB payload: {e}")
    
    def get_supported_types(self) -> list[str]:
        return ["eeb"]


class TileStreamLoader(BasePayloadLoader):
    """Tile-streaming loader for out-of-core computation"""
    
    def can_load(self, block: Block) -> bool:
        return block.header.payload_type == "tile_stream"
    
    def load(self, block: Block) -> 'TileStreamer':
        # TODO: Implement tile streaming system
        try:
            # This would return a streamer object that can load tiles on demand
            return TileStreamer(block)
        except Exception as e:
            raise ValueError(f"Failed to create tile streamer: {e}")
    
    def get_supported_types(self) -> list[str]:
        return ["tile_stream"]


class CodePayloadLoader(BasePayloadLoader):
    """Code block loader for executable evolution"""
    
    def can_load(self, block: Block) -> bool:
        return block.header.payload_type == "code"
    
    def load(self, block: Block) -> Dict[str, Any]:
        try:
            code_data = json.loads(block.payload.decode('utf-8'))
            
            # Validate code block structure
            required_fields = ['code', 'language', 'type']
            for field in required_fields:
                if field not in code_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Add metadata from header
            code_data.update({
                'code_type': block.header.code_type,
                'target_expert': block.header.target_expert,
                'execution_environment': block.header.execution_environment,
                'dependencies': block.header.dependencies or []
            })
            
            return code_data
        except Exception as e:
            raise ValueError(f"Failed to load code payload: {e}")
    
    def get_supported_types(self) -> list[str]:
        return ["code"]


class MigrationPayloadLoader(BasePayloadLoader):
    """Migration block loader"""
    
    def can_load(self, block: Block) -> bool:
        return (block.header.block_type == "migration" or 
                block.header.payload_type == "migration")
    
    def load(self, block: Block) -> Dict[str, Any]:
        try:
            from .migration import MigrationBlock
            migration_data = json.loads(block.payload.decode('utf-8'))
            migration = MigrationBlock.from_dict(migration_data)
            return migration
        except Exception as e:
            raise ValueError(f"Failed to load migration payload: {e}")
    
    def get_supported_types(self) -> list[str]:
        return ["migration"]


class TileStreamer:
    """Placeholder for tile streaming system"""
    
    def __init__(self, block: Block):
        self.block = block
        self.tile_index = self._parse_tile_index()
    
    def _parse_tile_index(self) -> Dict[str, Any]:
        # TODO: Parse tile index from block payload
        return {"total_tiles": 0, "tile_size": [256, 256]}
    
    async def stream_tile(self, tile_id: int) -> torch.Tensor:
        # TODO: Implement async tile streaming
        return torch.zeros(256, 256)


class BlockLoaderRegistry:
    """Registry for payload loader plugins"""
    
    def __init__(self):
        self._loaders: list[PayloadLoader] = []
        self._register_default_loaders()
    
    def _register_default_loaders(self):
        """Register default payload loaders"""
        self.register(PicklePayloadLoader())
        self.register(JSONPayloadLoader())
        self.register(TensorBlockLoader())
        self.register(EEBLoader())
        self.register(TileStreamLoader())
        self.register(CodePayloadLoader())
        self.register(MigrationPayloadLoader())
    
    def register(self, loader: PayloadLoader):
        """Register a new payload loader"""
        self._loaders.append(loader)
    
    def get_loader(self, block: Block) -> Optional[PayloadLoader]:
        """Get appropriate loader for a block"""
        for loader in self._loaders:
            if loader.can_load(block):
                return loader
        return None
    
    def load_payload(self, block: Block) -> Any:
        """Load payload using appropriate loader"""
        loader = self.get_loader(block)
        if loader is None:
            raise ValueError(f"No loader found for payload type: {block.header.payload_type}")
        
        return loader.load(block)
    
    def get_supported_types(self) -> Dict[str, list[str]]:
        """Get all supported payload types by loader"""
        types_by_loader = {}
        for loader in self._loaders:
            loader_name = loader.__class__.__name__
            types_by_loader[loader_name] = loader.get_supported_types()
        return types_by_loader


class EvolutionAwareBlockLoader:
    """Enhanced block loader with evolution support"""
    
    def __init__(self, registry: Optional[BlockLoaderRegistry] = None):
        self.registry = registry or BlockLoaderRegistry()
        self._version_cache: Dict[str, Any] = {}
    
    def load_block(self, block: Block) -> Any:
        """Load block with evolution context"""
        payload = self.registry.load_payload(block)
        
        # Add evolution metadata if available
        if block.header.is_evolution_block():
            if isinstance(payload, dict):
                payload['_evolution_metadata'] = {
                    'version': block.header.version,
                    'evolution_type': block.header.evolution_type,
                    'parent_hash': block.header.parent_hash,
                    'dimension_changes': block.header.dimension_changes,
                    'compatibility_range': block.header.compatibility_range
                }
        
        return payload
    
    def load_expert_with_version(self, block: Block, target_version: str) -> Any:
        """Load expert with version compatibility checking"""
        if not block.header.is_compatible_with_version(target_version):
            raise ValueError(f"Block version {block.header.version} not compatible with {target_version}")
        
        return self.load_block(block)
    
    def load_migration_chain(self, blocks: list[Block]) -> list[Any]:
        """Load a chain of migration blocks"""
        migrations = []
        for block in blocks:
            if not block.header.is_migration_block():
                raise ValueError(f"Block {block.compute_hash()[:8]} is not a migration block")
            
            migration = self.load_block(block)
            migrations.append(migration)
        
        return migrations
    
    def get_evolution_tree(self, blocks: list[Block]) -> Dict[str, list[str]]:
        """Build evolution tree from blocks"""
        tree = {}
        
        for block in blocks:
            if block.header.is_evolution_block():
                version = block.header.version or "unknown"
                parent = block.header.parent_hash
                
                if parent:
                    if parent not in tree:
                        tree[parent] = []
                    tree[parent].append(block.compute_hash())
        
        return tree
    
    def find_compatible_blocks(self, blocks: list[Block], target_version: str) -> list[Block]:
        """Find all blocks compatible with target version"""
        compatible = []
        
        for block in blocks:
            if block.header.is_compatible_with_version(target_version):
                compatible.append(block)
        
        return compatible


# Global registry instance
_global_registry = BlockLoaderRegistry()


def load_block_payload(block: Block) -> Any:
    """Convenience function to load block payload"""
    return _global_registry.load_payload(block)


def register_payload_loader(loader: PayloadLoader):
    """Register a custom payload loader"""
    _global_registry.register(loader)


def get_supported_payload_types() -> Dict[str, list[str]]:
    """Get all supported payload types"""
    return _global_registry.get_supported_types()


# TODO: Implement additional specialized loaders
class MetaSpecLoader(BasePayloadLoader):
    """Loader for MetaSpec v2 blocks"""
    
    def can_load(self, block: Block) -> bool:
        return (block.header.block_type == "meta" and 
                block.header.payload_type in ["json", None])
    
    def load(self, block: Block) -> MetaSpecV2:
        try:
            if block.header.payload_type == "json":
                data = json.loads(block.payload.decode('utf-8'))
            else:
                # Legacy format
                data = pickle.loads(block.payload)
            
            # Convert to MetaSpecV2 if needed
            if isinstance(data, dict):
                return MetaSpecV2.from_dict(data)
            else:
                # Legacy MetaSpec - need conversion logic
                return self._convert_legacy_spec(data)
        except Exception as e:
            raise ValueError(f"Failed to load MetaSpec: {e}")
    
    def _convert_legacy_spec(self, legacy_spec: Any) -> MetaSpecV2:
        """Convert legacy meta spec to v2 format"""
        # TODO: Implement legacy conversion
        return MetaSpecV2(
            model_id="legacy_model",
            version="1.0.0",
            model_name=getattr(legacy_spec, 'model_name', 'unknown')
        )
    
    def get_supported_types(self) -> list[str]:
        return ["meta_spec_v2"]


# Register the meta spec loader
_global_registry.register(MetaSpecLoader())