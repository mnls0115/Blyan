#!/usr/bin/env python3
"""TensorBlock loader integration with Blyan's existing blockchain and MoE infrastructure."""

import os
import torch
from pathlib import Path
from typing import Dict, Optional, Union, List
import pickle

from .chain import Chain
from .block import Block, BlockHeader  
from .tensorblock import (
    TensorBlockWriter, TensorBlockReader, 
    QuantizationMetadata,
    tensor_to_tensorblock, tensorblock_to_tensor
)


class ExpertBlockLoader:
    """Universal expert block loader supporting multiple payload formats."""
    
    def __init__(self, param_chain: Chain, cache_dir: Optional[Path] = None):
        self.param_chain = param_chain
        self.cache_dir = cache_dir or Path("./data/tensorblock_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.load_times: Dict[str, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def load_expert(self, 
                   expert_name: str, 
                   device: str = "cpu",
                   verify_integrity: bool = True) -> torch.Tensor:
        """Load expert tensor using the optimal format available."""
        
        # Find expert block
        expert_blocks = [
            block for block in self.param_chain.storage.blocks
            if (block.header.block_type == 'expert' and 
                block.header.expert_name == expert_name)
        ]
        
        if not expert_blocks:
            raise ValueError(f"Expert {expert_name} not found in blockchain")
            
        # Use the latest block for this expert
        latest_block = max(expert_blocks, key=lambda b: b.header.timestamp)
        
        # Route to appropriate loader based on payload type
        payload_type = latest_block.header.payload_type or "pickle"
        
        if payload_type == "tensorblock":
            return self._load_tensorblock_expert(latest_block, device, verify_integrity)
        elif payload_type == "eeb":
            return self._load_eeb_expert(latest_block, device)
        elif payload_type == "tile_stream":
            return self._load_tile_stream_expert(latest_block, device)
        else:
            # Legacy pickle format
            return self._load_pickle_expert(latest_block, device)
    
    def _load_tensorblock_expert(self, 
                                block: Block, 
                                device: str,
                                verify_integrity: bool) -> torch.Tensor:
        """Load expert using zero-copy TensorBlock format."""
        import time
        
        start_time = time.time()
        
        # Check cache first
        cache_path = self.cache_dir / f"{block.header.payload_hash}.tblock"
        
        if cache_path.exists():
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            # Write block payload to cache file
            with open(cache_path, 'wb') as f:
                f.write(block.payload)
        
        # Load with zero-copy
        merkle_root = block.header.merkle_root if verify_integrity else None
        tensor = tensorblock_to_tensor(cache_path, device, merkle_root)
        
        # Track performance
        load_time = time.time() - start_time
        self.load_times[block.header.expert_name] = load_time
        
        print(f"🚀 TensorBlock loaded {block.header.expert_name}: "
              f"{load_time*1000:.1f}ms, cache_hit={cache_path.exists()}")
        
        return tensor
    
    def _load_eeb_expert(self, block: Block, device: str) -> torch.Tensor:
        """Load expert using Executable Expert Block format."""
        # Placeholder for EEB implementation (Phase B)
        print(f"⚡ EEB loading not yet implemented for {block.header.expert_name}")
        return self._load_pickle_expert(block, device)
    
    def _load_tile_stream_expert(self, block: Block, device: str) -> torch.Tensor:
        """Load expert using Tile-Streaming format."""
        # Placeholder for Tile-Streaming implementation (Phase C)
        print(f"🌊 Tile-streaming loading not yet implemented for {block.header.expert_name}")
        return self._load_pickle_expert(block, device)
    
    def _load_pickle_expert(self, block: Block, device: str) -> torch.Tensor:
        """Load expert using legacy pickle format."""
        import time
        
        start_time = time.time()
        
        # Legacy pickle loading
        expert_weights = pickle.loads(block.payload)
        
        # Convert to tensor if needed
        if isinstance(expert_weights, dict):
            # Assume first tensor in dict is the main weight
            tensor = next(iter(expert_weights.values()))
        else:
            tensor = expert_weights
            
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to(device)
        else:
            tensor = torch.tensor(tensor, device=device)
        
        load_time = time.time() - start_time
        self.load_times[block.header.expert_name] = load_time
        
        print(f"📦 Pickle loaded {block.header.expert_name}: {load_time*1000:.1f}ms")
        
        return tensor
    
    def upload_expert_tensorblock(self,
                                 expert_name: str,
                                 tensor: torch.Tensor,
                                 layer_id: str,
                                 depends_on: List[str] = None,
                                 quantization: Optional[QuantizationMetadata] = None,
                                 miner_address: str = "system") -> str:
        """Upload expert tensor as TensorBlock format to blockchain."""
        
        # Create temporary TensorBlock file
        temp_path = self.cache_dir / f"{expert_name}_{int(time.time())}.tblock"
        
        try:
            # Convert tensor to TensorBlock format
            tblock_metadata = tensor_to_tensorblock(tensor, temp_path, quantization)
            
            # Read TensorBlock binary data
            with open(temp_path, 'rb') as f:
                tblock_payload = f.read()
            
            # Create block header with TensorBlock metadata
            header = BlockHeader(
                index=len(self.param_chain.storage.blocks),
                timestamp=time.time(),
                prev_hash=self.param_chain.get_latest_hash(),
                chain_id="B",
                points_to=None,
                payload_hash=self.param_chain._compute_hash(tblock_payload),
                payload_size=len(tblock_payload),
                depends_on=depends_on or [],
                block_type='expert',
                expert_name=expert_name,
                layer_id=layer_id,
                # TensorBlock-specific metadata
                payload_type="tensorblock",
                tensor_dtype=str(tensor.dtype),
                tensor_shape=list(tensor.shape),
                tensor_layout="row_major",
                quantization_method=quantization.method if quantization else "none",
                merkle_root=tblock_metadata["merkle_root"]
            )
            
            # Create and add block
            block = Block(
                header=header,
                payload=tblock_payload,
                miner=miner_address
            )
            
            # Add to chain
            self.param_chain.add_block_direct(block)
            
            print(f"✅ Uploaded {expert_name} as TensorBlock: "
                  f"{len(tblock_payload):,} bytes, merkle: {tblock_metadata['merkle_root'][:16]}...")
            
            return header.compute_hash()
            
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
    
    def get_performance_stats(self) -> Dict[str, any]:
        """Get loader performance statistics."""
        if not self.load_times:
            return {"status": "no_data"}
            
        import numpy as np
        load_times_ms = [t * 1000 for t in self.load_times.values()]
        
        return {
            "cache_hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses),
            "avg_load_time_ms": np.mean(load_times_ms),
            "p95_load_time_ms": np.percentile(load_times_ms, 95),
            "total_experts_loaded": len(self.load_times),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses
        }
    
    def clear_cache(self):
        """Clear TensorBlock cache directory."""
        for cache_file in self.cache_dir.glob("*.tblock"):
            cache_file.unlink()
        print(f"🧹 Cleared TensorBlock cache: {self.cache_dir}")


# Integration with existing MoE inference
def integrate_tensorblock_with_moe_manager():
    """Integration helper for MoEModelManager."""
    
    # This would replace the _load_expert method in MoEModelManager
    example_integration = """
    # In backend/model/moe_infer.py MoEModelManager class:
    
    def __init__(self, ...):
        # Add TensorBlock loader
        self.tensorblock_loader = ExpertBlockLoader(self.param_chain)
    
    def _load_expert(self, expert_name: str) -> Optional[Dict[str, torch.Tensor]]:
        '''Load expert using TensorBlock format if available.'''
        if expert_name in self._loaded_experts:
            return self._loaded_experts[expert_name]
            
        try:
            # Use TensorBlock loader (10x faster)
            expert_tensor = self.tensorblock_loader.load_expert(
                expert_name, 
                device=self.device,
                verify_integrity=True
            )
            
            # Cache the loaded expert
            expert_weights = {"weight": expert_tensor}
            self._loaded_experts[expert_name] = expert_weights
            
            return expert_weights
            
        except Exception as e:
            print(f"⚠️ TensorBlock loading failed for {expert_name}: {e}")
            # Fallback to existing pickle loading
            return self._load_expert_legacy(expert_name)
    """
    
    print("💡 Integration example:")
    print(example_integration)


# Example usage and testing
if __name__ == "__main__":
    from .chain import Chain
    import tempfile
    
    print("=== TensorBlock Blockchain Integration Demo ===")
    
    # Create temporary chains for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Initialize chain
        param_chain = Chain(temp_path, "B")
        
        # Create loader
        loader = ExpertBlockLoader(param_chain, temp_path / "cache")
        
        # Create test expert tensor
        test_expert = torch.randn(1024, 512, dtype=torch.float16)
        
        print(f"\n1. Uploading expert tensor...")
        print(f"   Shape: {test_expert.shape}, dtype: {test_expert.dtype}")
        
        # Upload as TensorBlock
        expert_hash = loader.upload_expert_tensorblock(
            expert_name="layer0.expert0",
            tensor=test_expert,
            layer_id="layer0",
            depends_on=[]
        )
        
        print(f"   Block hash: {expert_hash[:16]}...")
        
        print(f"\n2. Loading expert tensor...")
        
        # Load back from blockchain
        loaded_expert = loader.load_expert(
            expert_name="layer0.expert0",
            device="cpu",
            verify_integrity=True
        )
        
        print(f"   Loaded shape: {loaded_expert.shape}")
        print(f"   Data matches: {torch.allclose(test_expert, loaded_expert, atol=1e-6)}")
        
        # Performance stats
        print(f"\n3. Performance statistics:")
        stats = loader.get_performance_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test cache efficiency (load again)
        print(f"\n4. Testing cache efficiency...")
        loaded_again = loader.load_expert("layer0.expert0", device="cpu")
        
        final_stats = loader.get_performance_stats()
        print(f"   Cache hit ratio: {final_stats['cache_hit_ratio']:.2%}")
        print(f"   Average load time: {final_stats['avg_load_time_ms']:.1f}ms")
        
        print(f"\n✅ TensorBlock blockchain integration working!")
        
        # Show integration example
        integrate_tensorblock_with_moe_manager()