"""
Expert Sharding/Slicing Utilities
Splits large experts into smaller chunks for distributed storage
"""

import torch
import hashlib
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExpertSlice:
    """Represents a slice of an expert model"""
    expert_name: str
    slice_id: str  # Format: "expert_name#0/4"
    slice_index: int
    total_slices: int
    data: torch.Tensor
    size_mb: float
    hash: str
    
    def to_block_metadata(self) -> dict:
        """Convert slice to blockchain metadata"""
        return {
            "expert_name": self.expert_name,
            "slice_id": self.slice_id,
            "slice_index": self.slice_index,
            "total_slices": self.total_slices,
            "size_mb": self.size_mb,
            "hash": self.hash,
            "shape": list(self.data.shape),
            "dtype": str(self.data.dtype)
        }

class ShardUtils:
    """Utilities for sharding and reconstructing experts"""
    
    @staticmethod
    def calculate_optimal_shards(expert_size_mb: float, target_size_mb: float = 100) -> int:
        """Calculate optimal number of shards for an expert"""
        return max(1, int((expert_size_mb + target_size_mb - 1) // target_size_mb))
    
    @staticmethod
    def slice_expert(
        expert_name: str,
        weight_tensor: torch.Tensor,
        num_slices: Optional[int] = None,
        dim: int = 0
    ) -> List[ExpertSlice]:
        """
        Slice an expert weight tensor into smaller chunks
        
        Args:
            expert_name: Name of the expert (e.g., "layer0.expert1")
            weight_tensor: The weight tensor to slice
            num_slices: Number of slices (auto-calculated if None)
            dim: Dimension along which to slice (0 for row-shard)
        
        Returns:
            List of ExpertSlice objects
        """
        # Calculate size
        size_mb = weight_tensor.element_size() * weight_tensor.nelement() / (1024 * 1024)
        
        # Auto-calculate slices if not specified
        if num_slices is None:
            num_slices = ShardUtils.calculate_optimal_shards(size_mb)
        
        logger.info(f"Slicing expert {expert_name} ({size_mb:.2f}MB) into {num_slices} slices")
        
        # Use torch.chunk for slicing
        slices = torch.chunk(weight_tensor, num_slices, dim=dim)
        
        expert_slices = []
        for i, slice_tensor in enumerate(slices):
            slice_id = f"{expert_name}#{i}/{num_slices}"
            slice_size_mb = slice_tensor.element_size() * slice_tensor.nelement() / (1024 * 1024)
            
            # Calculate hash for integrity
            slice_bytes = slice_tensor.cpu().numpy().tobytes()
            slice_hash = hashlib.sha256(slice_bytes).hexdigest()[:16]
            
            expert_slice = ExpertSlice(
                expert_name=expert_name,
                slice_id=slice_id,
                slice_index=i,
                total_slices=num_slices,
                data=slice_tensor,
                size_mb=slice_size_mb,
                hash=slice_hash
            )
            
            expert_slices.append(expert_slice)
            logger.debug(f"Created slice {slice_id}: {slice_size_mb:.2f}MB, hash={slice_hash}")
        
        return expert_slices
    
    @staticmethod
    def reconstruct_expert(
        slices: List[ExpertSlice],
        dim: int = 0
    ) -> Tuple[str, torch.Tensor]:
        """
        Reconstruct an expert from its slices
        
        Args:
            slices: List of ExpertSlice objects
            dim: Dimension along which slices were created
        
        Returns:
            Tuple of (expert_name, reconstructed_tensor)
        """
        if not slices:
            raise ValueError("No slices provided")
        
        # Sort slices by index
        slices = sorted(slices, key=lambda s: s.slice_index)
        
        # Verify we have all slices
        expert_name = slices[0].expert_name
        total_slices = slices[0].total_slices
        
        if len(slices) != total_slices:
            raise ValueError(f"Missing slices: have {len(slices)}, need {total_slices}")
        
        # Verify all slices are from same expert
        for slice_obj in slices:
            if slice_obj.expert_name != expert_name:
                raise ValueError(f"Mixed experts: {expert_name} vs {slice_obj.expert_name}")
        
        # Concatenate tensors
        tensors = [s.data for s in slices]
        reconstructed = torch.cat(tensors, dim=dim)
        
        logger.info(f"Reconstructed expert {expert_name} from {len(slices)} slices")
        
        return expert_name, reconstructed
    
    @staticmethod
    def validate_slice_integrity(slice_obj: ExpertSlice) -> bool:
        """Validate slice data integrity using hash"""
        slice_bytes = slice_obj.data.cpu().numpy().tobytes()
        calculated_hash = hashlib.sha256(slice_bytes).hexdigest()[:16]
        
        is_valid = calculated_hash == slice_obj.hash
        if not is_valid:
            logger.warning(f"Slice {slice_obj.slice_id} integrity check failed: "
                          f"expected {slice_obj.hash}, got {calculated_hash}")
        
        return is_valid
    
    @staticmethod
    def get_slice_distribution_plan(
        expert_slices: List[ExpertSlice],
        available_nodes: List[Dict[str, any]]
    ) -> Dict[str, List[str]]:
        """
        Plan distribution of slices across nodes based on their capabilities
        
        Returns:
            Dict mapping node_id to list of slice_ids
        """
        distribution = {}
        
        # Sort nodes by available capacity (simplified)
        nodes = sorted(available_nodes, key=lambda n: n.get('available_vram_gb', 0), reverse=True)
        
        # Round-robin distribution with capacity check
        for i, slice_obj in enumerate(expert_slices):
            node_idx = i % len(nodes)
            node = nodes[node_idx]
            node_id = node['node_id']
            
            if node_id not in distribution:
                distribution[node_id] = []
            
            distribution[node_id].append(slice_obj.slice_id)
        
        logger.info(f"Distributed {len(expert_slices)} slices across {len(distribution)} nodes")
        
        return distribution


# Example usage and testing
if __name__ == "__main__":
    # Simulate a large expert weight
    large_expert = torch.randn(8192, 4096)  # ~128MB
    
    # Slice it
    shard_util = ShardUtils()
    slices = shard_util.slice_expert(
        expert_name="layer0.expert1",
        weight_tensor=large_expert,
        num_slices=4
    )
    
    print(f"Created {len(slices)} slices:")
    for s in slices:
        print(f"  {s.slice_id}: {s.size_mb:.2f}MB")
    
    # Test reconstruction
    expert_name, reconstructed = shard_util.reconstruct_expert(slices)
    
    # Verify reconstruction
    is_same = torch.allclose(large_expert, reconstructed)
    print(f"Reconstruction successful: {is_same}")
    
    # Test integrity
    for s in slices:
        is_valid = shard_util.validate_slice_integrity(s)
        print(f"Slice {s.slice_id} integrity: {is_valid}")