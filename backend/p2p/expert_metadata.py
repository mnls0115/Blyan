"""
Expert Metadata System for Efficient P2P Registration
======================================================
Production-grade metadata system to avoid sending thousands of expert names.
Uses cryptographic hashes and layer-based metadata for verification.
"""

import hashlib
import json
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum


class ExpertCoverage(Enum):
    """Expert coverage types for a node."""
    FULL = "full"  # All experts for all layers
    PARTIAL = "partial"  # Some experts missing
    LAYER_COMPLETE = "layer_complete"  # Complete layers but not all layers
    SPARSE = "sparse"  # Few experts scattered across layers


@dataclass
class LayerMetadata:
    """Metadata for experts in a single layer."""
    layer_id: int
    expert_count: int
    expert_range: tuple  # (min_expert_id, max_expert_id)
    is_complete: bool  # True if has all experts for this layer
    
    def to_dict(self) -> Dict:
        return {
            "layer_id": self.layer_id,
            "expert_count": self.expert_count,
            "expert_range": list(self.expert_range),
            "is_complete": self.is_complete
        }


@dataclass 
class ExpertMetadata:
    """Complete metadata for a node's expert coverage."""
    node_id: str
    total_experts: int
    total_layers: int
    coverage_type: ExpertCoverage
    layer_metadata: List[LayerMetadata]
    experts_hash: str  # SHA256 hash of sorted expert list
    model_name: str
    model_version: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "total_experts": self.total_experts,
            "total_layers": self.total_layers,
            "coverage_type": self.coverage_type.value,
            "layer_metadata": [lm.to_dict() for lm in self.layer_metadata],
            "experts_hash": self.experts_hash,
            "model_name": self.model_name,
            "model_version": self.model_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExpertMetadata':
        """Reconstruct from dictionary."""
        return cls(
            node_id=data["node_id"],
            total_experts=data["total_experts"],
            total_layers=data["total_layers"],
            coverage_type=ExpertCoverage(data["coverage_type"]),
            layer_metadata=[
                LayerMetadata(
                    layer_id=lm["layer_id"],
                    expert_count=lm["expert_count"],
                    expert_range=tuple(lm["expert_range"]),
                    is_complete=lm["is_complete"]
                )
                for lm in data["layer_metadata"]
            ],
            experts_hash=data["experts_hash"],
            model_name=data["model_name"],
            model_version=data.get("model_version")
        )
    
    def has_expert(self, expert_name: str) -> bool:
        """Check if this node has a specific expert based on metadata."""
        try:
            # Parse expert name (e.g., "layer0.expert5")
            parts = expert_name.split('.')
            if len(parts) != 2:
                return False
            
            layer_str, expert_str = parts
            if not layer_str.startswith('layer') or not expert_str.startswith('expert'):
                return False
            
            layer_id = int(layer_str[5:])
            expert_id = int(expert_str[6:])
            
            # Check if we have this layer
            for lm in self.layer_metadata:
                if lm.layer_id == layer_id:
                    # Check if expert is in range
                    min_id, max_id = lm.expert_range
                    return min_id <= expert_id <= max_id
            
            return False
            
        except (ValueError, IndexError):
            return False
    
    def get_available_experts_for_layer(self, layer_id: int) -> List[str]:
        """Get list of available experts for a specific layer."""
        for lm in self.layer_metadata:
            if lm.layer_id == layer_id:
                min_id, max_id = lm.expert_range
                return [f"layer{layer_id}.expert{i}" for i in range(min_id, max_id + 1)]
        return []


class ExpertMetadataGenerator:
    """Generate metadata from expert lists."""
    
    @staticmethod
    def generate(
        experts: List[str], 
        node_id: str, 
        model_name: str,
        expected_layers: int = 48,
        expected_experts_per_layer: int = 128
    ) -> ExpertMetadata:
        """Generate metadata from a list of expert names."""
        
        # Group experts by layer
        layer_experts: Dict[int, Set[int]] = {}
        
        for expert_name in experts:
            try:
                parts = expert_name.split('.')
                if len(parts) == 2:
                    layer_id = int(parts[0].replace('layer', ''))
                    expert_id = int(parts[1].replace('expert', ''))
                    
                    if layer_id not in layer_experts:
                        layer_experts[layer_id] = set()
                    layer_experts[layer_id].add(expert_id)
            except (ValueError, IndexError):
                continue
        
        # Generate layer metadata
        layer_metadata = []
        complete_layers = 0
        
        for layer_id in sorted(layer_experts.keys()):
            expert_ids = sorted(layer_experts[layer_id])
            is_complete = len(expert_ids) == expected_experts_per_layer
            
            if is_complete:
                complete_layers += 1
            
            layer_metadata.append(LayerMetadata(
                layer_id=layer_id,
                expert_count=len(expert_ids),
                expert_range=(min(expert_ids), max(expert_ids)),
                is_complete=is_complete
            ))
        
        # Determine coverage type
        total_experts = sum(len(experts) for experts in layer_experts.values())
        
        if complete_layers == expected_layers and total_experts == expected_layers * expected_experts_per_layer:
            coverage_type = ExpertCoverage.FULL
        elif complete_layers > 0:
            coverage_type = ExpertCoverage.LAYER_COMPLETE
        elif total_experts > expected_layers * 10:  # More than 10 experts per layer average
            coverage_type = ExpertCoverage.PARTIAL
        else:
            coverage_type = ExpertCoverage.SPARSE
        
        # Generate hash of sorted expert list
        sorted_experts = sorted(experts)
        experts_str = ','.join(sorted_experts)
        experts_hash = hashlib.sha256(experts_str.encode()).hexdigest()
        
        return ExpertMetadata(
            node_id=node_id,
            total_experts=total_experts,
            total_layers=len(layer_experts),
            coverage_type=coverage_type,
            layer_metadata=layer_metadata,
            experts_hash=experts_hash,
            model_name=model_name,
            model_version=None
        )
    
    @staticmethod
    def generate_full_coverage(
        node_id: str,
        model_name: str,
        num_layers: int = 48,
        num_experts_per_layer: int = 128
    ) -> ExpertMetadata:
        """Generate metadata for a node with full expert coverage."""
        
        # Generate all expert names
        all_experts = []
        layer_metadata = []
        
        for layer_id in range(num_layers):
            layer_metadata.append(LayerMetadata(
                layer_id=layer_id,
                expert_count=num_experts_per_layer,
                expert_range=(0, num_experts_per_layer - 1),
                is_complete=True
            ))
            
            for expert_id in range(num_experts_per_layer):
                all_experts.append(f"layer{layer_id}.expert{expert_id}")
        
        # Generate hash
        experts_str = ','.join(all_experts)
        experts_hash = hashlib.sha256(experts_str.encode()).hexdigest()
        
        return ExpertMetadata(
            node_id=node_id,
            total_experts=num_layers * num_experts_per_layer,
            total_layers=num_layers,
            coverage_type=ExpertCoverage.FULL,
            layer_metadata=layer_metadata,
            experts_hash=experts_hash,
            model_name=model_name,
            model_version=None
        )


class ExpertQueryOptimizer:
    """Optimize expert queries using metadata."""
    
    @staticmethod
    def find_nodes_for_expert(
        expert_name: str,
        node_metadata: Dict[str, ExpertMetadata]
    ) -> List[str]:
        """Find all nodes that have a specific expert."""
        nodes = []
        
        for node_id, metadata in node_metadata.items():
            if metadata.has_expert(expert_name):
                nodes.append(node_id)
        
        return nodes
    
    @staticmethod
    def find_best_node_for_experts(
        required_experts: List[str],
        node_metadata: Dict[str, ExpertMetadata]
    ) -> Optional[str]:
        """Find the best single node that has all required experts."""
        
        for node_id, metadata in node_metadata.items():
            # Prefer nodes with full coverage
            if metadata.coverage_type == ExpertCoverage.FULL:
                return node_id
            
            # Check if node has all required experts
            has_all = True
            for expert in required_experts:
                if not metadata.has_expert(expert):
                    has_all = False
                    break
            
            if has_all:
                return node_id
        
        return None
    
    @staticmethod
    def verify_expert_availability(
        expert_name: str,
        metadata: ExpertMetadata,
        full_verification: bool = False
    ) -> bool:
        """Verify if an expert is available on a node.
        
        Args:
            expert_name: Name of the expert to check
            metadata: Node's expert metadata
            full_verification: If True, may trigger additional verification
        
        Returns:
            True if expert is available
        """
        # Quick check using metadata
        if not metadata.has_expert(expert_name):
            return False
        
        if full_verification:
            # In production, this could trigger an RPC call to verify
            # For now, trust the metadata
            pass
        
        return True