from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Optional, List, Literal, Dict, Any
from ..utils.json_canonical import dumps_canonical


@dataclass
class BlockHeader:
    """Metadata for a block stored on-chain."""

    index: int
    timestamp: float
    prev_hash: str
    chain_id: str  # "A" for meta, "B" for parameter, or other future chains
    points_to: Optional[str]  # hash in the sister chain this block is bound to
    payload_hash: str
    payload_size: int
    nonce: int = 0  # proof-of-work nonce
    
    # DAG and Dense Model extensions
    depends_on: List[str] = None  # list of block hashes this block depends on
    block_type: Literal['meta', 'layer', 'dense_layer', 'expert', 'router', 'code', 'architecture', 'migration', 'genesis_pact', 'dataset'] = 'meta'  # block type for dense model
    layer_name: Optional[str] = None  # layer identifier for dense model (e.g., "layer_0", "embedding", "lm_head")
    layer_id: Optional[str] = None  # layer identifier for model architecture
    
    # Legacy MoE fields for backward compatibility
    expert_name: Optional[str] = None  # Legacy MoE expert identifier
    target_expert: Optional[str] = None  # Legacy MoE target expert field
    
    # TensorBlock format extensions
    payload_type: Optional[str] = None  # "pickle", "tensorblock", "eeb", "tile_stream", "json", "code"
    tensor_dtype: Optional[str] = None  # "fp16", "int8", "fp8"
    tensor_shape: Optional[List[int]] = None  # tensor dimensions
    tensor_layout: Optional[str] = None  # "row_major", "col_major" 
    quantization_method: Optional[str] = None  # "none", "per_tensor_int8", "per_channel_int8"
    architecture: Optional[str] = None  # "sm_86", "sm_89", "sm_90" for EEB
    merkle_root: Optional[str] = None  # merkle root for integrity verification
    
    # Evolution System Extensions
    version: Optional[str] = None  # SemVer format (e.g., "1.3.0")
    parent_hash: Optional[str] = None  # Hash of parent block in evolution chain
    evolution_type: Optional[str] = None  # "expansion", "mutation", "pruning", "migration"
    dimension_changes: Optional[Dict[str, Any]] = None  # Dimension change information
    compatibility_range: Optional[List[str]] = None  # Compatible version range
    evolution_metadata: Optional[Dict[str, Any]] = None  # Evolution-specific metadata
    
    # Code Block Extensions (for executable evolution)
    code_type: Optional[str] = None  # "inference_logic", "activation", "layer"
    target_layer: Optional[str] = None  # Target layer for code application
    language: Optional[str] = None  # "python", "torch_script", "wasm"
    execution_environment: Optional[str] = None  # "torch_2.0", "tensorrt_10"
    dependencies: Optional[List[str]] = None  # Code dependencies
    
    # Migration Block Extensions  
    migration_from: Optional[str] = None  # Source version for migration
    migration_to: Optional[str] = None  # Target version for migration
    migration_ops: Optional[List[str]] = None  # List of migration operation hashes
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []
        if self.dependencies is None:
            self.dependencies = []
        if self.migration_ops is None:
            self.migration_ops = []

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    def to_json(self) -> str:
        """
        Stable JSON representation used for hashing.
        WARNING: Uses canonical JSON for consensus operations.
        """
        return dumps_canonical(asdict(self))

    def compute_hash(self) -> str:
        """SHA-256 of the header JSON."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()
    
    # ---------------------------------------------------------------------
    # Evolution System Utilities
    # ---------------------------------------------------------------------
    def is_evolution_block(self) -> bool:
        """Check if this is an evolution-related block"""
        return self.block_type in ['migration', 'code', 'architecture'] or self.evolution_type is not None
    
    def is_layer_evolution(self) -> bool:
        """Check if this is a layer evolution block"""
        return self.block_type in ['layer', 'dense_layer'] and self.evolution_type is not None
    
    def is_migration_block(self) -> bool:
        """Check if this is a migration block"""
        return self.block_type == 'migration'
    
    def is_code_block(self) -> bool:
        """Check if this is a code block"""
        return self.block_type == 'code'
    
    def get_evolution_parent(self) -> Optional[str]:
        """Get the parent block hash for evolution tracking"""
        return self.parent_hash
    
    def get_version(self) -> Optional[str]:
        """Get the version of this block"""
        return self.version
    
    def get_layer_identifier(self) -> Optional[str]:
        """Get layer identifier, preferring new dense format over legacy MoE format."""
        if self.layer_name:
            return self.layer_name
        elif self.expert_name:
            # Convert legacy MoE expert name to layer identifier
            return f"expert_{self.expert_name}"
        return None
    
    def is_dense_layer(self) -> bool:
        """Check if this is a dense model layer block."""
        return self.block_type == 'dense_layer' or (self.block_type == 'layer' and self.layer_name)
    
    def is_moe_expert(self) -> bool:
        """Check if this is a legacy MoE expert block."""
        return self.block_type == 'expert' or self.expert_name is not None
    
    def is_compatible_with_version(self, target_version: str) -> bool:
        """Check if this block is compatible with a target version"""
        if not self.compatibility_range:
            return True
        
        # TODO: Implement proper SemVer compatibility checking
        return target_version in self.compatibility_range or any(
            target_version.startswith(ver.replace('.x', '')) for ver in self.compatibility_range
        )
    
    def get_dimension_change_summary(self) -> str:
        """Get a summary of dimension changes"""
        if not self.dimension_changes:
            return "No dimension changes"
        
        changes = []
        for param, change in self.dimension_changes.items():
            if isinstance(change, dict) and 'from' in change and 'to' in change:
                changes.append(f"{param}: {change['from']} → {change['to']}")
        
        return "; ".join(changes) if changes else "Dimension metadata available"
    
    def get_migration_summary(self) -> str:
        """Get a summary of migration operations"""
        if not self.is_migration_block():
            return "Not a migration block"
        
        return f"Migration from {self.migration_from} to {self.migration_to}"


@dataclass
class Block:
    """A full block: header + payload + optional miner signature."""

    header: BlockHeader
    payload: bytes
    miner_pub: Optional[str] = None  # hex-encoded compressed public key
    payload_sig: Optional[str] = None  # hex ECDSA signature of payload

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def compute_hash(self) -> str:
        header_json = self.header.to_json()
        return hashlib.sha256(header_json.encode() + self.payload).hexdigest()

    def to_dict(self) -> dict:
        d = {
            "header": asdict(self.header),
            "payload": self.payload.hex(),
        }
        if self.miner_pub is not None:
            d["miner_pub"] = self.miner_pub
        if self.payload_sig is not None:
            d["payload_sig"] = self.payload_sig
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Block":
        header_data = data["header"]
        payload = bytes.fromhex(data["payload"])
        header = BlockHeader(**header_data)
        return cls(
            header=header,
            payload=payload,
            miner_pub=data.get("miner_pub"),
            payload_sig=data.get("payload_sig"),
        )


# ---------------------------------------------------------------------
# DAG validation utilities
# ---------------------------------------------------------------------

def has_cycle(blocks: List[Block]) -> bool:
    """Check if the DAG has cycles using DFS."""
    # Build adjacency list: hash -> list of dependent hashes
    graph = {}
    block_hashes = set()
    
    for block in blocks:
        block_hash = block.compute_hash()
        block_hashes.add(block_hash)
        graph[block_hash] = block.header.depends_on.copy()
    
    # DFS cycle detection
    WHITE, GRAY, BLACK = 0, 1, 2
    colors = {h: WHITE for h in block_hashes}
    
    def dfs(node: str) -> bool:
        if colors[node] == GRAY:  # Back edge found - cycle detected
            return True
        if colors[node] == BLACK:  # Already processed
            return False
            
        colors[node] = GRAY
        for neighbor in graph.get(node, []):
            if neighbor in colors and dfs(neighbor):
                return True
        colors[node] = BLACK
        return False
    
    for node in block_hashes:
        if colors[node] == WHITE and dfs(node):
            return True
    return False


def topological_sort(blocks: List[Block]) -> Optional[List[str]]:
    """Return topologically sorted block hashes, None if cycle exists."""
    if has_cycle(blocks):
        return None
    
    # Build graph and in-degree count
    graph = {}
    in_degree = {}
    block_hashes = set()
    
    for block in blocks:
        block_hash = block.compute_hash()
        block_hashes.add(block_hash)
        graph[block_hash] = block.header.depends_on.copy()
        in_degree[block_hash] = 0
    
    # Calculate in-degrees
    for block_hash in block_hashes:
        for dep in graph[block_hash]:
            if dep in in_degree:
                in_degree[dep] += 1
    
    # Kahn's algorithm
    queue = [h for h in block_hashes if in_degree[h] == 0]
    result = []
    
    while queue:
        current = queue.pop(0)
        result.append(current)
        
        for dep in graph[current]:
            if dep in in_degree:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)
    
    return result if len(result) == len(block_hashes) else None


def validate_dag_structure(blocks: List[Block]) -> bool:
    """Validate that blocks form a valid DAG structure."""
    if not blocks:
        return True
    
    # Check for cycles
    if has_cycle(blocks):
        return False
    
    # Check that all dependencies exist
    block_hashes = {block.compute_hash() for block in blocks}
    for block in blocks:
        for dep_hash in block.header.depends_on:
            if dep_hash not in block_hashes:
                return False  # Dependency not found
    
    return True 