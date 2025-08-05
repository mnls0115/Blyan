"""
Blyan Evolution System - Migration Block Operations
Handles version-to-version model evolution with semantic versioning
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Literal
import json
from enum import Enum

from .meta_v2 import MetaSpecV2, SemVer


class MigrationOpType(Enum):
    """Types of migration operations"""
    ADD_EXPERTS = "add_experts"
    REMOVE_EXPERTS = "remove_experts" 
    WIDEN_FFN = "widen_ffn"
    DEEPEN_LAYERS = "deepen_layers"
    REGISTER_CODE_BLOCK = "register_code_block"
    UPDATE_ROUTER = "update_router"
    CHANGE_ACTIVATION = "change_activation"
    SCALE_ATTENTION = "scale_attention"


@dataclass
class MigrationOp:
    """Single migration operation"""
    op: str  # MigrationOpType value
    layer: Optional[int] = None
    count: Optional[int] = None
    old: Optional[Union[int, str]] = None
    new: Optional[Union[int, str]] = None
    ref: Optional[str] = None  # Block hash reference
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate operation parameters"""
        try:
            op_type = MigrationOpType(self.op)
            
            if op_type == MigrationOpType.ADD_EXPERTS:
                return self.layer is not None and self.count is not None and self.count > 0
            
            elif op_type == MigrationOpType.REMOVE_EXPERTS:
                return self.layer is not None and self.count is not None and self.count > 0
            
            elif op_type == MigrationOpType.WIDEN_FFN:
                return (self.layer is not None and 
                       self.old is not None and self.new is not None and
                       isinstance(self.old, int) and isinstance(self.new, int) and
                       self.new > self.old)
            
            elif op_type == MigrationOpType.DEEPEN_LAYERS:
                return self.count is not None and self.count > 0
            
            elif op_type == MigrationOpType.REGISTER_CODE_BLOCK:
                return self.ref is not None
            
            elif op_type == MigrationOpType.UPDATE_ROUTER:
                return self.layer is not None and self.ref is not None
            
            else:
                # For other operations, basic validation
                return True
                
        except ValueError:
            return False
    
    def get_impact_level(self) -> Literal["patch", "minor", "major"]:
        """Determine the impact level of this operation"""
        op_type = MigrationOpType(self.op)
        
        # MAJOR changes (breaking compatibility)
        if op_type in [MigrationOpType.CHANGE_ACTIVATION, MigrationOpType.DEEPEN_LAYERS]:
            return "major"
        
        # MINOR changes (backward compatible extensions)
        elif op_type in [MigrationOpType.ADD_EXPERTS, MigrationOpType.WIDEN_FFN, 
                        MigrationOpType.SCALE_ATTENTION, MigrationOpType.UPDATE_ROUTER]:
            return "minor"
        
        # PATCH changes (same structure, different weights)
        else:
            return "patch"


@dataclass
class MigrationBlock:
    """Migration block for version transitions"""
    type: str = "migration"
    from_version: str = ""
    to_version: str = ""
    ops: List[MigrationOp] = field(default_factory=list)
    pol_threshold: float = 0.01
    validator: str = "ChainValidatorV2"
    
    # Execution metadata
    estimated_cost: Optional[float] = None
    rollback_plan: Optional[List[str]] = None
    dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate migration block"""
        if not self.from_version or not self.to_version:
            raise ValueError("Both from_version and to_version are required")
        
        try:
            self.from_semver = SemVer(self.from_version)
            self.to_semver = SemVer(self.to_version)
        except ValueError as e:
            raise ValueError(f"Invalid version format: {e}")
        
        # Validate that to_version is greater than from_version
        if not (self.to_semver > self.from_semver):
            raise ValueError("to_version must be greater than from_version")
    
    def validate_operations(self) -> bool:
        """Validate all operations in the migration"""
        if not self.ops:
            return False
        
        for op in self.ops:
            if not op.validate():
                return False
        
        return True
    
    def get_required_version_bump(self) -> Literal["patch", "minor", "major"]:
        """Determine the minimum version bump required"""
        max_impact = "patch"
        
        for op in self.ops:
            impact = op.get_impact_level()
            if impact == "major":
                max_impact = "major"
            elif impact == "minor" and max_impact == "patch":
                max_impact = "minor"
        
        return max_impact
    
    def is_version_bump_valid(self) -> bool:
        """Check if the version bump matches the operation impacts"""
        required_bump = self.get_required_version_bump()
        
        if required_bump == "major":
            return self.to_semver.major > self.from_semver.major
        elif required_bump == "minor":
            return (self.to_semver.major == self.from_semver.major and 
                   self.to_semver.minor > self.from_semver.minor)
        else:  # patch
            return (self.to_semver.major == self.from_semver.major and
                   self.to_semver.minor == self.from_semver.minor and
                   self.to_semver.patch > self.from_semver.patch)
    
    def get_affected_layers(self) -> List[int]:
        """Get list of layers affected by this migration"""
        layers = set()
        for op in self.ops:
            if op.layer is not None:
                layers.add(op.layer)
        return sorted(list(layers))
    
    def get_new_expert_count(self, layer: int, current_count: int) -> int:
        """Calculate new expert count for a layer after migration"""
        new_count = current_count
        
        for op in self.ops:
            if op.layer == layer:
                if op.op == MigrationOpType.ADD_EXPERTS.value:
                    new_count += op.count
                elif op.op == MigrationOpType.REMOVE_EXPERTS.value:
                    new_count = max(1, new_count - op.count)  # At least 1 expert
        
        return new_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "type": self.type,
            "from": self.from_version,
            "to": self.to_version,
            "ops": [
                {
                    "op": op.op,
                    "layer": op.layer,
                    "count": op.count,
                    "old": op.old,
                    "new": op.new,
                    "ref": op.ref,
                    "metadata": op.metadata
                }
                for op in self.ops
            ],
            "pol_threshold": self.pol_threshold,
            "validator": self.validator,
            "estimated_cost": self.estimated_cost,
            "rollback_plan": self.rollback_plan,
            "dependencies": self.dependencies
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MigrationBlock':
        """Create MigrationBlock from dictionary"""
        ops = []
        for op_data in data.get("ops", []):
            ops.append(MigrationOp(
                op=op_data["op"],
                layer=op_data.get("layer"),
                count=op_data.get("count"),
                old=op_data.get("old"),
                new=op_data.get("new"),
                ref=op_data.get("ref"),
                metadata=op_data.get("metadata", {})
            ))
        
        return cls(
            type=data.get("type", "migration"),
            from_version=data["from"],
            to_version=data["to"],
            ops=ops,
            pol_threshold=data.get("pol_threshold", 0.01),
            validator=data.get("validator", "ChainValidatorV2"),
            estimated_cost=data.get("estimated_cost"),
            rollback_plan=data.get("rollback_plan"),
            dependencies=data.get("dependencies", [])
        )
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class MigrationExecutor:
    """Executes migration operations"""
    
    def __init__(self, meta_manager, param_chain):
        self.meta_manager = meta_manager
        self.param_chain = param_chain
    
    def apply_migration(self, migration: MigrationBlock, current_spec: MetaSpecV2) -> MetaSpecV2:
        """Apply migration to create new specification"""
        # Validate migration
        if not migration.validate_operations():
            raise ValueError("Invalid migration operations")
        
        if not migration.is_version_bump_valid():
            raise ValueError("Version bump doesn't match operation impacts")
        
        # Create new spec based on current spec
        new_spec = self._create_evolved_spec(current_spec, migration)
        
        # Apply each operation
        for op in migration.ops:
            new_spec = self._apply_operation(new_spec, op)
        
        # Update version and generation
        new_spec.version = migration.to_version
        if migration.get_required_version_bump() == "minor":
            new_spec.generation += 1
        
        # Update evolution history
        new_spec.parent_version = current_spec.version
        new_spec.evolution_history = current_spec.evolution_history + [current_spec.version]
        
        return new_spec
    
    def _create_evolved_spec(self, current_spec: MetaSpecV2, migration: MigrationBlock) -> MetaSpecV2:
        """Create a copy of current spec for evolution"""
        # TODO: Implement deep copy with proper field handling
        spec_dict = current_spec.to_dict()
        return MetaSpecV2.from_dict(spec_dict)
    
    def _apply_operation(self, spec: MetaSpecV2, op: MigrationOp) -> MetaSpecV2:
        """Apply a single migration operation to the spec"""
        op_type = MigrationOpType(op.op)
        
        if op_type == MigrationOpType.ADD_EXPERTS:
            self._add_experts(spec, op.layer, op.count)
        
        elif op_type == MigrationOpType.REMOVE_EXPERTS:
            self._remove_experts(spec, op.layer, op.count)
        
        elif op_type == MigrationOpType.WIDEN_FFN:
            self._widen_ffn(spec, op.layer, op.old, op.new)
        
        elif op_type == MigrationOpType.DEEPEN_LAYERS:
            self._deepen_layers(spec, op.count)
        
        elif op_type == MigrationOpType.REGISTER_CODE_BLOCK:
            self._register_code_block(spec, op.ref)
        
        # TODO: Implement other operation types
        
        return spec
    
    def _add_experts(self, spec: MetaSpecV2, layer: int, count: int):
        """Add experts to a layer"""
        layer_key = f"layer{layer}"
        if layer_key in spec.dynamic_experts:
            config = spec.dynamic_experts[layer_key]
            new_count = min(config.current_experts + count, config.max_experts)
            config.current_experts = new_count
    
    def _remove_experts(self, spec: MetaSpecV2, layer: int, count: int):
        """Remove experts from a layer"""
        layer_key = f"layer{layer}"
        if layer_key in spec.dynamic_experts:
            config = spec.dynamic_experts[layer_key]
            new_count = max(config.current_experts - count, config.min_experts)
            config.current_experts = new_count
    
    def _widen_ffn(self, spec: MetaSpecV2, layer: int, old_dim: int, new_dim: int):
        """Widen FFN layer dimensions"""
        # TODO: Implement FFN widening logic
        # This would update dimension metadata for the specific layer
        pass
    
    def _deepen_layers(self, spec: MetaSpecV2, count: int):
        """Add new layers to the model"""
        spec.base_num_layers += count
        
        # Add dynamic expert configs for new layers
        for i in range(count):
            new_layer_id = spec.base_num_layers - count + i
            layer_key = f"layer{new_layer_id}"
            spec.dynamic_experts[layer_key] = spec.dynamic_experts[f"layer{0}"]  # Copy from layer 0
    
    def _register_code_block(self, spec: MetaSpecV2, block_ref: str):
        """Register a code block reference"""
        # TODO: Implement code block registration
        # This would add the block reference to spec metadata
        pass
    
    def estimate_migration_cost(self, migration: MigrationBlock) -> float:
        """Estimate computational cost of migration"""
        # TODO: Implement cost estimation logic
        base_cost = 1.0
        
        for op in migration.ops:
            op_type = MigrationOpType(op.op)
            
            if op_type == MigrationOpType.ADD_EXPERTS:
                base_cost += op.count * 0.5
            elif op_type == MigrationOpType.WIDEN_FFN:
                ratio = op.new / op.old if op.old > 0 else 2.0
                base_cost += ratio * 0.3
            elif op_type == MigrationOpType.DEEPEN_LAYERS:
                base_cost += op.count * 2.0
        
        return base_cost
    
    def validate_migration_path(self, from_spec: MetaSpecV2, to_spec: MetaSpecV2) -> bool:
        """Validate that migration path is possible"""
        # TODO: Implement migration path validation
        return True


def create_expert_addition_migration(
    from_version: str, 
    to_version: str, 
    layer: int, 
    expert_count: int
) -> MigrationBlock:
    """Helper to create expert addition migration"""
    op = MigrationOp(
        op=MigrationOpType.ADD_EXPERTS.value,
        layer=layer,
        count=expert_count
    )
    
    return MigrationBlock(
        from_version=from_version,
        to_version=to_version,
        ops=[op]
    )


def create_ffn_widening_migration(
    from_version: str,
    to_version: str,
    layer: int,
    old_dim: int,
    new_dim: int
) -> MigrationBlock:
    """Helper to create FFN widening migration"""
    op = MigrationOp(
        op=MigrationOpType.WIDEN_FFN.value,
        layer=layer,
        old=old_dim,
        new=new_dim
    )
    
    return MigrationBlock(
        from_version=from_version,
        to_version=to_version,
        ops=[op]
    )