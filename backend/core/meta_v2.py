"""
Blyan Evolution System - Meta Chain v2 Schema
Supports Semantic Versioning and evolutionary model architecture
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
import json


@dataclass
class RuntimeRequirements:
    """Runtime environment requirements for model execution"""
    api: str = "InferenceAPI@2"
    engine: List[str] = field(default_factory=lambda: ["torch@2.3", "tensorrt@10"])
    image_digest: Optional[str] = None
    python_version: str = ">=3.9"
    
    def is_compatible(self, available_runtime: Dict[str, Any]) -> bool:
        """Check if available runtime meets requirements"""
        # TODO: Implement detailed compatibility checking
        return True


@dataclass
class ArchitectureMutations:
    """Allowed architectural changes for evolution"""
    allow_layer_addition: bool = True
    allow_expert_multiplication: bool = True
    allow_dimension_scaling: bool = True
    allow_router_evolution: bool = True
    allow_activation_changes: bool = False
    
    def can_apply(self, mutation_type: str) -> bool:
        """Check if a specific mutation type is allowed"""
        mutation_map = {
            "layer_addition": self.allow_layer_addition,
            "expert_multiplication": self.allow_expert_multiplication,
            "dimension_scaling": self.allow_dimension_scaling,
            "router_evolution": self.allow_router_evolution,
            "activation_changes": self.allow_activation_changes
        }
        return mutation_map.get(mutation_type, False)


@dataclass
class DynamicExpertConfig:
    """Configuration for dynamic expert management"""
    min_experts: int = 2
    max_experts: int = 16
    current_experts: int = 8
    expansion_trigger_threshold: float = 0.85  # Usage threshold for expansion
    pruning_trigger_threshold: float = 0.15   # Usage threshold for pruning
    
    def can_expand(self) -> bool:
        return self.current_experts < self.max_experts
    
    def can_prune(self) -> bool:
        return self.current_experts > self.min_experts


class SemVer:
    """Semantic Versioning implementation"""
    
    def __init__(self, version: str):
        self.version = version
        self.major, self.minor, self.patch = self._parse(version)
    
    def _parse(self, version: str) -> tuple[int, int, int]:
        """Parse semantic version string"""
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
        match = re.match(pattern, version)
        if not match:
            raise ValueError(f"Invalid semantic version: {version}")
        
        major, minor, patch = match.groups()[:3]
        return int(major), int(minor), int(patch)
    
    def is_compatible(self, other: 'SemVer') -> bool:
        """Check if versions are compatible (same major version)"""
        return self.major == other.major
    
    def is_backward_compatible(self, other: 'SemVer') -> bool:
        """Check if this version is backward compatible with other"""
        if self.major != other.major:
            return False
        if self.minor < other.minor:
            return False
        return True
    
    def bump_patch(self) -> 'SemVer':
        """Create new version with bumped patch"""
        return SemVer(f"{self.major}.{self.minor}.{self.patch + 1}")
    
    def bump_minor(self) -> 'SemVer':
        """Create new version with bumped minor (resets patch)"""
        return SemVer(f"{self.major}.{self.minor + 1}.0")
    
    def bump_major(self) -> 'SemVer':
        """Create new version with bumped major (resets minor and patch)"""
        return SemVer(f"{self.major + 1}.0.0")
    
    def __str__(self) -> str:
        return self.version
    
    def __lt__(self, other: 'SemVer') -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other: 'SemVer') -> bool:
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)


@dataclass
class MetaSpecV2:
    """Enhanced Meta Chain specification with evolution support"""
    
    # Core identification
    model_id: str
    version: str  # SemVer format
    generation: int = 0  # Auto-increment on minor bumps
    
    # Architecture definition
    architecture: str = "adaptive-moe"
    model_name: str = "evolving_blyan"
    
    # Base structure
    base_num_layers: int = 24
    expandable_layers: List[int] = field(default_factory=lambda: [2, 3])
    
    # Dynamic expert configuration
    dynamic_experts: Dict[str, DynamicExpertConfig] = field(default_factory=dict)
    
    # Evolution capabilities
    architecture_mutations: ArchitectureMutations = field(default_factory=ArchitectureMutations)
    
    # Runtime requirements
    runtime_req: RuntimeRequirements = field(default_factory=RuntimeRequirements)
    
    # Compatibility
    compatibility_range: List[str] = field(default_factory=list)
    
    # Evolution history
    parent_version: Optional[str] = None
    evolution_history: List[str] = field(default_factory=list)
    
    # Performance thresholds
    pol_threshold: float = 0.01  # Minimum improvement required
    validator: str = "ChainValidatorV2"

    # Workflow flags
    is_snapshot: bool = False
    is_draft: bool = False
    
    def __post_init__(self):
        """Validate and initialize fields"""
        # Validate SemVer format
        try:
            self.semver = SemVer(self.version)
        except ValueError as e:
            raise ValueError(f"Invalid version format: {e}")
        
        # Initialize dynamic experts if empty
        if not self.dynamic_experts:
            for layer_id in range(self.base_num_layers):
                layer_key = f"layer{layer_id}"
                self.dynamic_experts[layer_key] = DynamicExpertConfig()
        
        # Set compatibility range if empty
        if not self.compatibility_range:
            self.compatibility_range = [f"{self.semver.major}.0.0", f"{self.semver.major}.99.99"]
    
    def is_compatible_with(self, other_version: str) -> bool:
        """Check if this spec is compatible with another version"""
        try:
            other_semver = SemVer(other_version)
            my_semver = SemVer(self.version)
            return my_semver.is_compatible(other_semver)
        except ValueError:
            return False
    
    def can_migrate_from(self, from_version: str) -> bool:
        """Check if migration from another version is possible"""
        try:
            from_semver = SemVer(from_version)
            my_semver = SemVer(self.version)
            
            # Can migrate within same major version
            if from_semver.major == my_semver.major:
                return from_semver <= my_semver
            
            # Cross-major migration requires explicit support
            return from_version in self.compatibility_range
        except ValueError:
            return False
    
    def get_total_experts(self) -> int:
        """Calculate total number of experts across all layers"""
        return sum(config.current_experts for config in self.dynamic_experts.values())
    
    def get_expandable_expert_count(self) -> int:
        """Get number of experts that can be added"""
        expandable = 0
        for layer_id in self.expandable_layers:
            layer_key = f"layer{layer_id}"
            if layer_key in self.dynamic_experts:
                config = self.dynamic_experts[layer_key]
                expandable += config.max_experts - config.current_experts
        return expandable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "generation": self.generation,
            "architecture": self.architecture,
            "model_name": self.model_name,
            "base_num_layers": self.base_num_layers,
            "expandable_layers": self.expandable_layers,
            "dynamic_experts": {
                k: {
                    "min_experts": v.min_experts,
                    "max_experts": v.max_experts,
                    "current_experts": v.current_experts,
                    "expansion_trigger_threshold": v.expansion_trigger_threshold,
                    "pruning_trigger_threshold": v.pruning_trigger_threshold
                }
                for k, v in self.dynamic_experts.items()
            },
            "architecture_mutations": {
                "allow_layer_addition": self.architecture_mutations.allow_layer_addition,
                "allow_expert_multiplication": self.architecture_mutations.allow_expert_multiplication,
                "allow_dimension_scaling": self.architecture_mutations.allow_dimension_scaling,
                "allow_router_evolution": self.architecture_mutations.allow_router_evolution,
                "allow_activation_changes": self.architecture_mutations.allow_activation_changes
            },
            "runtime_req": {
                "api": self.runtime_req.api,
                "engine": self.runtime_req.engine,
                "image_digest": self.runtime_req.image_digest,
                "python_version": self.runtime_req.python_version
            },
            "compatibility_range": self.compatibility_range,
            "parent_version": self.parent_version,
            "evolution_history": self.evolution_history,
            "pol_threshold": self.pol_threshold,
            "validator": self.validator,
            "is_snapshot": self.is_snapshot,
            "is_draft": self.is_draft
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaSpecV2':
        """Create MetaSpecV2 from dictionary"""
        # Extract and convert dynamic experts
        dynamic_experts = {}
        for k, v in data.get("dynamic_experts", {}).items():
            dynamic_experts[k] = DynamicExpertConfig(**v)
        
        # Extract architecture mutations
        arch_mutations = ArchitectureMutations(**data.get("architecture_mutations", {}))
        
        # Extract runtime requirements
        runtime_req = RuntimeRequirements(**data.get("runtime_req", {}))
        
        return cls(
            model_id=data["model_id"],
            version=data["version"],
            generation=data.get("generation", 0),
            architecture=data.get("architecture", "adaptive-moe"),
            model_name=data.get("model_name", "evolving_blyan"),
            base_num_layers=data.get("base_num_layers", 24),
            expandable_layers=data.get("expandable_layers", [2, 3]),
            dynamic_experts=dynamic_experts,
            architecture_mutations=arch_mutations,
            runtime_req=runtime_req,
            compatibility_range=data.get("compatibility_range", []),
            parent_version=data.get("parent_version"),
            evolution_history=data.get("evolution_history", []),
            pol_threshold=data.get("pol_threshold", 0.01),
            validator=data.get("validator", "ChainValidatorV2"),
            is_snapshot=data.get("is_snapshot", False),
            is_draft=data.get("is_draft", False)
        )
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


def validate_meta_spec_v2(spec_dict: Dict[str, Any]) -> bool:
    """Validate MetaSpecV2 dictionary"""
    try:
        spec = MetaSpecV2.from_dict(spec_dict)
        return True
    except Exception as e:
        print(f"Validation failed: {e}")
        return False


# TODO: Implement MetaChainV2Manager for CRUD operations
class MetaChainV2Manager:
    """Manager for MetaChain v2 operations"""
    
    def __init__(self, chain):
        self.chain = chain
    
    def create_spec_block(self, spec: MetaSpecV2) -> str:
        """Create a new spec block and return its hash"""
        payload_obj = {
            "type": "meta_spec_v2",
            "spec": spec.to_dict()
        }
        payload = json.dumps(payload_obj).encode()
        blk = self.chain.add_block(
            payload=payload,
            block_type='meta'
        )
        return blk.compute_hash()
    
    def get_latest_spec(self) -> Optional[MetaSpecV2]:
        """Get the latest meta specification"""
        # Iterate blocks in reverse to find latest meta_spec_v2
        blocks = list(self.chain.storage.iter_blocks())
        for block in reversed(blocks):
            if getattr(block.header, 'block_type', None) == 'meta':
                try:
                    obj = json.loads(block.payload.decode())
                    if obj.get('type') == 'meta_spec_v2' and 'spec' in obj:
                        return MetaSpecV2.from_dict(obj['spec'])
                except Exception:
                    continue
        return None
    
    def get_spec_by_version(self, version: str) -> Optional[MetaSpecV2]:
        """Get meta specification by version"""
        for block in reversed(list(self.chain.storage.iter_blocks())):
            if getattr(block.header, 'block_type', None) == 'meta':
                try:
                    obj = json.loads(block.payload.decode())
                    if obj.get('type') == 'meta_spec_v2' and 'spec' in obj:
                        if obj['spec'].get('version') == version:
                            return MetaSpecV2.from_dict(obj['spec'])
                except Exception:
                    continue
        return None
    
    def get_compatible_specs(self, target_version: str) -> List[MetaSpecV2]:
        """Get all specs compatible with target version"""
        specs: List[MetaSpecV2] = []
        for block in self.chain.storage.iter_blocks():
            if getattr(block.header, 'block_type', None) == 'meta':
                try:
                    obj = json.loads(block.payload.decode())
                    if obj.get('type') == 'meta_spec_v2' and 'spec' in obj:
                        spec = MetaSpecV2.from_dict(obj['spec'])
                        if spec.is_compatible_with(target_version):
                            specs.append(spec)
                except Exception:
                    continue
        return specs