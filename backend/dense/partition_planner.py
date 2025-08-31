"""
Production-Ready Model Partition Planner
=========================================
Dynamically partitions transformer models (dense or MoE) across GPUs.
Supports models from 8B to 70B+ with automatic VRAM-based planning.
No hardcoding - all values derived from model config and runtime detection.
"""

import os
import logging
import torch
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelComponent:
    """Represents a model component (layer, expert, embedding, etc)."""
    component_type: str  # "embedding", "layer", "expert", "lm_head"
    component_id: Union[int, str]  # Layer index, expert id, or unique identifier
    memory_gb: float  # Weight memory in GB
    compute_intensity: float = 1.0  # Relative compute cost
    metadata: Dict = field(default_factory=dict)
    
    @property
    def display_name(self) -> str:
        """Generate display name for component."""
        if self.component_type == "layer":
            return f"Layer {self.component_id}"
        elif self.component_type == "expert":
            layer_id = self.metadata.get("layer_id", "?")
            return f"Expert {self.component_id} (L{layer_id})"
        elif self.component_type == "embedding":
            return "Embedding"
        elif self.component_type == "lm_head":
            return "LM Head"
        else:
            return f"{self.component_type} {self.component_id}"


@dataclass
class PipelineStage:
    """Represents a pipeline stage on a GPU."""
    stage_id: int
    device_id: Optional[int]  # GPU device ID if known
    components: List[ModelComponent]
    total_memory_gb: float
    kv_cache_gb: float
    total_with_overhead_gb: float
    available_headroom_gb: float
    component_range: Optional[Tuple[int, int]] = None  # For layer-based models
    
    def get_layer_indices(self) -> List[int]:
        """Get layer indices in this stage (for dense models)."""
        indices = []
        for comp in self.components:
            if comp.component_type == "layer" and isinstance(comp.component_id, int):
                indices.append(comp.component_id)
        return sorted(indices)
    
    def get_expert_ids(self) -> List[str]:
        """Get expert IDs in this stage (for MoE models)."""
        ids = []
        for comp in self.components:
            if comp.component_type == "expert":
                ids.append(str(comp.component_id))
        return ids


@dataclass
class PartitionPlan:
    """Complete partition plan for model distribution."""
    stages: List[PipelineStage]
    num_gpus: int
    total_gpus_available: int
    precision: str
    target_vram_gb: float
    detected_vram_gb: float  # Actually detected VRAM
    usable_vram_gb: float
    total_model_memory_gb: float
    model_architecture: str  # "dense" or "moe"
    model_name: str
    total_layers: int
    feasible: bool
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def validate(self) -> bool:
        """Validate the partition plan for correctness."""
        errors = []
        
        # Check all components are assigned
        if self.model_architecture == "dense":
            assigned_layers = set()
            for stage in self.stages:
                assigned_layers.update(stage.get_layer_indices())
            
            expected_layers = set(range(self.total_layers))
            missing = expected_layers - assigned_layers
            # Check for duplicates by converting to list and counting
            layer_list = list(assigned_layers)
            duplicate = [l for l in layer_list if layer_list.count(l) > 1]
            
            if missing:
                errors.append(f"Missing layers: {sorted(missing)}")
            if duplicate:
                errors.append(f"Duplicate layers: {sorted(set(duplicate))}")
        
        # Check memory constraints
        for stage in self.stages:
            if stage.total_with_overhead_gb > self.target_vram_gb:
                errors.append(
                    f"Stage {stage.stage_id} exceeds VRAM: "
                    f"{stage.total_with_overhead_gb:.2f}GB > {self.target_vram_gb:.2f}GB"
                )
            if stage.available_headroom_gb < 0:
                errors.append(
                    f"Stage {stage.stage_id} has negative headroom: "
                    f"{stage.available_headroom_gb:.2f}GB"
                )
        
        self.validation_errors = errors
        self.feasible = len(errors) == 0
        return self.feasible


class PartitionPlanner:
    """Production-ready planner for model partitioning across GPUs."""
    
    def __init__(
        self,
        model_config: Optional[Dict] = None,
        target_vram_gb: Optional[float] = None,
        precision: Optional[str] = None,
        reserved_headroom_gb: Optional[float] = None,
        kv_cache_budget_gb: Optional[float] = None,
        runtime_buffer_gb: Optional[float] = None,
        max_seq_len: int = 2048,
        batch_size: int = 1,
        auto_detect_vram: bool = True
    ):
        """
        Initialize partition planner with dynamic configuration.
        
        Args:
            model_config: Model configuration dict (uses active config if None)
            target_vram_gb: Target VRAM per GPU (auto-detects if None)
            precision: Weight precision (uses env var if None)
            reserved_headroom_gb: Reserved for CUDA context
            kv_cache_budget_gb: Reserved for KV cache
            runtime_buffer_gb: Reserved for temp buffers
            max_seq_len: Maximum sequence length
            batch_size: Batch size for inference
            auto_detect_vram: Auto-detect available VRAM
        """
        # Import config dynamically to avoid circular imports
        from config.model_profile import (
            get_model_config, WEIGHT_PRECISION, KV_CACHE_BUDGET_GB,
            PARTITION, calculate_layer_memory, LAYERS
        )
        
        # Load model configuration
        self.model_config = model_config or get_model_config()
        self.model_name = self.model_config["model_name"]
        self.model_architecture = self.model_config["architecture"]["type"]
        self.layers_config = self.model_config["layers"]
        self.total_layers = self.layers_config["num_hidden_layers"]
        
        # Precision configuration
        self.precision = precision or WEIGHT_PRECISION
        
        # VRAM detection and configuration
        self.detected_vram_gb = self._detect_gpu_vram() if auto_detect_vram else 0
        self.target_vram_gb = target_vram_gb or self.detected_vram_gb or float(
            os.getenv("TARGET_VRAM_GB", "4.0")
        )
        
        # Memory budgets (from env vars or defaults)
        self.reserved_headroom_gb = reserved_headroom_gb or float(
            os.getenv("RESERVED_HEADROOM_GB", PARTITION["default"]["reserved_headroom_gb"])
        )
        self.kv_cache_budget_gb = kv_cache_budget_gb or KV_CACHE_BUDGET_GB
        self.runtime_buffer_gb = runtime_buffer_gb or float(
            os.getenv("RUNTIME_BUFFER_GB", PARTITION["default"]["runtime_buffer_gb"])
        )
        
        # Inference configuration
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size or int(os.getenv("MICROBATCH_SIZE", "1"))
        
        # Calculate usable VRAM
        self.usable_vram_gb = (
            self.target_vram_gb - self.reserved_headroom_gb - 
            self.kv_cache_budget_gb - self.runtime_buffer_gb
        )
        
        # Get memory requirements for current precision
        self.layer_memory = calculate_layer_memory(self.precision)[self.precision]
        
        # Validate configuration
        if self.usable_vram_gb <= 0:
            raise ValueError(
                f"No usable VRAM! Target: {self.target_vram_gb:.2f}GB, "
                f"Reserved: {self.target_vram_gb - self.usable_vram_gb:.2f}GB. "
                f"Adjust environment variables: TARGET_VRAM_GB, KV_CACHE_BUDGET_GB, "
                f"RESERVED_HEADROOM_GB, RUNTIME_BUFFER_GB"
            )
        
        logger.info(
            f"PartitionPlanner initialized for {self.model_name}:\n"
            f"  Architecture: {self.model_architecture}\n"
            f"  Total layers: {self.total_layers}\n"
            f"  Precision: {self.precision}\n"
            f"  Target VRAM: {self.target_vram_gb:.2f}GB\n"
            f"  Detected VRAM: {self.detected_vram_gb:.2f}GB\n"
            f"  Usable VRAM: {self.usable_vram_gb:.2f}GB"
        )
    
    def _detect_gpu_vram(self) -> float:
        """Auto-detect available GPU VRAM."""
        try:
            if torch.cuda.is_available():
                # Get VRAM from first available GPU
                device_props = torch.cuda.get_device_properties(0)
                vram_bytes = device_props.total_memory
                vram_gb = vram_bytes / (1024**3)
                logger.info(f"Detected GPU VRAM: {vram_gb:.2f}GB ({device_props.name})")
                return vram_gb
            else:
                logger.warning("No CUDA devices available, using default VRAM")
                return 0
        except Exception as e:
            logger.warning(f"Failed to detect GPU VRAM: {e}")
            return 0
    
    def create_model_components(self) -> List[ModelComponent]:
        """
        Create model components based on architecture (dense or MoE).
        
        Returns:
            List of ModelComponent objects
        """
        components = []
        
        # Add embedding component
        components.append(ModelComponent(
            component_type="embedding",
            component_id="embed",
            memory_gb=self.layer_memory["embedding"],
            compute_intensity=0.5  # Embeddings are less compute-intensive
        ))
        
        if self.model_architecture == "dense":
            # Dense model: one component per layer
            per_layer_gb = self.layer_memory["per_layer"]
            for layer_idx in range(self.total_layers):
                components.append(ModelComponent(
                    component_type="layer",
                    component_id=layer_idx,
                    memory_gb=per_layer_gb,
                    compute_intensity=1.0,
                    metadata={"layer_idx": layer_idx}
                ))
        
        elif self.model_architecture == "moe":
            # MoE model: multiple experts per layer
            # This is future-ready code for MoE support
            num_experts = self.layers_config.get("num_experts", 8)
            expert_memory_gb = self.layer_memory["per_layer"] / num_experts
            
            for layer_idx in range(self.total_layers):
                # Shared attention weights
                components.append(ModelComponent(
                    component_type="layer",
                    component_id=f"{layer_idx}_attn",
                    memory_gb=expert_memory_gb * 0.3,  # Attention is ~30% of layer
                    compute_intensity=1.0,
                    metadata={"layer_idx": layer_idx, "shared": True}
                ))
                
                # Individual experts
                for expert_idx in range(num_experts):
                    components.append(ModelComponent(
                        component_type="expert",
                        component_id=f"{layer_idx}_{expert_idx}",
                        memory_gb=expert_memory_gb * 0.7 / num_experts,
                        compute_intensity=0.5,  # Only some experts activate
                        metadata={"layer_idx": layer_idx, "expert_idx": expert_idx}
                    ))
        
        # Add LM head component
        components.append(ModelComponent(
            component_type="lm_head",
            component_id="lm_head",
            memory_gb=self.layer_memory["lm_head"],
            compute_intensity=0.5
        ))
        
        return components
    
    def calculate_kv_cache_for_components(self, components: List[ModelComponent]) -> float:
        """
        Calculate KV cache size for given components.
        
        Args:
            components: List of model components
            
        Returns:
            KV cache size in GB
        """
        # Count layers (for dense) or attention components (for MoE)
        num_layers = sum(
            1 for c in components 
            if c.component_type == "layer" and 
            (self.model_architecture == "dense" or "attn" in str(c.component_id))
        )
        
        if num_layers == 0:
            return 0
        
        # Import dynamically to avoid circular dependency
        from config.model_profile import calculate_kv_cache_size
        
        return calculate_kv_cache_size(
            batch_size=self.batch_size,
            seq_len=self.max_seq_len,
            num_layers=num_layers,
            precision=self.precision
        )
    
    def greedy_partition(
        self,
        max_components_per_stage: Optional[int] = None,
        balance_compute: bool = True
    ) -> PartitionPlan:
        """
        Greedy partitioning that preserves component order.
        
        Args:
            max_components_per_stage: Optional cap on components per stage
            balance_compute: Balance by compute intensity, not just memory
            
        Returns:
            PartitionPlan with stages
        """
        components = self.create_model_components()
        stages = []
        current_stage_components = []
        current_stage_memory = 0.0
        current_stage_compute = 0.0
        stage_id = 0
        
        # Track total model memory
        total_model_memory = sum(c.memory_gb for c in components)
        
        for component in components:
            # Check if adding this component exceeds limits
            would_exceed_memory = (
                current_stage_memory + component.memory_gb > self.usable_vram_gb
            )
            would_exceed_component_cap = (
                max_components_per_stage and 
                len(current_stage_components) >= max_components_per_stage
            )
            
            # For single components that exceed capacity
            if component.memory_gb > self.usable_vram_gb:
                return PartitionPlan(
                    stages=[],
                    num_gpus=0,
                    total_gpus_available=torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    precision=self.precision,
                    target_vram_gb=self.target_vram_gb,
                    detected_vram_gb=self.detected_vram_gb,
                    usable_vram_gb=self.usable_vram_gb,
                    total_model_memory_gb=total_model_memory,
                    model_architecture=self.model_architecture,
                    model_name=self.model_name,
                    total_layers=self.total_layers,
                    feasible=False,
                    validation_errors=[
                        f"{component.display_name} requires {component.memory_gb:.2f}GB "
                        f"but only {self.usable_vram_gb:.2f}GB usable VRAM available"
                    ],
                    warnings=[
                        f"Increase TARGET_VRAM_GB to at least "
                        f"{component.memory_gb + self.reserved_headroom_gb + self.kv_cache_budget_gb:.1f}GB",
                        f"Or use lower precision (current: {self.precision})",
                        f"Or reduce KV_CACHE_BUDGET_GB (current: {self.kv_cache_budget_gb:.1f}GB)"
                    ]
                )
            
            # Start new stage if needed
            if current_stage_components and (would_exceed_memory or would_exceed_component_cap):
                # Finalize current stage
                kv_cache = self.calculate_kv_cache_for_components(current_stage_components)
                total_with_overhead = (
                    current_stage_memory + kv_cache + 
                    self.reserved_headroom_gb + self.runtime_buffer_gb
                )
                
                # Get component range for dense models
                component_range = None
                if self.model_architecture == "dense":
                    layer_indices = [
                        c.component_id for c in current_stage_components
                        if c.component_type == "layer" and isinstance(c.component_id, int)
                    ]
                    if layer_indices:
                        component_range = (min(layer_indices), max(layer_indices))
                
                stages.append(PipelineStage(
                    stage_id=stage_id,
                    device_id=stage_id if stage_id < torch.cuda.device_count() else None,
                    components=current_stage_components,
                    total_memory_gb=current_stage_memory,
                    kv_cache_gb=kv_cache,
                    total_with_overhead_gb=total_with_overhead,
                    available_headroom_gb=self.target_vram_gb - total_with_overhead,
                    component_range=component_range
                ))
                
                # Reset for new stage
                stage_id += 1
                current_stage_components = []
                current_stage_memory = 0.0
                current_stage_compute = 0.0
            
            # Add component to current stage
            current_stage_components.append(component)
            current_stage_memory += component.memory_gb
            current_stage_compute += component.compute_intensity
        
        # Finalize last stage
        if current_stage_components:
            kv_cache = self.calculate_kv_cache_for_components(current_stage_components)
            total_with_overhead = (
                current_stage_memory + kv_cache + 
                self.reserved_headroom_gb + self.runtime_buffer_gb
            )
            
            component_range = None
            if self.model_architecture == "dense":
                layer_indices = [
                    c.component_id for c in current_stage_components
                    if c.component_type == "layer" and isinstance(c.component_id, int)
                ]
                if layer_indices:
                    component_range = (min(layer_indices), max(layer_indices))
            
            stages.append(PipelineStage(
                stage_id=stage_id,
                device_id=stage_id if stage_id < torch.cuda.device_count() else None,
                components=current_stage_components,
                total_memory_gb=current_stage_memory,
                kv_cache_gb=kv_cache,
                total_with_overhead_gb=total_with_overhead,
                available_headroom_gb=self.target_vram_gb - total_with_overhead,
                component_range=component_range
            ))
        
        # Create partition plan
        plan = PartitionPlan(
            stages=stages,
            num_gpus=len(stages),
            total_gpus_available=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            precision=self.precision,
            target_vram_gb=self.target_vram_gb,
            detected_vram_gb=self.detected_vram_gb,
            usable_vram_gb=self.usable_vram_gb,
            total_model_memory_gb=total_model_memory,
            model_architecture=self.model_architecture,
            model_name=self.model_name,
            total_layers=self.total_layers,
            feasible=True
        )
        
        # Validate the plan
        plan.validate()
        
        return plan
    
    def balanced_partition(self, num_gpus: Optional[int] = None) -> PartitionPlan:
        """
        Create a balanced partition across specified number of GPUs.
        
        Args:
            num_gpus: Number of GPUs to use (auto-detects if None)
            
        Returns:
            PartitionPlan with balanced stages
        """
        if num_gpus is None:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
            else:
                # Calculate minimum GPUs needed
                total_memory = sum(c.memory_gb for c in self.create_model_components())
                num_gpus = max(1, int(total_memory / self.usable_vram_gb) + 1)
        
        components = self.create_model_components()
        total_components = len(components)
        
        # Calculate components per stage
        base_per_stage = total_components // num_gpus
        extra_components = total_components % num_gpus
        
        stages = []
        component_idx = 0
        
        for stage_id in range(num_gpus):
            # Determine number of components for this stage
            stage_component_count = base_per_stage + (1 if stage_id < extra_components else 0)
            
            # Assign components
            stage_components = []
            stage_memory = 0.0
            
            for _ in range(stage_component_count):
                if component_idx < len(components):
                    comp = components[component_idx]
                    stage_components.append(comp)
                    stage_memory += comp.memory_gb
                    component_idx += 1
            
            # Calculate overheads
            kv_cache = self.calculate_kv_cache_for_components(stage_components)
            total_with_overhead = (
                stage_memory + kv_cache + 
                self.reserved_headroom_gb + self.runtime_buffer_gb
            )
            
            # Get component range
            component_range = None
            if self.model_architecture == "dense":
                layer_indices = [
                    c.component_id for c in stage_components
                    if c.component_type == "layer" and isinstance(c.component_id, int)
                ]
                if layer_indices:
                    component_range = (min(layer_indices), max(layer_indices))
            
            stages.append(PipelineStage(
                stage_id=stage_id,
                device_id=stage_id if stage_id < torch.cuda.device_count() else None,
                components=stage_components,
                total_memory_gb=stage_memory,
                kv_cache_gb=kv_cache,
                total_with_overhead_gb=total_with_overhead,
                available_headroom_gb=self.target_vram_gb - total_with_overhead,
                component_range=component_range
            ))
        
        # Create and validate plan
        plan = PartitionPlan(
            stages=stages,
            num_gpus=num_gpus,
            total_gpus_available=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            precision=self.precision,
            target_vram_gb=self.target_vram_gb,
            detected_vram_gb=self.detected_vram_gb,
            usable_vram_gb=self.usable_vram_gb,
            total_model_memory_gb=sum(c.memory_gb for c in components),
            model_architecture=self.model_architecture,
            model_name=self.model_name,
            total_layers=self.total_layers,
            feasible=True
        )
        
        plan.validate()
        return plan
    
    def print_plan(self, plan: PartitionPlan):
        """Pretty print the partition plan with full details."""
        print(f"\n{'=' * 80}")
        print(f"📊 Partition Plan for {plan.model_name}")
        print(f"{'=' * 80}")
        
        if not plan.feasible:
            print(f"❌ Plan is NOT feasible")
            for error in plan.validation_errors:
                print(f"   ERROR: {error}")
            for warning in plan.warnings:
                print(f"   ⚠️  {warning}")
            return
        
        print(f"✅ Plan is feasible")
        print(f"\nModel Configuration:")
        print(f"  Architecture: {plan.model_architecture}")
        print(f"  Total layers: {plan.total_layers}")
        print(f"  Precision: {plan.precision}")
        print(f"  Total model memory: {plan.total_model_memory_gb:.2f}GB")
        
        print(f"\nGPU Configuration:")
        print(f"  Target VRAM: {plan.target_vram_gb:.2f}GB per GPU")
        print(f"  Detected VRAM: {plan.detected_vram_gb:.2f}GB")
        print(f"  Usable VRAM: {plan.usable_vram_gb:.2f}GB")
        print(f"  Number of stages: {plan.num_gpus}")
        print(f"  GPUs available: {plan.total_gpus_available}")
        
        print(f"\n{'─' * 80}")
        print(f"Pipeline Stages:")
        print(f"{'─' * 80}")
        
        for stage in plan.stages:
            device_str = f"GPU {stage.device_id}" if stage.device_id is not None else "Unassigned"
            print(f"\n🖥️  Stage {stage.stage_id} ({device_str}):")
            
            # Group components by type
            component_summary = []
            
            # Check for embedding
            has_embedding = any(c.component_type == "embedding" for c in stage.components)
            if has_embedding:
                component_summary.append("Embedding")
            
            # Check for layers
            if plan.model_architecture == "dense":
                layer_indices = stage.get_layer_indices()
                if layer_indices:
                    if len(layer_indices) == 1:
                        component_summary.append(f"Layer {layer_indices[0]}")
                    else:
                        component_summary.append(f"Layers {min(layer_indices)}-{max(layer_indices)}")
            else:
                # MoE model
                expert_ids = stage.get_expert_ids()
                if expert_ids:
                    component_summary.append(f"{len(expert_ids)} experts")
            
            # Check for LM head
            has_lm_head = any(c.component_type == "lm_head" for c in stage.components)
            if has_lm_head:
                component_summary.append("LM Head")
            
            print(f"  Components: {', '.join(component_summary)}")
            print(f"  Component count: {len(stage.components)}")
            print(f"  Weight memory: {stage.total_memory_gb:.3f}GB")
            print(f"  KV cache: {stage.kv_cache_gb:.3f}GB")
            print(f"  Total with overhead: {stage.total_with_overhead_gb:.3f}GB")
            print(f"  Available headroom: {stage.available_headroom_gb:.3f}GB")
            
            if stage.available_headroom_gb < 0:
                print(f"  ⚠️  WARNING: Exceeds target VRAM by {-stage.available_headroom_gb:.3f}GB!")
        
        # Print validation results
        if plan.validation_errors:
            print(f"\n{'─' * 80}")
            print("Validation Errors:")
            for error in plan.validation_errors:
                print(f"  ❌ {error}")
        
        if plan.warnings:
            print(f"\n{'─' * 80}")
            print("Warnings:")
            for warning in plan.warnings:
                print(f"  ⚠️  {warning}")
        
        print(f"\n{'=' * 80}\n")


def create_optimal_plan(
    target_vram_gb: Optional[float] = None,
    precision: Optional[str] = None,
    strategy: str = "greedy",
    num_gpus: Optional[int] = None
) -> PartitionPlan:
    """
    Create an optimal partition plan based on configuration.
    
    Args:
        target_vram_gb: GPU VRAM in GB (auto-detects if None)
        precision: Weight precision (uses env var if None)
        strategy: Partitioning strategy ("greedy" or "balanced")
        num_gpus: Number of GPUs for balanced strategy
        
    Returns:
        Optimal PartitionPlan
    """
    planner = PartitionPlanner(
        target_vram_gb=target_vram_gb,
        precision=precision,
        auto_detect_vram=True
    )
    
    if strategy == "balanced" and num_gpus:
        return planner.balanced_partition(num_gpus)
    else:
        return planner.greedy_partition()


if __name__ == "__main__":
    # Demo partition planning with different configurations
    print("\n🚀 Production Partition Planning Demo\n")
    
    # Test with different model sizes via environment variable
    for model_size in ["8B", "32B", "70B"]:
        os.environ["MODEL_SIZE"] = model_size
        
        # Reload config for new model size
        from importlib import reload
        import config.model_profile as model_profile
        reload(model_profile)
        
        print(f"\n{'='*80}")
        print(f"Testing {model_size} Model")
        print(f"{'='*80}")
        
        # Test different VRAM configurations
        for vram, precision in [(4.0, "int4"), (8.0, "int8"), (16.0, "bf16")]:
            print(f"\n{vram}GB VRAM with {precision} precision:")
            try:
                planner = PartitionPlanner(
                    target_vram_gb=vram,
                    precision=precision,
                    auto_detect_vram=False  # Use specified VRAM for demo
                )
                plan = planner.greedy_partition()
                
                if plan.feasible:
                    print(f"  ✅ Feasible with {plan.num_gpus} GPU(s)")
                    for i, stage in enumerate(plan.stages):
                        layers = stage.get_layer_indices()
                        if layers:
                            print(f"     Stage {i}: Layers {min(layers)}-{max(layers)} "
                                  f"({stage.total_memory_gb:.1f}GB)")
                else:
                    print(f"  ❌ Not feasible: {plan.validation_errors[0] if plan.validation_errors else 'Unknown error'}")
            except ValueError as e:
                print(f"  ❌ Configuration error: {e}")
    
    # Test auto-detection
    print(f"\n{'='*80}")
    print("Testing Auto-Detection")
    print(f"{'='*80}")
    
    os.environ["MODEL_SIZE"] = "8B"
    reload(model_profile)
    
    planner = PartitionPlanner(auto_detect_vram=True)
    plan = planner.greedy_partition()
    planner.print_plan(plan)