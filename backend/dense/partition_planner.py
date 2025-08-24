"""
Layer-Preserving Model Partition Planner
========================================
Partitions dense transformer models into pipeline stages that fit on target GPUs.
Each stage contains contiguous layers with configurable memory headroom.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from config.model_profile import (
    LAYERS, LAYER_MEMORY, PARTITION, 
    DISTRIBUTION_STRATEGIES, get_kv_cache_size
)

logger = logging.getLogger(__name__)


@dataclass
class LayerBlock:
    """Represents a block of model components."""
    block_type: str  # "embedding", "layer", "lm_head"
    layer_indices: List[int]  # Empty for embedding/lm_head
    memory_gb: float  # Weight memory in GB
    name: str  # Display name


@dataclass 
class PipelineStage:
    """Represents a pipeline stage on a GPU."""
    stage_id: int
    blocks: List[LayerBlock]
    total_memory_gb: float
    kv_cache_gb: float
    total_with_overhead_gb: float
    layer_range: Tuple[int, int]  # (start, end) layer indices


@dataclass
class PartitionPlan:
    """Complete partition plan for the model."""
    stages: List[PipelineStage]
    num_gpus: int
    precision: str
    target_vram_gb: float
    usable_vram_gb: float
    total_model_memory_gb: float
    feasible: bool
    error_message: Optional[str] = None


class PartitionPlanner:
    """Plans how to partition a dense model across multiple GPUs."""
    
    def __init__(self, 
                 target_vram_gb: float = 4.0,
                 precision: str = "int4",
                 reserved_headroom_gb: float = 1.0,
                 kv_cache_budget_gb: float = 0.6,
                 runtime_buffer_gb: float = 0.2,
                 max_seq_len: int = 2048,
                 batch_size: int = 1):
        """
        Initialize partition planner.
        
        Args:
            target_vram_gb: Total VRAM per GPU
            precision: Weight precision (int4, int8, fp16, fp32)
            reserved_headroom_gb: Reserved for CUDA context
            kv_cache_budget_gb: Reserved for KV cache
            runtime_buffer_gb: Reserved for temp buffers
            max_seq_len: Maximum sequence length
            batch_size: Batch size for inference
        """
        self.target_vram_gb = target_vram_gb
        self.precision = precision
        self.reserved_headroom_gb = reserved_headroom_gb
        self.kv_cache_budget_gb = kv_cache_budget_gb
        self.runtime_buffer_gb = runtime_buffer_gb
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        
        # Calculate usable VRAM for weights
        self.usable_vram_gb = (
            target_vram_gb - reserved_headroom_gb - 
            kv_cache_budget_gb - runtime_buffer_gb
        )
        
        if self.usable_vram_gb <= 0:
            raise ValueError(
                f"No usable VRAM! Target: {target_vram_gb}GB, "
                f"Reserved: {reserved_headroom_gb + kv_cache_budget_gb + runtime_buffer_gb}GB"
            )
        
        logger.info(
            f"Partition planner initialized: {target_vram_gb}GB total, "
            f"{self.usable_vram_gb:.2f}GB usable for weights"
        )
    
    def create_layer_blocks(self) -> List[LayerBlock]:
        """Create ordered list of model blocks with their memory requirements."""
        blocks = []
        layer_mem = LAYER_MEMORY[self.precision]
        
        # Embedding block
        blocks.append(LayerBlock(
            block_type="embedding",
            layer_indices=[],
            memory_gb=layer_mem["embedding"],
            name="Embedding"
        ))
        
        # Layer blocks
        for i in range(LAYERS["num_layers"]):
            blocks.append(LayerBlock(
                block_type="layer",
                layer_indices=[i],
                memory_gb=layer_mem["per_layer"],
                name=f"Layer {i}"
            ))
        
        # LM head block
        blocks.append(LayerBlock(
            block_type="lm_head",
            layer_indices=[],
            memory_gb=layer_mem["lm_head"],
            name="LM Head"
        ))
        
        return blocks
    
    def calculate_kv_cache_for_layers(self, num_layers: int) -> float:
        """Calculate KV cache size for given number of layers."""
        return get_kv_cache_size(
            batch_size=self.batch_size,
            seq_len=self.max_seq_len,
            num_layers=num_layers
        )
    
    def greedy_partition(self, max_layers_per_stage: Optional[int] = None) -> PartitionPlan:
        """
        Greedy partition algorithm that preserves layer order.
        
        Args:
            max_layers_per_stage: Optional cap on layers per stage
            
        Returns:
            PartitionPlan with stages
        """
        blocks = self.create_layer_blocks()
        stages = []
        current_stage_blocks = []
        current_stage_memory = 0.0
        current_stage_layers = 0
        stage_id = 0
        
        for block in blocks:
            block_memory = block.memory_gb
            block_layers = len(block.layer_indices) if block.block_type == "layer" else 0
            
            # Check if adding this block would exceed limits
            would_exceed_memory = (current_stage_memory + block_memory) > self.usable_vram_gb
            would_exceed_layer_cap = (
                max_layers_per_stage and 
                block_layers > 0 and 
                current_stage_layers + block_layers > max_layers_per_stage
            )
            
            # Check if single block exceeds capacity
            if block_memory > self.usable_vram_gb:
                return PartitionPlan(
                    stages=[],
                    num_gpus=0,
                    precision=self.precision,
                    target_vram_gb=self.target_vram_gb,
                    usable_vram_gb=self.usable_vram_gb,
                    total_model_memory_gb=sum(b.memory_gb for b in blocks),
                    feasible=False,
                    error_message=(
                        f"{block.name} requires {block_memory:.2f}GB but only "
                        f"{self.usable_vram_gb:.2f}GB available. Try: "
                        f"1) Increase target_vram_gb, 2) Use higher quantization (int4), "
                        f"3) Reduce reserved memory"
                    )
                )
            
            # Start new stage if needed
            if current_stage_blocks and (would_exceed_memory or would_exceed_layer_cap):
                # Finalize current stage
                layer_indices = []
                for b in current_stage_blocks:
                    if b.block_type == "layer":
                        layer_indices.extend(b.layer_indices)
                
                layer_range = (min(layer_indices), max(layer_indices)) if layer_indices else (-1, -1)
                kv_cache = self.calculate_kv_cache_for_layers(len(layer_indices))
                
                stages.append(PipelineStage(
                    stage_id=stage_id,
                    blocks=current_stage_blocks,
                    total_memory_gb=current_stage_memory,
                    kv_cache_gb=kv_cache,
                    total_with_overhead_gb=(
                        current_stage_memory + kv_cache + 
                        self.reserved_headroom_gb + self.runtime_buffer_gb
                    ),
                    layer_range=layer_range
                ))
                
                # Reset for new stage
                stage_id += 1
                current_stage_blocks = []
                current_stage_memory = 0.0
                current_stage_layers = 0
            
            # Add block to current stage
            current_stage_blocks.append(block)
            current_stage_memory += block_memory
            current_stage_layers += block_layers
        
        # Finalize last stage
        if current_stage_blocks:
            layer_indices = []
            for b in current_stage_blocks:
                if b.block_type == "layer":
                    layer_indices.extend(b.layer_indices)
            
            layer_range = (min(layer_indices), max(layer_indices)) if layer_indices else (-1, -1)
            kv_cache = self.calculate_kv_cache_for_layers(len(layer_indices))
            
            stages.append(PipelineStage(
                stage_id=stage_id,
                blocks=current_stage_blocks,
                total_memory_gb=current_stage_memory,
                kv_cache_gb=kv_cache,
                total_with_overhead_gb=(
                    current_stage_memory + kv_cache + 
                    self.reserved_headroom_gb + self.runtime_buffer_gb
                ),
                layer_range=layer_range
            ))
        
        total_model_memory = sum(b.memory_gb for b in blocks)
        
        return PartitionPlan(
            stages=stages,
            num_gpus=len(stages),
            precision=self.precision,
            target_vram_gb=self.target_vram_gb,
            usable_vram_gb=self.usable_vram_gb,
            total_model_memory_gb=total_model_memory,
            feasible=True
        )
    
    def get_predefined_strategy(self, strategy_name: str) -> Optional[PartitionPlan]:
        """Get a predefined distribution strategy."""
        if strategy_name not in DISTRIBUTION_STRATEGIES:
            return None
        
        strategy = DISTRIBUTION_STRATEGIES[strategy_name]
        
        # Check if this GPU can handle the strategy
        if self.target_vram_gb < strategy["min_vram_gb"]:
            return PartitionPlan(
                stages=[],
                num_gpus=strategy["num_stages"],
                precision=strategy["precision"],
                target_vram_gb=self.target_vram_gb,
                usable_vram_gb=self.usable_vram_gb,
                total_model_memory_gb=0,
                feasible=False,
                error_message=(
                    f"Strategy '{strategy_name}' requires {strategy['min_vram_gb']}GB VRAM "
                    f"but only {self.target_vram_gb}GB available"
                )
            )
        
        # Create stages based on strategy
        blocks = self.create_layer_blocks()
        stages = []
        block_idx = 0
        
        for stage_id, num_layers in enumerate(strategy["layers_per_stage"]):
            stage_blocks = []
            stage_memory = 0.0
            
            # Add embedding to first stage
            if stage_id == 0:
                stage_blocks.append(blocks[0])  # Embedding
                stage_memory += blocks[0].memory_gb
                block_idx = 1
            
            # Add layers
            layer_start = sum(strategy["layers_per_stage"][:stage_id])
            layer_end = layer_start + num_layers
            
            for layer_idx in range(layer_start, layer_end):
                if block_idx < len(blocks) and blocks[block_idx].block_type == "layer":
                    stage_blocks.append(blocks[block_idx])
                    stage_memory += blocks[block_idx].memory_gb
                    block_idx += 1
            
            # Add lm_head to last stage
            if stage_id == len(strategy["layers_per_stage"]) - 1:
                stage_blocks.append(blocks[-1])  # LM head
                stage_memory += blocks[-1].memory_gb
            
            kv_cache = self.calculate_kv_cache_for_layers(num_layers)
            
            stages.append(PipelineStage(
                stage_id=stage_id,
                blocks=stage_blocks,
                total_memory_gb=stage_memory,
                kv_cache_gb=kv_cache,
                total_with_overhead_gb=(
                    stage_memory + kv_cache + 
                    self.reserved_headroom_gb + self.runtime_buffer_gb
                ),
                layer_range=(layer_start, layer_end - 1)
            ))
        
        return PartitionPlan(
            stages=stages,
            num_gpus=strategy["num_stages"],
            precision=strategy["precision"],
            target_vram_gb=self.target_vram_gb,
            usable_vram_gb=self.usable_vram_gb,
            total_model_memory_gb=sum(b.memory_gb for b in blocks),
            feasible=True
        )
    
    def print_plan(self, plan: PartitionPlan):
        """Pretty print the partition plan."""
        if not plan.feasible:
            print(f"❌ Partition plan not feasible: {plan.error_message}")
            return
        
        print(f"\n📊 Partition Plan")
        print(f"=" * 60)
        print(f"Model: Qwen3-8B ({plan.total_model_memory_gb:.2f}GB in {plan.precision})")
        print(f"Target VRAM: {plan.target_vram_gb}GB per GPU")
        print(f"Usable VRAM: {plan.usable_vram_gb:.2f}GB for weights")
        print(f"Number of GPUs: {plan.num_gpus}")
        print(f"\nStages:")
        print(f"-" * 60)
        
        for stage in plan.stages:
            print(f"\n🖥️  Stage {stage.stage_id} (GPU {stage.stage_id}):")
            
            # Group blocks by type
            has_embedding = any(b.block_type == "embedding" for b in stage.blocks)
            has_lm_head = any(b.block_type == "lm_head" for b in stage.blocks)
            layer_blocks = [b for b in stage.blocks if b.block_type == "layer"]
            
            components = []
            if has_embedding:
                components.append("Embedding")
            if layer_blocks:
                if stage.layer_range[0] >= 0:
                    components.append(f"Layers {stage.layer_range[0]}-{stage.layer_range[1]}")
            if has_lm_head:
                components.append("LM Head")
            
            print(f"  Components: {', '.join(components)}")
            print(f"  Weight memory: {stage.total_memory_gb:.3f}GB")
            print(f"  KV cache: {stage.kv_cache_gb:.3f}GB")
            print(f"  Total (with overhead): {stage.total_with_overhead_gb:.3f}GB")
            
            if stage.total_with_overhead_gb > plan.target_vram_gb:
                print(f"  ⚠️  WARNING: Exceeds target VRAM!")
        
        print(f"\n{'=' * 60}")


def create_optimal_plan(target_vram_gb: float = 4.0, 
                        precision: str = "int4",
                        strategy: Optional[str] = None) -> PartitionPlan:
    """
    Create an optimal partition plan for the given GPU.
    
    Args:
        target_vram_gb: GPU VRAM in GB
        precision: Weight precision
        strategy: Optional predefined strategy name
        
    Returns:
        Optimal PartitionPlan
    """
    planner = PartitionPlanner(
        target_vram_gb=target_vram_gb,
        precision=precision,
        reserved_headroom_gb=PARTITION["default"]["reserved_headroom_gb"],
        kv_cache_budget_gb=PARTITION["default"]["kv_cache_budget_gb"],
        runtime_buffer_gb=PARTITION["default"]["runtime_buffer_gb"]
    )
    
    if strategy:
        plan = planner.get_predefined_strategy(strategy)
        if plan and plan.feasible:
            return plan
    
    # Use greedy algorithm
    return planner.greedy_partition()


if __name__ == "__main__":
    # Demo different partition strategies
    print("\n🚀 Partition Planning Demo\n")
    
    # Small GPU (4GB)
    print("=" * 60)
    print("Small GPU (4GB) with int4:")
    planner = PartitionPlanner(target_vram_gb=4.0, precision="int4")
    plan = planner.greedy_partition()
    planner.print_plan(plan)
    
    # Medium GPU (8GB)
    print("\n" + "=" * 60)
    print("Medium GPU (8GB) with int8:")
    planner = PartitionPlanner(target_vram_gb=8.0, precision="int8")
    plan = planner.greedy_partition()
    planner.print_plan(plan)
    
    # Large GPU (16GB)
    print("\n" + "=" * 60)
    print("Large GPU (16GB) with fp16:")
    planner = PartitionPlanner(target_vram_gb=16.0, precision="fp16")
    plan = planner.greedy_partition()
    planner.print_plan(plan)
    
    # Predefined strategy
    print("\n" + "=" * 60)
    print("Using 'triple' strategy (3 GPUs):")
    planner = PartitionPlanner(target_vram_gb=4.0, precision="int4")
    plan = planner.get_predefined_strategy("triple")
    planner.print_plan(plan)