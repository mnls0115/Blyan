"""
Dense Runtime Module - Clean implementation without MoE complexity
==================================================================
This module provides a straightforward dense model runtime that bypasses
all MoE routing/gating complexity while reusing existing infrastructure.
"""

# Dense module components
from .partition_planner import PartitionPlanner, ModelComponent, PipelineStage, PartitionPlan, create_optimal_plan
from .pipeline_coordinator import DistributedPipelineCoordinator, GPUNode, PipelineRequest

__all__ = [
    'PartitionPlanner', 
    'ModelComponent',
    'PipelineStage',
    'PartitionPlan',
    'create_optimal_plan',
    'DistributedPipelineCoordinator',
    'GPUNode',
    'PipelineRequest'
]