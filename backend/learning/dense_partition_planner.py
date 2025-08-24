"""
Dense Model Partition Planner for Layer-Sharded Pipeline Training
==================================================================
Model-agnostic partition planning that scales from 8B to 100B+ models.
Uses model profiles to dynamically assign layers to heterogeneous GPUs.

Core principle: Each GPU gets contiguous layers based on VRAM and compute.
Supports automatic LoRA/Dense routing based on GPU capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class TrainingMode(Enum):
    """Training mode based on GPU capabilities."""
    DENSE = "dense"  # Full parameter updates
    LORA = "lora"   # Low-rank adaptation
    QLORA = "qlora"  # Quantized LoRA
    SPECULATIVE = "speculative"  # Draft model for speculative decoding
    VALIDATOR = "validator"  # PoL validation only


@dataclass
class DeviceProfile:
    """Profile of a GPU device's capabilities."""
    device_id: str
    vram_gb: float
    tflops: float  # Theoretical FLOPs
    net_bandwidth_gbps: float  # Network bandwidth
    compute_capability: Tuple[int, int]  # CUDA compute capability
    is_nvlink: bool = False  # NVLink available for fast inter-GPU
    reliability_score: float = 1.0  # Historical reliability (0-1)
    
    @property
    def effective_vram_gb(self) -> float:
        """Calculate effective VRAM after system overhead."""
        # Reserve ~10% for CUDA context and kernels
        return self.vram_gb * 0.9
    
    def can_handle_layers(self, layer_memory_gb: float, num_layers: int, 
                         headroom_gb: float = 1.0) -> bool:
        """Check if device can handle given layers."""
        required = layer_memory_gb * num_layers + headroom_gb
        return required <= self.effective_vram_gb


@dataclass 
class LayerAssignment:
    """Assignment of layers to a device."""
    device_id: str
    layer_indices: List[int]  # e.g., [0, 1, 2] for first 3 layers
    layer_names: List[str]  # e.g., ["embedding", "layer_0", "layer_1"]
    memory_usage_gb: float
    compute_load: float  # Estimated FLOPs
    training_mode: TrainingMode
    precision: str  # fp32, fp16, int8, int4
    
    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)
    
    @property
    def is_first_stage(self) -> bool:
        return 0 in self.layer_indices or "embedding" in self.layer_names
    
    @property 
    def is_last_stage(self) -> bool:
        return "lm_head" in self.layer_names


@dataclass
class PartitionPlan:
    """Complete partition plan for distributed training."""
    model_name: str
    total_layers: int
    assignments: List[LayerAssignment]
    total_memory_gb: float
    total_compute_gflops: float
    pipeline_stages: int
    estimated_throughput: float  # tokens/sec
    bottleneck_stage: Optional[str] = None
    warnings: List[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "total_layers": self.total_layers,
            "assignments": [asdict(a) for a in self.assignments],
            "total_memory_gb": self.total_memory_gb,
            "total_compute_gflops": self.total_compute_gflops,
            "pipeline_stages": self.pipeline_stages,
            "estimated_throughput": self.estimated_throughput,
            "bottleneck_stage": self.bottleneck_stage,
            "warnings": self.warnings or []
        }
    
    def get_device_assignment(self, device_id: str) -> Optional[LayerAssignment]:
        """Get assignment for specific device."""
        for assignment in self.assignments:
            if assignment.device_id == device_id:
                return assignment
        return None


class DensePartitionPlanner:
    """
    Planner for partitioning dense models across heterogeneous GPUs.
    Model-agnostic design that uses profile configurations.
    """
    
    def __init__(self, model_profile_path: Optional[Path] = None):
        """
        Initialize planner with model profile.
        
        Args:
            model_profile_path: Path to model profile JSON or Python module
        """
        self.model_profile = self._load_model_profile(model_profile_path)
        self.layer_memory_cache = {}
        
    def _load_model_profile(self, profile_path: Optional[Path]) -> Dict:
        """Load model profile from file or use default."""
        if profile_path and profile_path.exists():
            if profile_path.suffix == '.json':
                with open(profile_path) as f:
                    return json.load(f)
            else:
                # Import Python module
                import importlib.util
                spec = importlib.util.spec_from_file_location("model_profile", profile_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.get_model_config()
        else:
            # Use default profile
            try:
                from config.model_profile import get_model_config
                return get_model_config()
            except ImportError:
                # Fallback to minimal config
                return self._get_default_profile()
    
    def _get_default_profile(self) -> Dict:
        """Get default model profile for testing."""
        return {
            "model_name": "dense-model",
            "layers": {
                "num_layers": 32,
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_attention_heads": 32,
                "vocab_size": 32000
            },
            "layer_memory": {
                "fp16": {
                    "per_layer": 0.5,
                    "embedding": 0.25,
                    "lm_head": 0.25
                },
                "int8": {
                    "per_layer": 0.25,
                    "embedding": 0.125,
                    "lm_head": 0.125
                },
                "int4": {
                    "per_layer": 0.125,
                    "embedding": 0.0625,
                    "lm_head": 0.0625
                }
            }
        }
    
    def _calculate_layer_memory(self, layer_type: str, precision: str) -> float:
        """
        Calculate memory requirement for a layer.
        
        Args:
            layer_type: Type of layer (embedding, layer_N, lm_head)
            precision: Precision (fp32, fp16, int8, int4)
            
        Returns:
            Memory requirement in GB
        """
        cache_key = f"{layer_type}:{precision}"
        if cache_key in self.layer_memory_cache:
            return self.layer_memory_cache[cache_key]
        
        memory_config = self.model_profile.get("layer_memory", {}).get(precision, {})
        
        if layer_type == "embedding":
            mem = memory_config.get("embedding", 0.5)
        elif layer_type == "lm_head":
            mem = memory_config.get("lm_head", 0.5)
        else:
            mem = memory_config.get("per_layer", 0.5)
        
        self.layer_memory_cache[cache_key] = mem
        return mem
    
    def _calculate_layer_flops(self, layer_type: str, batch_size: int = 1, 
                              seq_len: int = 512) -> float:
        """
        Calculate FLOPs for a layer.
        
        Args:
            layer_type: Type of layer
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            FLOPs (in GFLOPs)
        """
        layers_config = self.model_profile.get("layers", {})
        hidden_size = layers_config.get("hidden_size", 4096)
        intermediate_size = layers_config.get("intermediate_size", 11008)
        
        if layer_type == "embedding":
            # Embedding lookup is memory-bound, not compute
            return 0.001 * batch_size * seq_len
        elif layer_type == "lm_head":
            # Output projection
            vocab_size = layers_config.get("vocab_size", 32000)
            return 2 * batch_size * seq_len * hidden_size * vocab_size / 1e9
        else:
            # Transformer layer: attention + FFN
            # Attention: 4 * batch * seq^2 * hidden
            attention_flops = 4 * batch_size * seq_len * seq_len * hidden_size
            # FFN: 2 * batch * seq * hidden * intermediate * 2
            ffn_flops = 4 * batch_size * seq_len * hidden_size * intermediate_size
            return (attention_flops + ffn_flops) / 1e9
    
    def _select_precision(self, vram_gb: float, num_layers: int) -> str:
        """
        Select optimal precision based on VRAM and layer count.
        
        Args:
            vram_gb: Available VRAM
            num_layers: Number of layers to fit
            
        Returns:
            Precision string (fp16, int8, int4)
        """
        precisions = ["fp16", "int8", "int4"]
        
        for precision in precisions:
            layer_mem = self._calculate_layer_memory("layer", precision)
            total_mem = layer_mem * num_layers + 1.0  # +1GB headroom
            if total_mem <= vram_gb:
                return precision
        
        return "int4"  # Fallback to most aggressive quantization
    
    def _assign_training_mode(self, device: DeviceProfile, 
                             assigned_layers: int) -> TrainingMode:
        """
        Determine training mode based on device capabilities.
        
        Args:
            device: Device profile
            assigned_layers: Number of layers assigned
            
        Returns:
            Appropriate training mode
        """
        # Small GPUs (< 8GB) with few layers -> LoRA
        if device.vram_gb < 8 and assigned_layers <= 2:
            return TrainingMode.QLORA if device.vram_gb < 6 else TrainingMode.LORA
        
        # Medium GPUs (8-16GB) with reasonable layers -> Dense
        if 8 <= device.vram_gb < 16 and assigned_layers >= 3:
            return TrainingMode.DENSE
        
        # Large GPUs (16GB+) -> Always Dense
        if device.vram_gb >= 16:
            return TrainingMode.DENSE
        
        # Very small GPUs (< 4GB) -> Speculative or Validator
        if device.vram_gb < 4:
            if device.tflops > 5:  # Decent compute
                return TrainingMode.SPECULATIVE
            else:
                return TrainingMode.VALIDATOR
        
        # Default to LoRA for edge cases
        return TrainingMode.LORA
    
    def plan_partition(
        self,
        devices: List[DeviceProfile],
        target_batch_size: int = 1,
        gradient_accumulation: int = 8,
        activation_checkpointing: bool = True,
        force_mode: Optional[TrainingMode] = None
    ) -> PartitionPlan:
        """
        Create partition plan for given devices.
        
        Args:
            devices: List of available devices
            target_batch_size: Target batch size per device
            gradient_accumulation: Gradient accumulation steps
            activation_checkpointing: Use activation checkpointing
            force_mode: Force specific training mode
            
        Returns:
            Complete partition plan
        """
        # Sort devices by VRAM (largest first for better load balancing)
        devices = sorted(devices, key=lambda d: d.vram_gb, reverse=True)
        
        # Get model configuration
        num_layers = self.model_profile["layers"]["num_layers"]
        model_name = self.model_profile.get("model_name", "unknown")
        
        assignments = []
        assigned_layers = []
        warnings = []
        
        # Reserve layers for special components
        special_layers = ["embedding", "lm_head"]
        regular_layers = [f"layer_{i}" for i in range(num_layers)]
        all_layers = ["embedding"] + regular_layers + ["lm_head"]
        
        # Calculate how many layers each device can handle
        remaining_layers = all_layers.copy()
        
        for device in devices:
            if not remaining_layers:
                warnings.append(f"Device {device.device_id} not needed (all layers assigned)")
                continue
            
            # Determine precision based on VRAM
            precision = self._select_precision(
                device.effective_vram_gb,
                len(remaining_layers)
            )
            
            # Calculate how many layers this device can handle
            layer_memory = self._calculate_layer_memory("layer", precision)
            
            # Account for training overhead
            overhead_multiplier = 1.0
            if activation_checkpointing:
                overhead_multiplier = 1.5  # Less memory for activations
            else:
                overhead_multiplier = 2.5  # More memory needed
            
            # Add optimizer state overhead (Adam needs 2x params)
            optimizer_overhead = 2.0 if force_mode != TrainingMode.LORA else 1.1
            total_multiplier = overhead_multiplier * optimizer_overhead
            
            available_memory = device.effective_vram_gb - 1.0  # Reserve 1GB
            max_layers = int(available_memory / (layer_memory * total_multiplier))
            max_layers = max(1, min(max_layers, len(remaining_layers)))
            
            # Assign layers to this device
            device_layers = remaining_layers[:max_layers]
            remaining_layers = remaining_layers[max_layers:]
            
            # Calculate actual memory usage
            total_memory = sum(
                self._calculate_layer_memory(
                    "embedding" if "embedding" in l else "lm_head" if "lm_head" in l else "layer",
                    precision
                )
                for l in device_layers
            ) * total_multiplier
            
            # Calculate compute load
            compute_load = sum(
                self._calculate_layer_flops(
                    "embedding" if "embedding" in l else "lm_head" if "lm_head" in l else "layer",
                    target_batch_size
                )
                for l in device_layers
            )
            
            # Determine training mode
            mode = force_mode or self._assign_training_mode(device, len(device_layers))
            
            # Create assignment
            layer_indices = []
            for layer in device_layers:
                if layer == "embedding":
                    layer_indices.append(-1)  # Special index for embedding
                elif layer == "lm_head":
                    layer_indices.append(num_layers)  # Special index for lm_head
                else:
                    layer_idx = int(layer.split("_")[1])
                    layer_indices.append(layer_idx)
            
            assignment = LayerAssignment(
                device_id=device.device_id,
                layer_indices=layer_indices,
                layer_names=device_layers,
                memory_usage_gb=total_memory,
                compute_load=compute_load,
                training_mode=mode,
                precision=precision
            )
            
            assignments.append(assignment)
            assigned_layers.extend(device_layers)
            
            logger.info(f"Assigned {len(device_layers)} layers to {device.device_id} "
                       f"({mode.value} mode, {precision} precision)")
        
        # Check if all layers are assigned
        if remaining_layers:
            warnings.append(f"Unassigned layers: {remaining_layers}")
            logger.warning(f"Could not assign all layers: {remaining_layers}")
        
        # Identify bottleneck stage (slowest)
        if assignments:
            slowest = max(assignments, key=lambda a: a.compute_load / devices[0].tflops)
            bottleneck = slowest.device_id
        else:
            bottleneck = None
        
        # Calculate total resources
        total_memory = sum(a.memory_usage_gb for a in assignments)
        total_compute = sum(a.compute_load for a in assignments)
        
        # Estimate throughput (limited by slowest stage)
        if assignments:
            stage_throughputs = [
                devices[i].tflops / max(a.compute_load, 0.001)
                for i, a in enumerate(assignments[:len(devices)])
            ]
            estimated_throughput = min(stage_throughputs) * target_batch_size
        else:
            estimated_throughput = 0
        
        return PartitionPlan(
            model_name=model_name,
            total_layers=len(all_layers),
            assignments=assignments,
            total_memory_gb=total_memory,
            total_compute_gflops=total_compute,
            pipeline_stages=len(assignments),
            estimated_throughput=estimated_throughput,
            bottleneck_stage=bottleneck,
            warnings=warnings
        )
    
    def validate_plan(self, plan: PartitionPlan) -> List[str]:
        """
        Validate a partition plan for correctness.
        
        Args:
            plan: Partition plan to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check all layers are assigned
        all_layers = set()
        for assignment in plan.assignments:
            all_layers.update(assignment.layer_names)
        
        expected_layers = set(["embedding", "lm_head"])
        expected_layers.update(f"layer_{i}" for i in range(plan.total_layers - 2))
        
        missing = expected_layers - all_layers
        if missing:
            errors.append(f"Missing layers: {missing}")
        
        duplicate = [l for l in all_layers if 
                    sum(1 for a in plan.assignments if l in a.layer_names) > 1]
        if duplicate:
            errors.append(f"Duplicate layer assignments: {duplicate}")
        
        # Check memory constraints
        for assignment in plan.assignments:
            if assignment.memory_usage_gb > 80:  # Sanity check
                errors.append(f"Unrealistic memory usage for {assignment.device_id}: "
                            f"{assignment.memory_usage_gb:.1f}GB")
        
        # Check for empty assignments
        empty = [a.device_id for a in plan.assignments if not a.layer_names]
        if empty:
            errors.append(f"Empty assignments: {empty}")
        
        return errors
    
    def rebalance_plan(
        self,
        plan: PartitionPlan,
        performance_metrics: Dict[str, float]
    ) -> PartitionPlan:
        """
        Rebalance plan based on observed performance.
        
        Args:
            plan: Current partition plan
            performance_metrics: Measured step times per device
            
        Returns:
            Rebalanced partition plan
        """
        # TODO: Implement dynamic rebalancing based on actual performance
        # For now, return original plan
        return plan


def create_partition_plan(
    devices: List[Dict[str, Any]],
    model_profile_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to create partition plan from device specs.
    
    Args:
        devices: List of device specifications
        model_profile_path: Path to model profile
        **kwargs: Additional arguments for planning
        
    Returns:
        Partition plan as dictionary
    """
    # Convert device dicts to DeviceProfile objects
    device_profiles = []
    for dev in devices:
        profile = DeviceProfile(
            device_id=dev["device_id"],
            vram_gb=dev["vram_gb"],
            tflops=dev.get("tflops", 10.0),
            net_bandwidth_gbps=dev.get("bandwidth", 10.0),
            compute_capability=dev.get("compute_capability", (7, 0)),
            is_nvlink=dev.get("nvlink", False),
            reliability_score=dev.get("reliability", 1.0)
        )
        device_profiles.append(profile)
    
    # Create planner and generate plan
    planner = DensePartitionPlanner(
        Path(model_profile_path) if model_profile_path else None
    )
    
    plan = planner.plan_partition(device_profiles, **kwargs)
    
    # Validate plan
    errors = planner.validate_plan(plan)
    if errors:
        logger.warning(f"Partition plan validation errors: {errors}")
    
    return plan.to_dict()