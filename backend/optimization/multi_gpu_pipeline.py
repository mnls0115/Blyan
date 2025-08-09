#!/usr/bin/env python3
"""
Multi-GPU Pipeline Parallelism for Blyan Network
Distributes model layers across GPUs for maximum throughput
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass
import logging
import time
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class GPUTopology:
    """GPU topology and capabilities."""
    gpu_id: int
    memory_gb: float
    compute_capability: Tuple[int, int]  # (major, minor)
    pcie_bandwidth_gbps: float
    nvlink_peers: List[int]  # GPUs connected via NVLink
    assigned_layers: List[int]  # Model layers assigned to this GPU

class PipelineStage:
    """Single stage in the pipeline (subset of layers on one GPU)."""
    
    def __init__(self, gpu_id: int, layers: nn.ModuleList, layer_indices: List[int]):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.layers = layers.to(self.device)
        self.layer_indices = layer_indices
        
        # Buffers for pipeline
        self.input_queue = asyncio.Queue(maxsize=10)
        self.output_queue = asyncio.Queue(maxsize=10)
        
        # Performance tracking
        self.process_times = deque(maxlen=100)
        self.gpu_utilization = deque(maxlen=100)
        
    async def process(self, input_tensor: torch.Tensor, batch_id: int) -> torch.Tensor:
        """Process input through this stage's layers."""
        start_time = time.time()
        
        # Move input to GPU
        x = input_tensor.to(self.device, non_blocking=True)
        
        # Forward through layers
        with torch.cuda.amp.autocast():  # Mixed precision for speed
            for layer in self.layers:
                x = layer(x)
        
        # Track performance
        process_time = time.time() - start_time
        self.process_times.append(process_time)
        
        # Record GPU utilization
        if torch.cuda.is_available():
            util = torch.cuda.utilization(self.gpu_id)
            self.gpu_utilization.append(util)
        
        return x
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage on this GPU."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0}
        
        torch.cuda.set_device(self.gpu_id)
        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(self.gpu_id).total_memory / 1e9
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "free": total - reserved
        }

class MultiGPUPipeline:
    """
    Multi-GPU pipeline parallelism coordinator.
    Distributes model layers across GPUs for maximum throughput.
    """
    
    def __init__(self, model: nn.Module, gpu_topology: List[GPUTopology]):
        self.model = model
        self.gpu_topology = gpu_topology
        self.num_gpus = len(gpu_topology)
        self.stages: List[PipelineStage] = []
        
        # Performance metrics
        self.total_throughput = 0
        self.batch_latencies = deque(maxlen=100)
        
        # Initialize pipeline stages
        self._distribute_layers()
        
    def _distribute_layers(self):
        """Distribute model layers across GPUs based on topology."""
        if not hasattr(self.model, 'layers'):
            logger.error("Model must have 'layers' attribute for pipeline parallelism")
            return
        
        layers = self.model.layers
        num_layers = len(layers)
        
        # Simple distribution strategy: equal layers per GPU
        # Can be optimized based on layer compute requirements
        layers_per_gpu = num_layers // self.num_gpus
        remainder = num_layers % self.num_gpus
        
        start_idx = 0
        for i, gpu_topo in enumerate(self.gpu_topology):
            # Add extra layer to first GPUs if remainder exists
            num_layers_for_gpu = layers_per_gpu + (1 if i < remainder else 0)
            end_idx = start_idx + num_layers_for_gpu
            
            # Get layers for this GPU
            gpu_layers = nn.ModuleList(layers[start_idx:end_idx])
            layer_indices = list(range(start_idx, end_idx))
            
            # Create pipeline stage
            stage = PipelineStage(
                gpu_id=gpu_topo.gpu_id,
                layers=gpu_layers,
                layer_indices=layer_indices
            )
            self.stages.append(stage)
            
            # Update topology
            gpu_topo.assigned_layers = layer_indices
            
            logger.info(f"GPU {gpu_topo.gpu_id}: Layers {start_idx}-{end_idx-1}")
            start_idx = end_idx
    
    async def forward_pipeline(self, input_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through pipeline with micro-batching.
        """
        batch_size = input_batch.size(0)
        micro_batch_size = max(1, batch_size // (self.num_gpus * 2))  # Overlap computation
        
        # Split into micro-batches
        micro_batches = torch.split(input_batch, micro_batch_size)
        outputs = []
        
        # Pipeline execution
        start_time = time.time()
        
        # Process micro-batches through pipeline
        for mb_idx, micro_batch in enumerate(micro_batches):
            x = micro_batch
            
            # Pass through each stage
            for stage_idx, stage in enumerate(self.stages):
                x = await stage.process(x, mb_idx)
                
                # Prefetch next micro-batch to next stage (pipeline parallelism)
                if stage_idx < len(self.stages) - 1 and mb_idx < len(micro_batches) - 1:
                    # This creates overlap between stages
                    await asyncio.sleep(0)  # Yield to allow other stages to process
            
            outputs.append(x)
        
        # Combine outputs
        output = torch.cat(outputs, dim=0)
        
        # Track latency
        latency = time.time() - start_time
        self.batch_latencies.append(latency)
        
        # Calculate throughput
        self.total_throughput = batch_size / latency
        
        return output
    
    def optimize_pipeline(self):
        """
        Dynamically optimize pipeline based on performance metrics.
        """
        # Analyze stage bottlenecks
        stage_times = []
        for stage in self.stages:
            if stage.process_times:
                avg_time = np.mean(stage.process_times)
                stage_times.append(avg_time)
            else:
                stage_times.append(0)
        
        if not stage_times:
            return
        
        # Find bottleneck stage
        bottleneck_idx = np.argmax(stage_times)
        bottleneck_time = stage_times[bottleneck_idx]
        
        # Calculate imbalance
        avg_time = np.mean(stage_times)
        imbalance = bottleneck_time / avg_time if avg_time > 0 else 1.0
        
        logger.info(f"Pipeline imbalance: {imbalance:.2f}x")
        logger.info(f"Bottleneck: Stage {bottleneck_idx} (GPU {self.stages[bottleneck_idx].gpu_id})")
        
        # Rebalancing strategy (if imbalance > threshold)
        if imbalance > 1.5:
            self._rebalance_layers(bottleneck_idx)
    
    def _rebalance_layers(self, bottleneck_idx: int):
        """
        Rebalance layers to reduce bottleneck.
        """
        logger.info(f"Rebalancing pipeline to address bottleneck at stage {bottleneck_idx}")
        
        # Strategy: Move one layer from bottleneck to adjacent stage with lowest load
        if bottleneck_idx > 0 and bottleneck_idx < len(self.stages) - 1:
            # Can move to either previous or next stage
            prev_stage = self.stages[bottleneck_idx - 1]
            next_stage = self.stages[bottleneck_idx + 1]
            
            prev_load = np.mean(prev_stage.process_times) if prev_stage.process_times else 0
            next_load = np.mean(next_stage.process_times) if next_stage.process_times else 0
            
            # Move to stage with lower load
            # (Implementation would require model architecture changes)
            target_stage = prev_stage if prev_load < next_load else next_stage
            logger.info(f"Would move layer to stage {self.stages.index(target_stage)}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = {
            "num_gpus": self.num_gpus,
            "throughput": self.total_throughput,
            "avg_latency": np.mean(self.batch_latencies) if self.batch_latencies else 0,
            "stages": []
        }
        
        for i, stage in enumerate(self.stages):
            stage_stats = {
                "gpu_id": stage.gpu_id,
                "layers": stage.layer_indices,
                "avg_process_time": np.mean(stage.process_times) if stage.process_times else 0,
                "avg_gpu_utilization": np.mean(stage.gpu_utilization) if stage.gpu_utilization else 0,
                "memory": stage.get_memory_usage()
            }
            stats["stages"].append(stage_stats)
        
        return stats

class GPUScheduler:
    """
    Intelligent GPU scheduling based on topology and workload.
    """
    
    def __init__(self):
        self.gpu_topology = self._detect_gpu_topology()
        self.request_queue = asyncio.Queue()
        self.gpu_assignments = {}
        
    def _detect_gpu_topology(self) -> List[GPUTopology]:
        """Detect available GPUs and their topology."""
        topology = []
        
        if not torch.cuda.is_available():
            logger.warning("No CUDA GPUs available")
            return topology
        
        num_gpus = torch.cuda.device_count()
        
        for gpu_id in range(num_gpus):
            props = torch.cuda.get_device_properties(gpu_id)
            
            # Detect NVLink peers (simplified - actual implementation would use nvidia-ml-py)
            nvlink_peers = []
            for peer_id in range(num_gpus):
                if peer_id != gpu_id:
                    # Check if P2P access is available
                    if torch.cuda.can_device_access_peer(gpu_id, peer_id):
                        nvlink_peers.append(peer_id)
            
            gpu_topo = GPUTopology(
                gpu_id=gpu_id,
                memory_gb=props.total_memory / 1e9,
                compute_capability=(props.major, props.minor),
                pcie_bandwidth_gbps=16.0,  # Typical PCIe 3.0 x16
                nvlink_peers=nvlink_peers,
                assigned_layers=[]
            )
            topology.append(gpu_topo)
            
            logger.info(
                f"GPU {gpu_id}: {props.name}, "
                f"{gpu_topo.memory_gb:.1f}GB, "
                f"CC {props.major}.{props.minor}, "
                f"NVLink peers: {nvlink_peers}"
            )
        
        return topology
    
    def assign_request_to_gpu(self, request_id: str, memory_required_gb: float) -> Optional[int]:
        """
        Assign a request to the most suitable GPU.
        """
        best_gpu = None
        min_utilization = float('inf')
        
        for gpu_topo in self.gpu_topology:
            # Check memory availability
            torch.cuda.set_device(gpu_topo.gpu_id)
            free_memory = (torch.cuda.get_device_properties(gpu_topo.gpu_id).total_memory - 
                          torch.cuda.memory_reserved(gpu_topo.gpu_id)) / 1e9
            
            if free_memory < memory_required_gb:
                continue
            
            # Check current utilization
            utilization = torch.cuda.utilization(gpu_topo.gpu_id)
            
            # Prefer GPU with lowest utilization
            if utilization < min_utilization:
                min_utilization = utilization
                best_gpu = gpu_topo.gpu_id
        
        if best_gpu is not None:
            self.gpu_assignments[request_id] = best_gpu
            logger.info(f"Assigned request {request_id} to GPU {best_gpu}")
        else:
            logger.warning(f"No GPU available for request {request_id} requiring {memory_required_gb}GB")
        
        return best_gpu
    
    def release_request(self, request_id: str):
        """Release GPU assignment for a completed request."""
        if request_id in self.gpu_assignments:
            gpu_id = self.gpu_assignments.pop(request_id)
            logger.info(f"Released GPU {gpu_id} from request {request_id}")

# Singleton instances
_pipeline_manager = None
_gpu_scheduler = None

def get_pipeline_manager(model: Optional[nn.Module] = None) -> Optional[MultiGPUPipeline]:
    """Get or create pipeline manager."""
    global _pipeline_manager
    
    if _pipeline_manager is None and model is not None:
        scheduler = get_gpu_scheduler()
        if scheduler.gpu_topology:
            _pipeline_manager = MultiGPUPipeline(model, scheduler.gpu_topology)
    
    return _pipeline_manager

def get_gpu_scheduler() -> GPUScheduler:
    """Get or create GPU scheduler."""
    global _gpu_scheduler
    
    if _gpu_scheduler is None:
        _gpu_scheduler = GPUScheduler()
    
    return _gpu_scheduler