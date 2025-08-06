#!/usr/bin/env python3
"""Dual model instance management with CUDA stream separation."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging
import copy
import time
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Configuration for CUDA stream management."""
    inference_stream_priority: int = -1  # Higher priority
    learning_stream_priority: int = 1    # Lower priority
    enable_stream_synchronization: bool = True
    memory_pool_size_mb: int = 1024
    enable_graph_capture: bool = False  # CUDA graphs for inference

class DualModelManager:
    """Manages dual model instances with stream separation for concurrent execution."""
    
    def __init__(
        self,
        base_model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        stream_config: StreamConfig = None
    ):
        self.device = torch.device(device)
        self.stream_config = stream_config or StreamConfig()
        self.base_model = base_model
        
        # Initialize models
        self._initialize_models()
        
        # Initialize CUDA streams if available
        self._initialize_streams()
        
        # Memory management
        self.memory_allocated = {}
        self._setup_memory_pools()
        
        # Performance tracking
        self.stream_timings = {"inference": [], "learning": []}
        
    def _initialize_models(self):
        """Initialize inference and learning model instances."""
        logger.info("Initializing dual model instances...")
        
        # Learning model - full precision with gradients
        self.learning_model = self.base_model.to(self.device)
        self.learning_model.train()
        
        # Inference model - optimized copy without gradients
        self.inference_model = self._create_inference_model()
        
        # Track memory usage
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            self.memory_allocated["learning"] = torch.cuda.memory_allocated() / 1024 / 1024
            
            # Clear cache before creating inference model
            torch.cuda.empty_cache()
            
    def _create_inference_model(self) -> nn.Module:
        """Create optimized inference model."""
        # Clone model structure
        inference_model = copy.deepcopy(self.base_model)
        
        # Move to device and optimize
        inference_model = inference_model.to(self.device)
        inference_model.eval()
        
        # Disable gradients
        for param in inference_model.parameters():
            param.requires_grad = False
            
        # Optional: Convert to half precision for inference
        if self.device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported"):
            if torch.cuda.is_bf16_supported():
                inference_model = inference_model.to(torch.bfloat16)
                logger.info("Inference model using bfloat16")
            else:
                inference_model = inference_model.half()
                logger.info("Inference model using float16")
                
        # Optional: Compile with torch.compile for faster inference
        if hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                inference_model = torch.compile(inference_model, mode="reduce-overhead")
                logger.info("Inference model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
                
        return inference_model
        
    def _initialize_streams(self):
        """Initialize CUDA streams for concurrent execution."""
        if self.device.type != "cuda":
            self.inference_stream = None
            self.learning_stream = None
            return
            
        # Create streams with different priorities
        self.inference_stream = torch.cuda.Stream(
            priority=self.stream_config.inference_stream_priority
        )
        self.learning_stream = torch.cuda.Stream(
            priority=self.stream_config.learning_stream_priority
        )
        
        # Default stream for coordination
        self.default_stream = torch.cuda.current_stream()
        
        logger.info(
            f"Initialized CUDA streams - "
            f"Inference priority: {self.stream_config.inference_stream_priority}, "
            f"Learning priority: {self.stream_config.learning_stream_priority}"
        )
        
    def _setup_memory_pools(self):
        """Setup memory pools for efficient allocation."""
        if self.device.type != "cuda":
            return
            
        # Set memory fraction to prevent OOM
        memory_fraction = 0.85  # Use 85% of available memory
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        # Enable memory pool if available
        if hasattr(torch.cuda, "set_allocator_settings"):
            torch.cuda.set_allocator_settings("expandable_segments:True")
            
        logger.info(f"Memory pool configured with {memory_fraction*100}% allocation")
        
    async def inference(
        self,
        input_data: Dict[str, torch.Tensor],
        use_graph: bool = False
    ) -> Dict[str, Any]:
        """Run inference on dedicated stream."""
        start_time = time.time()
        
        if self.device.type == "cuda" and self.inference_stream:
            with torch.cuda.stream(self.inference_stream):
                result = await self._run_inference(input_data, use_graph)
        else:
            result = await self._run_inference(input_data, use_graph)
            
        # Track timing
        inference_time = (time.time() - start_time) * 1000
        self.stream_timings["inference"].append(inference_time)
        
        return {
            "output": result,
            "inference_time_ms": inference_time,
            "stream": "inference"
        }
        
    async def _run_inference(
        self,
        input_data: Dict[str, torch.Tensor],
        use_graph: bool
    ) -> torch.Tensor:
        """Execute inference with optional CUDA graph optimization."""
        import asyncio
        
        # Move inputs to device with inference dtype
        processed_inputs = {}
        for key, tensor in input_data.items():
            if self.inference_model.dtype == torch.float16:
                processed_inputs[key] = tensor.to(self.device, dtype=torch.float16)
            elif self.inference_model.dtype == torch.bfloat16:
                processed_inputs[key] = tensor.to(self.device, dtype=torch.bfloat16)
            else:
                processed_inputs[key] = tensor.to(self.device)
                
        # Run inference
        with torch.no_grad():
            if use_graph and self.stream_config.enable_graph_capture:
                # CUDA graph capture for repeated inference
                if not hasattr(self, "_inference_graph"):
                    self._capture_inference_graph(processed_inputs)
                    
                # Replay graph
                self._inference_graph.replay()
                output = self._graph_output.clone()
            else:
                # Regular inference
                output = await asyncio.to_thread(
                    self.inference_model,
                    **processed_inputs
                )
                
        return output
        
    def _capture_inference_graph(self, sample_inputs: Dict[str, torch.Tensor]):
        """Capture CUDA graph for inference optimization."""
        if self.device.type != "cuda":
            return
            
        logger.info("Capturing CUDA graph for inference...")
        
        # Warmup
        for _ in range(3):
            _ = self.inference_model(**sample_inputs)
            
        # Capture graph
        self._inference_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._inference_graph):
            self._graph_output = self.inference_model(**sample_inputs)
            
        logger.info("CUDA graph captured successfully")
        
    async def learning_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 1
    ) -> Dict[str, Any]:
        """Execute learning step on dedicated stream."""
        start_time = time.time()
        
        if self.device.type == "cuda" and self.learning_stream:
            with torch.cuda.stream(self.learning_stream):
                result = await self._run_learning_step(batch, optimizer, accumulation_steps)
        else:
            result = await self._run_learning_step(batch, optimizer, accumulation_steps)
            
        # Track timing
        learning_time = (time.time() - start_time) * 1000
        self.stream_timings["learning"].append(learning_time)
        
        result["learning_time_ms"] = learning_time
        result["stream"] = "learning"
        
        return result
        
    async def _run_learning_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int
    ) -> Dict[str, Any]:
        """Execute actual learning step."""
        import asyncio
        
        # Move batch to device
        device_batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = await asyncio.to_thread(self.learning_model, **device_batch)
        loss = outputs.loss / accumulation_steps
        
        # Backward pass
        await asyncio.to_thread(loss.backward)
        
        # Gradient metrics
        total_norm = 0
        for p in self.learning_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        return {
            "loss": loss.item() * accumulation_steps,
            "gradient_norm": total_norm,
            "batch_size": next(iter(device_batch.values())).size(0)
        }
        
    def synchronize_models(self, learning_to_inference: bool = True):
        """Synchronize weights between models."""
        if learning_to_inference:
            # Copy learning weights to inference model
            source_state = self.learning_model.state_dict()
            
            # Convert dtype if necessary
            if self.inference_model.dtype != self.learning_model.dtype:
                converted_state = {}
                for key, value in source_state.items():
                    if isinstance(value, torch.Tensor):
                        converted_state[key] = value.to(dtype=self.inference_model.dtype)
                    else:
                        converted_state[key] = value
                source_state = converted_state
                
            self.inference_model.load_state_dict(source_state)
            logger.info("Synchronized learning model -> inference model")
        else:
            # Copy inference weights to learning model (rare case)
            self.learning_model.load_state_dict(self.inference_model.state_dict())
            logger.info("Synchronized inference model -> learning model")
            
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage by each model."""
        if self.device.type != "cuda":
            return {"total_mb": 0, "learning_mb": 0, "inference_mb": 0}
            
        torch.cuda.synchronize()
        
        return {
            "total_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "learning_model_params": sum(p.numel() for p in self.learning_model.parameters()) / 1e6,
            "inference_model_params": sum(p.numel() for p in self.inference_model.parameters()) / 1e6
        }
        
    def get_stream_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for each stream."""
        stats = {}
        
        for stream_name, timings in self.stream_timings.items():
            if timings:
                stats[stream_name] = {
                    "count": len(timings),
                    "avg_ms": sum(timings) / len(timings),
                    "min_ms": min(timings),
                    "max_ms": max(timings),
                    "recent_ms": timings[-10:]  # Last 10 timings
                }
            else:
                stats[stream_name] = {"count": 0}
                
        # Add stream concurrency info
        if self.device.type == "cuda":
            stats["cuda_streams_enabled"] = True
            stats["stream_priorities"] = {
                "inference": self.stream_config.inference_stream_priority,
                "learning": self.stream_config.learning_stream_priority
            }
        else:
            stats["cuda_streams_enabled"] = False
            
        return stats

# Production code - demo removed