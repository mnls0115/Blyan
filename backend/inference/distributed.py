"""
Centralized Distributed Inference
==================================
Single entry point for all distributed inference operations.
Eliminates duplication across multiple coordinators.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DistributedRequest:
    """Unified distributed inference request."""
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    request_id: Optional[str] = None
    user_address: Optional[str] = None


class CentralizedDistributedInference:
    """Single coordinator for all distributed inference."""
    
    def __init__(self, coordinator=None):
        """
        Initialize centralized distributed inference.
        
        Args:
            coordinator: Optional existing coordinator to wrap
        """
        self.coordinator = coordinator
        self.nodes = {}
        self.pipeline_stages = []
        
    async def run_inference(
        self,
        request: DistributedRequest
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Single entry point for distributed inference.
        
        Args:
            request: Unified distributed request
            
        Returns:
            Tuple of (response_text, routing_info)
        """
        start_time = time.time()
        
        # Generate request ID if not provided
        if not request.request_id:
            request.request_id = f"dist_{int(time.time() * 1000)}"
        
        routing_info = {
            "request_id": request.request_id,
            "start_time": start_time,
            "nodes_used": [],
            "pipeline_stages": {},
            "success": False
        }
        
        try:
            if self.coordinator:
                # Use existing coordinator
                response_text, info = await self.coordinator.distribute_inference(
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    stream=request.stream
                )
                routing_info.update(info)
            else:
                # Direct implementation
                response_text = await self._direct_inference(request)
                routing_info["nodes_used"] = list(self.nodes.keys())
            
            routing_info["success"] = True
            routing_info["latency_ms"] = (time.time() - start_time) * 1000
            
            return response_text, routing_info
            
        except Exception as e:
            logger.error(f"Distributed inference failed: {e}")
            routing_info["error"] = str(e)
            routing_info["latency_ms"] = (time.time() - start_time) * 1000
            raise
    
    async def _direct_inference(self, request: DistributedRequest) -> str:
        """
        Direct distributed inference implementation.
        
        Args:
            request: Distributed request
            
        Returns:
            Generated text
        """
        # Setup pipeline stages
        if not self.pipeline_stages:
            self._setup_pipeline()
        
        # Process through pipeline
        hidden_states = None
        current_output = request.prompt
        
        for stage_idx, stage in enumerate(self.pipeline_stages):
            logger.info(f"Processing stage {stage_idx}: layers {stage['start']}-{stage['end']}")
            
            # Send to appropriate node
            node = self._select_node_for_stage(stage)
            if not node:
                raise RuntimeError(f"No node available for stage {stage_idx}")
            
            # Process on node (simplified)
            current_output = await self._process_on_node(
                node,
                current_output,
                stage,
                request
            )
        
        return current_output
    
    def _setup_pipeline(self) -> None:
        """Setup pipeline stages for dense model."""
        # Dense model with 36 layers
        total_layers = 36
        num_stages = min(3, len(self.nodes))  # Max 3 stages
        
        if num_stages == 0:
            # No distributed nodes, single stage
            self.pipeline_stages = [{
                "stage": 0,
                "start": 0,
                "end": total_layers - 1
            }]
        else:
            # Distribute layers across stages
            layers_per_stage = total_layers // num_stages
            remainder = total_layers % num_stages
            
            start = 0
            for i in range(num_stages):
                end = start + layers_per_stage - 1
                if i < remainder:
                    end += 1
                
                self.pipeline_stages.append({
                    "stage": i,
                    "start": start,
                    "end": end
                })
                
                start = end + 1
    
    def _select_node_for_stage(self, stage: Dict[str, Any]) -> Optional[str]:
        """Select best node for pipeline stage."""
        # Simple selection - use stage index to pick node
        node_list = list(self.nodes.keys())
        if stage["stage"] < len(node_list):
            return node_list[stage["stage"]]
        return None
    
    async def _process_on_node(
        self,
        node_id: str,
        input_data: str,
        stage: Dict[str, Any],
        request: DistributedRequest
    ) -> str:
        """
        Process on specific node.
        
        Args:
            node_id: Node identifier
            input_data: Input for this stage
            stage: Pipeline stage info
            request: Original request
            
        Returns:
            Output from node
        """
        # This would make actual HTTP/gRPC call to node
        # For now, return mock processed output
        logger.info(f"Processing on node {node_id} for stage {stage['stage']}")
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # In reality, this would transform hidden states
        # For final stage, generate tokens
        if stage["stage"] == len(self.pipeline_stages) - 1:
            return f"Response to: {request.prompt[:50]}..."
        
        return input_data
    
    def register_node(self, node_id: str, node_info: Dict[str, Any]) -> None:
        """
        Register a node for distributed inference.
        
        Args:
            node_id: Node identifier
            node_info: Node capabilities and info
        """
        self.nodes[node_id] = node_info
        # Recalculate pipeline stages
        self._setup_pipeline()
        logger.info(f"Registered node {node_id}, recalculated pipeline")
    
    def unregister_node(self, node_id: str) -> None:
        """
        Unregister a node.
        
        Args:
            node_id: Node to remove
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
            self._setup_pipeline()
            logger.info(f"Unregistered node {node_id}")


# Global instance
_global_distributed: Optional[CentralizedDistributedInference] = None


def get_distributed_inference(
    coordinator=None,
    force_new: bool = False
) -> CentralizedDistributedInference:
    """
    Get or create global distributed inference coordinator.
    
    Args:
        coordinator: Optional existing coordinator
        force_new: Force new instance
        
    Returns:
        CentralizedDistributedInference instance
    """
    global _global_distributed
    
    if _global_distributed is None or force_new:
        _global_distributed = CentralizedDistributedInference(coordinator)
    
    return _global_distributed


async def run_distributed_inference(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    stream: bool = False,
    **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function for distributed inference.
    
    Args:
        prompt: Input prompt
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        stream: Whether to stream
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (response, routing_info)
    """
    distributed = get_distributed_inference()
    
    request = DistributedRequest(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stream=stream,
        **kwargs
    )
    
    return await distributed.run_inference(request)