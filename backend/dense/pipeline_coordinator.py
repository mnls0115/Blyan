"""
Distributed Pipeline Coordinator for Dense Models
=================================================
Manages multi-GPU inference with layer-based partitioning.
Supports thinking mode and efficient pipeline parallelism.
"""

import asyncio
import time
import torch
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from backend.model.chunked_blockchain_loader import ChunkedBlockchainModelManager, create_chunked_loader
from backend.dense.partition_planner import PartitionPlan, create_optimal_plan
from backend.dense.thinking_mode import ThinkingModeHandler, ThinkingConfig, ThinkingMode
from config.model_profile import MODEL_ID, LAYERS

logger = logging.getLogger(__name__)


@dataclass
class GPUNode:
    """Represents a GPU node in the distributed pipeline."""
    node_id: str
    host: str
    port: int
    stage_id: int
    chunk_manager: Optional[ChunkedBlockchainModelManager] = None
    is_local: bool = False
    is_healthy: bool = True
    last_heartbeat: float = field(default_factory=time.time)


@dataclass
class PipelineRequest:
    """Request for distributed pipeline inference."""
    request_id: str
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    thinking: bool = False
    thinking_mode: str = "local_then_distribute"
    max_think_tokens: int = 128


@dataclass
class PipelineState:
    """State of a pipeline inference request."""
    request_id: str
    current_stage: int = 0
    hidden_states: Optional[torch.Tensor] = None
    past_key_values: Optional[List] = None
    tokens_generated: int = 0
    start_time: float = field(default_factory=time.time)


class DistributedPipelineCoordinator:
    """Coordinates distributed inference across multiple GPU nodes."""
    
    def __init__(
        self,
        root_dir: Path,
        partition_plan: PartitionPlan,
        thinking_config: Optional[ThinkingConfig] = None
    ):
        """
        Initialize pipeline coordinator.
        
        Args:
            root_dir: Root directory for blockchain data
            partition_plan: Partition plan for model distribution
            thinking_config: Optional thinking mode configuration
        """
        self.root_dir = root_dir
        self.partition_plan = partition_plan
        self.nodes: Dict[str, GPUNode] = {}
        self.stage_to_node: Dict[int, str] = {}
        self.active_requests: Dict[str, PipelineState] = {}
        
        # Initialize thinking mode if configured
        self.thinking_handler = None
        if thinking_config:
            self.thinking_handler = ThinkingModeHandler(thinking_config)
        
        logger.info(
            f"Pipeline coordinator initialized: "
            f"{partition_plan.num_gpus} stages, "
            f"Thinking mode: {thinking_config.enabled if thinking_config else False}"
        )
    
    def register_node(self, node: GPUNode):
        """Register a GPU node for a specific stage."""
        self.nodes[node.node_id] = node
        self.stage_to_node[node.stage_id] = node.node_id
        
        # Load chunk if local node
        if node.is_local:
            try:
                node.chunk_manager = create_chunked_loader(
                    self.root_dir,
                    self.partition_plan,
                    node.stage_id
                )
                logger.info(f"Loaded chunk for local node {node.node_id} (stage {node.stage_id})")
            except Exception as e:
                logger.error(f"Failed to load chunk for node {node.node_id}: {e}")
                node.is_healthy = False
        
        logger.info(f"Registered node {node.node_id} for stage {node.stage_id}")
    
    def unregister_node(self, node_id: str):
        """Unregister a GPU node."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            del self.stage_to_node[node.stage_id]
            del self.nodes[node_id]
            logger.info(f"Unregistered node {node_id}")
    
    def get_node_for_stage(self, stage_id: int) -> Optional[GPUNode]:
        """Get the node responsible for a stage."""
        node_id = self.stage_to_node.get(stage_id)
        return self.nodes.get(node_id) if node_id else None
    
    def check_pipeline_ready(self) -> Tuple[bool, List[int]]:
        """Check if all pipeline stages have nodes assigned."""
        missing_stages = []
        for stage_id in range(self.partition_plan.num_gpus):
            if stage_id not in self.stage_to_node:
                missing_stages.append(stage_id)
        
        return len(missing_stages) == 0, missing_stages
    
    async def process_request(
        self,
        request: PipelineRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process an inference request through the pipeline.
        
        Args:
            request: Pipeline request
            
        Yields:
            Token dictionaries with generated text
        """
        # Check pipeline readiness
        ready, missing = self.check_pipeline_ready()
        if not ready:
            raise RuntimeError(f"Pipeline not ready, missing stages: {missing}")
        
        # Create pipeline state
        state = PipelineState(request_id=request.request_id)
        self.active_requests[request.request_id] = state
        
        try:
            # Determine execution strategy based on thinking mode
            if self.thinking_handler and request.thinking:
                self.thinking_handler.create_session(request.request_id)
                strategy = self.thinking_handler.get_execution_strategy(request.request_id)
            else:
                strategy = {"mode": "distributed", "sticky": False}
            
            # Generate tokens
            async for token_data in self._generate_tokens(request, state, strategy):
                # Filter through thinking handler if enabled
                if self.thinking_handler and request.thinking:
                    async for filtered_token in self.thinking_handler.process_token_stream(
                        request.request_id,
                        self._async_wrapper([token_data]),
                        self.nodes[list(self.nodes.keys())[0]].node_id if self.nodes else None
                    ):
                        yield filtered_token
                else:
                    yield token_data
        
        finally:
            # Cleanup
            del self.active_requests[request.request_id]
            if self.thinking_handler:
                self.thinking_handler.cleanup_session(request.request_id)
    
    async def _async_wrapper(self, items):
        """Wrap list as async generator."""
        for item in items:
            yield item
    
    async def _generate_tokens(
        self,
        request: PipelineRequest,
        state: PipelineState,
        strategy: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate tokens using the distributed pipeline.
        
        Args:
            request: Pipeline request
            state: Pipeline state
            strategy: Execution strategy
            
        Yields:
            Generated tokens
        """
        # Tokenize prompt
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # Apply chat template if using thinking mode
        if request.thinking:
            messages = [{"role": "user", "content": request.prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            input_ids = tokenizer.encode(text, return_tensors="pt")
        else:
            input_ids = tokenizer.encode(request.prompt, return_tensors="pt")
        
        # Initialize hidden states (embedding)
        hidden_states = await self._run_embedding(input_ids)
        state.hidden_states = hidden_states
        
        # Generate tokens
        for token_idx in range(request.max_new_tokens):
            # Check if we should use sticky assignment (for thinking mode)
            if strategy.get("sticky") and strategy.get("node_id"):
                # Run all stages on the sticky node
                hidden_states = await self._run_all_stages_on_node(
                    hidden_states, 
                    strategy["node_id"],
                    state
                )
            else:
                # Run through pipeline stages
                for stage_id in range(self.partition_plan.num_gpus):
                    hidden_states = await self._run_stage(
                        stage_id,
                        hidden_states,
                        state
                    )
            
            # Get logits from final hidden states
            logits = hidden_states
            
            # Sample next token
            next_token_id = self._sample_token(
                logits,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )
            
            # Decode token
            token_text = tokenizer.decode([next_token_id])
            
            # Update state
            state.tokens_generated += 1
            
            # Yield token
            yield {
                "token_id": next_token_id,
                "token": token_text,
                "timestamp": time.time()
            }
            
            # Check for EOS
            if next_token_id == tokenizer.eos_token_id:
                break
            
            # Update hidden states for next iteration
            # (In real implementation, would append to input and get new embeddings)
            hidden_states = hidden_states  # Simplified
    
    async def _run_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run embedding layer."""
        # Get node with embedding (stage 0)
        node = self.get_node_for_stage(0)
        if not node:
            raise RuntimeError("No node available for embedding")
        
        if node.is_local and node.chunk_manager:
            # Local execution
            # Simplified - in reality would use actual embedding
            batch_size, seq_len = input_ids.shape
            hidden_size = LAYERS["hidden_size"]
            return torch.randn(batch_size, seq_len, hidden_size)
        else:
            # Remote execution
            return await self._remote_embedding(node, input_ids)
    
    async def _run_stage(
        self,
        stage_id: int,
        hidden_states: torch.Tensor,
        state: PipelineState
    ) -> torch.Tensor:
        """Run a pipeline stage."""
        node = self.get_node_for_stage(stage_id)
        if not node:
            raise RuntimeError(f"No node available for stage {stage_id}")
        
        if node.is_local and node.chunk_manager:
            # Local execution
            result = node.chunk_manager.forward_chunk(
                hidden_states,
                past_key_values=state.past_key_values
            )
            return result["hidden_states"]
        else:
            # Remote execution
            return await self._remote_forward(node, hidden_states, state)
    
    async def _run_all_stages_on_node(
        self,
        hidden_states: torch.Tensor,
        node_id: str,
        state: PipelineState
    ) -> torch.Tensor:
        """Run all stages on a single node (for sticky assignment)."""
        node = self.nodes.get(node_id)
        if not node:
            raise RuntimeError(f"Node {node_id} not found")
        
        # For simplicity, just return modified hidden states
        # In reality, would execute all layers on that node
        return hidden_states + 0.01 * torch.randn_like(hidden_states)
    
    async def _remote_embedding(
        self,
        node: GPUNode,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Execute embedding on remote node."""
        # Simplified - would use actual RPC
        batch_size, seq_len = input_ids.shape
        hidden_size = LAYERS["hidden_size"]
        return torch.randn(batch_size, seq_len, hidden_size)
    
    async def _remote_forward(
        self,
        node: GPUNode,
        hidden_states: torch.Tensor,
        state: PipelineState
    ) -> torch.Tensor:
        """Execute forward pass on remote node."""
        # Simplified - would use actual RPC
        # Track inter-node traffic
        if self.thinking_handler:
            bytes_sent = hidden_states.numel() * hidden_states.element_size()
            self.thinking_handler.update_inter_node_traffic(bytes_sent)
        
        return hidden_states + 0.01 * torch.randn_like(hidden_states)
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> int:
        """Sample next token from logits."""
        # Simplified sampling
        # In reality, would use proper sampling with temperature, top-p, top-k
        vocab_size = LAYERS["vocab_size"]
        probs = torch.randn(vocab_size).softmax(dim=-1)
        return torch.multinomial(probs, 1).item()
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        ready, missing = self.check_pipeline_ready()
        
        node_status = {}
        for node_id, node in self.nodes.items():
            node_status[node_id] = {
                "stage_id": node.stage_id,
                "is_healthy": node.is_healthy,
                "is_local": node.is_local,
                "last_heartbeat": time.time() - node.last_heartbeat
            }
        
        status = {
            "ready": ready,
            "missing_stages": missing,
            "num_stages": self.partition_plan.num_gpus,
            "nodes": node_status,
            "active_requests": len(self.active_requests),
            "partition_plan": {
                "num_gpus": self.partition_plan.num_gpus,
                "precision": self.partition_plan.precision,
                "target_vram_gb": self.partition_plan.target_vram_gb
            }
        }
        
        # Add thinking mode metrics if available
        if self.thinking_handler:
            status["thinking_metrics"] = self.thinking_handler.get_metrics()
        
        return status
    
    async def health_check(self):
        """Perform health check on all nodes."""
        for node_id, node in self.nodes.items():
            if node.is_local:
                # Check local node health
                if node.chunk_manager:
                    memory = node.chunk_manager.get_memory_usage()
                    node.is_healthy = memory["allocated_gb"] < node.chunk_manager.partition_plan.target_vram_gb
            else:
                # Check remote node health (simplified)
                node.is_healthy = (time.time() - node.last_heartbeat) < 30
        
        return self.get_pipeline_status()


# Demo usage
async def demo_pipeline():
    """Demonstrate distributed pipeline functionality."""
    from pathlib import Path
    
    # Create partition plan for 4GB GPUs
    plan = create_optimal_plan(target_vram_gb=4.0, precision="int4")
    
    # Create thinking config
    thinking_config = ThinkingConfig(
        enabled=True,
        mode=ThinkingMode.LOCAL_THEN_DISTRIBUTE,
        max_think_tokens=128
    )
    
    # Create coordinator
    coordinator = DistributedPipelineCoordinator(
        root_dir=Path("./data"),
        partition_plan=plan,
        thinking_config=thinking_config
    )
    
    # Register local nodes for demo
    for stage_id in range(min(3, plan.num_gpus)):  # Register first 3 stages
        node = GPUNode(
            node_id=f"node_{stage_id}",
            host="localhost",
            port=8000 + stage_id,
            stage_id=stage_id,
            is_local=True
        )
        coordinator.register_node(node)
    
    # Check status
    status = coordinator.get_pipeline_status()
    print(f"Pipeline status: {status}")
    
    # Create request
    request = PipelineRequest(
        request_id="test_001",
        prompt="Explain quantum computing",
        max_new_tokens=50,
        thinking=True,
        thinking_mode="local_then_distribute"
    )
    
    # Process request (would generate tokens in real implementation)
    print("\nProcessing request with thinking mode...")
    try:
        token_count = 0
        async for token in coordinator.process_request(request):
            print(token["token"], end="", flush=True)
            token_count += 1
            if token_count >= 10:  # Limit for demo
                break
    except RuntimeError as e:
        print(f"\nError: {e}")
    
    print(f"\n\nFinal status: {coordinator.get_pipeline_status()}")


if __name__ == "__main__":
    asyncio.run(demo_pipeline())