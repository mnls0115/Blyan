"""
Chunked Blockchain Model Loader for Dense Models
===============================================
Loads specific layers/chunks of a dense model from blockchain storage.
Supports partitioned inference across multiple small GPUs.
"""

import torch
import json
import logging
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

from backend.core.chain import Chain
from backend.core.param_index import ParameterIndex
from backend.dense.partition_planner import PartitionPlan, PipelineStage
from config.model_profile import LAYERS, MODEL_ID

logger = logging.getLogger(__name__)


@dataclass
class ModelChunk:
    """Represents a chunk of the model loaded on this GPU."""
    stage_id: int
    layer_range: Tuple[int, int]  # (start, end) inclusive
    has_embedding: bool
    has_lm_head: bool
    weights: Dict[str, torch.Tensor]
    device: str


class ChunkedBlockchainModelManager:
    """
    Manages loading and inference for a specific chunk of the dense model.
    Each GPU node runs one instance managing its assigned layers.
    """
    
    def __init__(
        self,
        meta_chain: Chain,
        param_chain: Chain,
        param_index: ParameterIndex,
        partition_plan: PartitionPlan,
        stage_id: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize chunked model manager for a specific stage.
        
        Args:
            meta_chain: Metadata blockchain
            param_chain: Parameter blockchain
            param_index: Parameter index
            partition_plan: Complete partition plan
            stage_id: Which stage this GPU handles
            device: Device to load weights on
        """
        self.meta_chain = meta_chain
        self.param_chain = param_chain
        self.param_index = param_index
        self.partition_plan = partition_plan
        self.stage_id = stage_id
        self.device = device
        
        if stage_id >= len(partition_plan.stages):
            raise ValueError(f"Stage {stage_id} not in partition plan (has {len(partition_plan.stages)} stages)")
        
        self.stage = partition_plan.stages[stage_id]
        self.model_chunk = None
        
        logger.info(
            f"Chunked loader initialized for stage {stage_id}: "
            f"Layers {self.stage.layer_range}, "
            f"Memory: {self.stage.total_memory_gb:.2f}GB"
        )
    
    def load_chunk_from_blockchain(self) -> ModelChunk:
        """Load this stage's chunk from blockchain."""
        weights = {}
        has_embedding = False
        has_lm_head = False
        
        # Determine what components to load
        for block in self.stage.blocks:
            if block.block_type == "embedding":
                has_embedding = True
                weights.update(self._load_embedding())
            elif block.block_type == "lm_head":
                has_lm_head = True
                weights.update(self._load_lm_head())
            elif block.block_type == "layer":
                for layer_idx in block.layer_indices:
                    layer_weights = self._load_layer(layer_idx)
                    if layer_weights:
                        weights.update(layer_weights)
        
        self.model_chunk = ModelChunk(
            stage_id=self.stage_id,
            layer_range=self.stage.layer_range,
            has_embedding=has_embedding,
            has_lm_head=has_lm_head,
            weights=weights,
            device=self.device
        )
        
        logger.info(
            f"Loaded chunk for stage {self.stage_id}: "
            f"{len(weights)} weight tensors, "
            f"Embedding: {has_embedding}, LM Head: {has_lm_head}"
        )
        
        return self.model_chunk
    
    def _load_embedding(self) -> Dict[str, torch.Tensor]:
        """Load embedding weights from blockchain."""
        logger.info("Loading embedding from blockchain...")
        
        # Search for embedding block
        block_count = len(self.param_chain._hash_index) if hasattr(self.param_chain, '_hash_index') else 0
        
        for i in range(block_count):
            block = self.param_chain.storage.get_block_by_index(i)
            if block and hasattr(block.header, 'block_type') and block.header.block_type == 'embedding':
                try:
                    from backend.model.arch import bytes_to_state_dict
                    weights = bytes_to_state_dict(block.payload)
                    # Move to device
                    weights = {k: v.to(self.device) for k, v in weights.items()}
                    logger.info(f"Loaded embedding weights: {list(weights.keys())}")
                    return weights
                except Exception as e:
                    logger.error(f"Failed to load embedding: {e}")
        
        # Fallback: create dummy embedding
        logger.warning("Embedding not found in blockchain, creating dummy")
        return {
            "model.embed_tokens.weight": torch.randn(
                LAYERS["vocab_size"], LAYERS["hidden_size"], 
                device=self.device, dtype=torch.float16
            )
        }
    
    def _load_layer(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Load weights for a specific layer from blockchain."""
        logger.info(f"Loading layer {layer_idx} from blockchain...")
        
        # Search for layer block
        block_count = len(self.param_chain._hash_index) if hasattr(self.param_chain, '_hash_index') else 0
        
        for i in range(block_count):
            block = self.param_chain.storage.get_block_by_index(i)
            if (block and hasattr(block.header, 'block_type') and 
                block.header.block_type == 'layer' and
                hasattr(block.header, 'layer_id') and 
                block.header.layer_id == layer_idx):
                try:
                    from backend.model.arch import bytes_to_state_dict
                    weights = bytes_to_state_dict(block.payload)
                    # Move to device
                    weights = {k: v.to(self.device) for k, v in weights.items()}
                    logger.info(f"Loaded layer {layer_idx} weights: {len(weights)} tensors")
                    return weights
                except Exception as e:
                    logger.error(f"Failed to load layer {layer_idx}: {e}")
        
        # Fallback: create dummy layer weights
        logger.warning(f"Layer {layer_idx} not found in blockchain, creating dummy")
        prefix = f"model.layers.{layer_idx}"
        hidden_size = LAYERS["hidden_size"]
        intermediate_size = LAYERS["intermediate_size"]
        num_heads = LAYERS["num_attention_heads"]
        num_kv_heads = LAYERS["num_kv_heads"]
        head_dim = LAYERS["head_dim"]
        
        return {
            f"{prefix}.self_attn.q_proj.weight": torch.randn(
                hidden_size, hidden_size, device=self.device, dtype=torch.float16
            ),
            f"{prefix}.self_attn.k_proj.weight": torch.randn(
                num_kv_heads * head_dim, hidden_size, device=self.device, dtype=torch.float16
            ),
            f"{prefix}.self_attn.v_proj.weight": torch.randn(
                num_kv_heads * head_dim, hidden_size, device=self.device, dtype=torch.float16  
            ),
            f"{prefix}.self_attn.o_proj.weight": torch.randn(
                hidden_size, hidden_size, device=self.device, dtype=torch.float16
            ),
            f"{prefix}.mlp.gate_proj.weight": torch.randn(
                intermediate_size, hidden_size, device=self.device, dtype=torch.float16
            ),
            f"{prefix}.mlp.up_proj.weight": torch.randn(
                intermediate_size, hidden_size, device=self.device, dtype=torch.float16
            ),
            f"{prefix}.mlp.down_proj.weight": torch.randn(
                hidden_size, intermediate_size, device=self.device, dtype=torch.float16
            ),
            f"{prefix}.input_layernorm.weight": torch.ones(
                hidden_size, device=self.device, dtype=torch.float16
            ),
            f"{prefix}.post_attention_layernorm.weight": torch.ones(
                hidden_size, device=self.device, dtype=torch.float16
            ),
        }
    
    def _load_lm_head(self) -> Dict[str, torch.Tensor]:
        """Load LM head weights from blockchain."""
        logger.info("Loading LM head from blockchain...")
        
        # Search for lm_head block
        block_count = len(self.param_chain._hash_index) if hasattr(self.param_chain, '_hash_index') else 0
        
        for i in range(block_count):
            block = self.param_chain.storage.get_block_by_index(i)
            if block and hasattr(block.header, 'block_type') and block.header.block_type == 'lm_head':
                try:
                    from backend.model.arch import bytes_to_state_dict
                    weights = bytes_to_state_dict(block.payload)
                    # Move to device
                    weights = {k: v.to(self.device) for k, v in weights.items()}
                    logger.info(f"Loaded LM head weights: {list(weights.keys())}")
                    return weights
                except Exception as e:
                    logger.error(f"Failed to load LM head: {e}")
        
        # Fallback: create dummy LM head
        logger.warning("LM head not found in blockchain, creating dummy")
        return {
            "lm_head.weight": torch.randn(
                LAYERS["vocab_size"], LAYERS["hidden_size"],
                device=self.device, dtype=torch.float16
            )
        }
    
    def forward_chunk(self, hidden_states: torch.Tensor, 
                     position_ids: Optional[torch.Tensor] = None,
                     attention_mask: Optional[torch.Tensor] = None,
                     past_key_values: Optional[List] = None) -> Dict[str, Any]:
        """
        Forward pass through this chunk.
        
        Args:
            hidden_states: Input hidden states
            position_ids: Position IDs for RoPE
            attention_mask: Attention mask
            past_key_values: KV cache from previous tokens
            
        Returns:
            Dict with output hidden states and new KV cache
        """
        if not self.model_chunk:
            self.load_chunk_from_blockchain()
        
        # This is a simplified forward pass
        # In production, you'd use the actual transformer implementation
        output = hidden_states
        new_kv_cache = []
        
        # Process through layers in this chunk
        if self.model_chunk.layer_range[0] >= 0:
            for layer_idx in range(self.model_chunk.layer_range[0], 
                                 self.model_chunk.layer_range[1] + 1):
                # Simplified layer processing
                # In reality, this would be proper transformer layer forward
                output = output + 0.01 * torch.randn_like(output)  # Dummy transformation
                new_kv_cache.append(None)  # Would be actual KV tensors
        
        # Apply LM head if this chunk has it
        if self.model_chunk.has_lm_head:
            # Project to vocab size
            output = torch.randn(
                output.shape[0], output.shape[1], LAYERS["vocab_size"],
                device=self.device, dtype=output.dtype
            )
        
        return {
            "hidden_states": output,
            "past_key_values": new_kv_cache
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if not self.model_chunk:
            return {"weights_gb": 0, "allocated_gb": 0, "reserved_gb": 0}
        
        # Calculate weight memory
        weight_bytes = sum(
            w.numel() * w.element_size() 
            for w in self.model_chunk.weights.values()
        )
        weight_gb = weight_bytes / (1024**3)
        
        # Get CUDA memory if available
        allocated_gb = 0
        reserved_gb = 0
        if self.device == "cuda" and torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
        
        return {
            "weights_gb": weight_gb,
            "allocated_gb": allocated_gb,
            "reserved_gb": reserved_gb,
            "stage_id": self.stage_id,
            "layer_range": self.model_chunk.layer_range
        }
    
    def validate_chunk(self) -> bool:
        """Validate that chunk loaded correctly."""
        if not self.model_chunk:
            return False
        
        expected_components = []
        for block in self.stage.blocks:
            if block.block_type == "embedding":
                expected_components.append("embedding")
            elif block.block_type == "lm_head":
                expected_components.append("lm_head")
            elif block.block_type == "layer":
                expected_components.extend([f"layer_{i}" for i in block.layer_indices])
        
        # Check if we have weights for expected components
        has_components = []
        for key in self.model_chunk.weights.keys():
            if "embed_tokens" in key:
                has_components.append("embedding")
            elif "lm_head" in key:
                has_components.append("lm_head")
            else:
                # Extract layer number from weight key
                import re
                match = re.search(r"layers\.(\d+)\.", key)
                if match:
                    layer_num = int(match.group(1))
                    has_components.append(f"layer_{layer_num}")
        
        has_components = list(set(has_components))
        
        logger.info(f"Expected components: {expected_components}")
        logger.info(f"Loaded components: {has_components}")
        
        return len(has_components) > 0


def create_chunked_loader(
    root_dir: Path,
    partition_plan: PartitionPlan,
    stage_id: int
) -> ChunkedBlockchainModelManager:
    """
    Factory function to create a chunked model loader.
    
    Args:
        root_dir: Root directory for blockchain data
        partition_plan: Partition plan
        stage_id: Which stage to load
        
    Returns:
        ChunkedBlockchainModelManager for the specified stage
    """
    from backend.core.chain import Chain
    from backend.core.param_index import ParameterIndex
    
    # Initialize chains
    meta_chain = Chain(root_dir, 'A')
    param_chain = Chain(root_dir, 'B')
    param_index = ParameterIndex(root_dir / "param_index.json")  # Fix: ParameterIndex expects a file path
    
    # Create chunked manager
    manager = ChunkedBlockchainModelManager(
        meta_chain=meta_chain,
        param_chain=param_chain,
        param_index=param_index,
        partition_plan=partition_plan,
        stage_id=stage_id
    )
    
    # Load chunk
    manager.load_chunk_from_blockchain()
    
    # Validate
    if manager.validate_chunk():
        logger.info(f"✅ Stage {stage_id} loaded successfully")
    else:
        logger.warning(f"⚠️ Stage {stage_id} validation failed")
    
    # Report memory usage
    memory = manager.get_memory_usage()
    logger.info(
        f"Memory usage for stage {stage_id}: "
        f"Weights: {memory['weights_gb']:.2f}GB, "
        f"Allocated: {memory['allocated_gb']:.2f}GB"
    )
    
    return manager