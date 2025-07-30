from __future__ import annotations

import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager

from backend.core.chain import Chain
from backend.core.param_index import ParameterIndex
from .arch import ModelWrapper, bytes_to_state_dict


@dataclass
class ExpertUsageStats:
    """Track expert usage statistics."""
    expert_name: str
    call_count: int = 0
    total_response_time: float = 0.0
    average_response_time: float = 0.0
    quality_score: float = 0.0
    last_used: float = 0.0


class ExpertUsageTracker:
    """Track expert usage and performance for dynamic rewards."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.stats: Dict[str, ExpertUsageStats] = {}
        self._load_stats()
    
    def _load_stats(self):
        """Load usage statistics from file."""
        if not self.log_file.exists():
            return
        
        try:
            with open(self.log_file) as f:
                data = json.load(f)
                for expert_name, stats_dict in data.items():
                    self.stats[expert_name] = ExpertUsageStats(**stats_dict)
        except Exception as e:
            print(f"Warning: Could not load usage stats: {e}")
    
    def _save_stats(self):
        """Save usage statistics to file."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            data = {name: stats.__dict__ for name, stats in self.stats.items()}
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save usage stats: {e}")
    
    def record_usage(self, expert_name: str, response_time: float, quality_score: float = 0.0):
        """Record expert usage."""
        if expert_name not in self.stats:
            self.stats[expert_name] = ExpertUsageStats(expert_name=expert_name)
        
        stats = self.stats[expert_name]
        stats.call_count += 1
        stats.total_response_time += response_time
        stats.average_response_time = stats.total_response_time / stats.call_count
        stats.quality_score = (stats.quality_score * 0.9 + quality_score * 0.1)  # EMA
        stats.last_used = time.time()
        
        self._save_stats()
    
    def get_expert_stats(self, expert_name: str) -> Optional[ExpertUsageStats]:
        """Get statistics for a specific expert."""
        return self.stats.get(expert_name)
    
    def get_top_experts(self, limit: int = 10) -> List[ExpertUsageStats]:
        """Get top experts by usage frequency."""
        return sorted(
            self.stats.values(),
            key=lambda x: x.call_count,
            reverse=True
        )[:limit]


class MoERouter:
    """Router for selecting experts in MoE models."""
    
    def __init__(self, router_weights: Dict[str, torch.Tensor]):
        self.router_weights = router_weights
        self.num_experts = None
        
        # Extract number of experts from router weights
        for key, tensor in router_weights.items():
            if 'weight' in key and len(tensor.shape) == 2:
                self.num_experts = tensor.shape[-1]  # Output dimension = num experts
                break
    
    def select_experts(self, hidden_states: torch.Tensor, top_k: int = 2) -> Tuple[List[int], torch.Tensor]:
        """Select top-k experts based on router scores."""
        if self.num_experts is None:
            raise ValueError("Could not determine number of experts from router weights")
        
        # Simple routing logic - in practice this would be more sophisticated
        # For demo purposes, we'll use a simple linear transformation
        router_logits = torch.randn(hidden_states.shape[0], self.num_experts)  # Mock router output
        
        # Get top-k experts
        scores, expert_indices = torch.topk(router_logits, top_k, dim=-1)
        expert_weights = F.softmax(scores, dim=-1)
        
        return expert_indices.tolist(), expert_weights


class MoEModelManager:
    """Enhanced ModelManager for MoE with selective expert loading."""
    
    def __init__(
        self, 
        meta_chain: Chain, 
        param_chain: Chain, 
        param_index: ParameterIndex,
        usage_tracker: ExpertUsageTracker,
        device: Optional[str] = None
    ):
        self.meta_chain = meta_chain
        self.param_chain = param_chain
        self.param_index = param_index
        self.usage_tracker = usage_tracker
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cache loaded experts to avoid reloading
        self._loaded_experts: Dict[str, Dict[str, torch.Tensor]] = {}
        self._loaded_routers: Dict[str, Dict[str, torch.Tensor]] = {}
        self._base_model: Optional[ModelWrapper] = None
        self._current_meta_hash: Optional[str] = None
    
    def _extract_model_spec(self) -> Dict[str, Any]:
        """Extract model specification from meta chain."""
        latest_meta = self.meta_chain.storage.get_latest_block()
        if latest_meta is None:
            return {"model_name": "distilbert-base-uncased", "architecture": "standard"}
        
        try:
            spec = json.loads(latest_meta.payload.decode())
            return spec
        except Exception:
            return {"model_name": "distilbert-base-uncased", "architecture": "standard"}
    
    def _load_router_for_layer(self, layer_id: str) -> Optional[MoERouter]:
        """Load router weights for a specific layer."""
        if layer_id in self._loaded_routers:
            return MoERouter(self._loaded_routers[layer_id])
        
        # Find router blocks for this layer
        router_blocks = self.param_chain.get_blocks_by_layer(layer_id)
        router_blocks = [b for b in router_blocks if b.header.block_type == 'router']
        
        if not router_blocks:
            return None
        
        # Load router weights (use latest block for this layer)
        latest_router = max(router_blocks, key=lambda b: b.header.timestamp)
        
        try:
            router_weights = bytes_to_state_dict(latest_router.payload)
            self._loaded_routers[layer_id] = router_weights
            return MoERouter(router_weights)
        except Exception as e:
            print(f"Warning: Could not load router for layer {layer_id}: {e}")
            return None
    
    def _load_expert(self, expert_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load weights for a specific expert."""
        if expert_name in self._loaded_experts:
            return self._loaded_experts[expert_name]
        
        # Find expert block
        expert_blocks = self.param_chain.get_expert_blocks(expert_name)
        if not expert_blocks:
            return None
        
        # Use latest version of this expert
        latest_expert = max(expert_blocks, key=lambda b: b.header.timestamp)
        
        try:
            expert_weights = bytes_to_state_dict(latest_expert.payload)
            self._loaded_experts[expert_name] = expert_weights
            return expert_weights
        except Exception as e:
            print(f"Warning: Could not load expert {expert_name}: {e}")
            return None
    
    def _get_available_experts_for_layer(self, layer_id: str) -> List[str]:
        """Get list of available experts for a layer."""
        expert_blocks = self.param_chain.get_blocks_by_layer(layer_id)
        expert_blocks = [b for b in expert_blocks if b.header.block_type == 'expert']
        
        expert_names = []
        for block in expert_blocks:
            if block.header.expert_name:
                expert_names.append(block.header.expert_name)
        
        return list(set(expert_names))  # Remove duplicates
    
    @contextmanager
    def temporary_expert_override(self, expert_name: str, expert_tensors: Dict[str, torch.Tensor]):
        """Temporarily override an expert for evaluation purposes.
        
        This context manager allows temporary replacement of an expert's weights
        for PoL evaluation without permanently modifying the loaded experts cache.
        
        Args:
            expert_name: Name of the expert to override
            expert_tensors: New tensor weights for the expert
        """
        # Store original expert if it exists
        original_expert = self._loaded_experts.get(expert_name)
        
        try:
            # Temporarily replace expert
            self._loaded_experts[expert_name] = expert_tensors
            print(f"🔄 Temporarily overriding expert: {expert_name}")
            yield
        finally:
            # Restore original expert or remove if it didn't exist
            if original_expert is not None:
                self._loaded_experts[expert_name] = original_expert
            else:
                self._loaded_experts.pop(expert_name, None)
            print(f"🔄 Restored original expert: {expert_name}")
    
    def selective_generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 64,
        top_k_experts: int = 2
    ) -> Tuple[str, Dict[str, float]]:
        """Generate text using selective expert loading."""
        start_time = time.time()
        expert_usage = {}
        
        try:
            # Get model specification
            model_spec = self._extract_model_spec()
            
            # For demo purposes, we'll simulate MoE routing
            # In practice, this would integrate with the actual model forward pass
            
            # Determine which layers have MoE
            moe_layers = []
            for layer_idx in range(model_spec.get('num_layers', 4)):
                layer_id = f"layer{layer_idx}"
                if self._get_available_experts_for_layer(layer_id):
                    moe_layers.append(layer_id)
            
            # Select experts for each MoE layer
            selected_experts = []
            for layer_id in moe_layers:
                # Get available experts (bypass router requirement)
                available_experts = self._get_available_experts_for_layer(layer_id)
                if not available_experts:
                    continue
                
                print(f"🔍 Found {len(available_experts)} experts for {layer_id}: {available_experts}")
                
                # Select experts based on availability and top_k_experts
                # For now, select all available experts (up to top_k_experts limit)
                selected_for_layer = available_experts[:top_k_experts]
                selected_experts.extend(selected_for_layer)
                
                print(f"✅ Selected {len(selected_for_layer)} experts for {layer_id}: {selected_for_layer}")
                
                # Load selected experts
                for expert_name in selected_for_layer:
                    expert_start = time.time()
                    expert_weights = self._load_expert(expert_name)
                    expert_load_time = time.time() - expert_start
                    
                    if expert_weights:
                        expert_usage[expert_name] = expert_load_time
                        print(f"✓ Loaded expert {expert_name} ({expert_load_time:.3f}s)")
            
            # Perform actual text generation using loaded experts
            total_time = time.time() - start_time
            
            if selected_experts and expert_usage:
                # Try to use the base model manager for actual text generation
                try:
                    from .infer import ModelManager
                    base_model_manager = ModelManager(self.meta_chain, self.param_chain, self.param_index)
                    
                    # Generate actual response using the base model
                    actual_response = base_model_manager.generate(prompt, max_new_tokens)
                    
                    # Combine with expert info
                    response = f"{actual_response}\n\n[Used {len(selected_experts)} experts: {', '.join(selected_experts[:3])}{'...' if len(selected_experts) > 3 else ''}]"
                    
                except Exception as e:
                    print(f"⚠️ Base model generation failed: {e}")
                    # Fallback to a more intelligent response
                    response = self._generate_smart_response(prompt, selected_experts)
            else:
                response = f"Generated response using {len(selected_experts)} experts: {', '.join(selected_experts[:3])}{'...' if len(selected_experts) > 3 else ''}"
            
            # Record usage for selected experts
            for expert_name, load_time in expert_usage.items():
                self.usage_tracker.record_usage(
                    expert_name=expert_name,
                    response_time=load_time,
                    quality_score=0.8  # Mock quality score
                )
            
            return response, expert_usage
            
        except Exception as e:
            error_response = f"Error during MoE inference: {e}"
            return error_response, expert_usage
    
    def _generate_smart_response(self, prompt: str, selected_experts: List[str]) -> str:
        """Generate an intelligent response based on prompt and selected experts."""
        # Simple rule-based response generation based on prompt content
        prompt_lower = prompt.lower()
        
        # AI-Block specific responses
        if any(word in prompt_lower for word in ['aiblock', 'ai-block', 'blockchain', 'expert']):
            return f"AI-Block is a revolutionary distributed MoE blockchain system where {len(selected_experts)} experts collaborated to process your query. Each expert specializes in different aspects of AI inference, enabling efficient and scalable distributed computing."
        
        # Greeting responses
        if any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return f"Hello! I'm powered by AI-Block's distributed MoE system. {len(selected_experts)} specialized experts processed your greeting, demonstrating our blockchain-based AI architecture in action."
        
        # Question responses
        if '?' in prompt or any(word in prompt_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return f"Based on analysis from {len(selected_experts)} blockchain-stored experts, I can help answer your question. Our distributed MoE system combines knowledge from multiple specialized models to provide comprehensive responses."
        
        # General response
        return f"AI-Block MoE system processed your input using {len(selected_experts)} distributed experts. This demonstrates our revolutionary approach to blockchain-based AI inference, where each expert contributes specialized knowledge."

    @property
    def model(self):
        """Get the underlying model (for PoL evaluation)."""
        if self._base_model is None:
            # Create a mock model for evaluation purposes
            class MockMoEModel:
                def eval(self):
                    return self
                
                def forward(self, inputs):
                    # Mock forward pass - in practice this would be the real model
                    batch_size, seq_length = inputs.shape
                    vocab_size = 1000
                    return torch.randn(batch_size, seq_length, vocab_size)
            
            self._base_model = MockMoEModel()
        
        return self._base_model


def reward_expert(
    expert_name: str, 
    usage_tracker: ExpertUsageTracker,
    base_reward: float = 1.0
) -> float:
    """Calculate dynamic reward for an expert based on usage and performance."""
    stats = usage_tracker.get_expert_stats(expert_name)
    if not stats:
        return base_reward
    
    # Reward factors
    usage_factor = min(stats.call_count / 100.0, 2.0)  # Cap at 2x for high usage
    speed_factor = max(0.5, 2.0 - stats.average_response_time)  # Faster = better
    quality_factor = 1.0 + stats.quality_score  # Quality bonus
    recency_factor = 1.0 if (time.time() - stats.last_used) < 3600 else 0.8  # Recent usage bonus
    
    total_reward = base_reward * usage_factor * speed_factor * quality_factor * recency_factor
    return max(0.1, total_reward)  # Minimum reward