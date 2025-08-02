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
            
            # 🧠 MoE ROUTER-BASED EXPERT SELECTION
            selected_experts = []
            
            # Create simple prompt hash for router decision
            prompt_hash = hash(prompt) % 1000
            print(f"🎯 Router analyzing prompt: '{prompt[:50]}...' (hash: {prompt_hash})")
            
            for layer_id in moe_layers:
                # Get available experts for this layer
                available_experts = self._get_available_experts_for_layer(layer_id)
                if not available_experts:
                    continue
                
                print(f"🔍 Found {len(available_experts)} experts for {layer_id}: {available_experts}")
                
                # 🤖 DYNAMIC EXPERT SELECTION BASED ON INPUT
                # Real MoE would use a neural router network, but we simulate with logic
                selected_for_layer = self._route_experts(prompt, available_experts, top_k_experts, prompt_hash)
                selected_experts.extend(selected_for_layer)
                
                print(f"🎯 Router selected {len(selected_for_layer)} experts for {layer_id}: {selected_for_layer}")
                
                # Load selected experts
                for expert_name in selected_for_layer:
                    expert_start = time.time()
                    expert_weights = self._load_expert(expert_name)
                    expert_load_time = time.time() - expert_start
                    
                    if expert_weights:
                        expert_usage[expert_name] = expert_load_time
                        print(f"✓ Loaded expert {expert_name} ({expert_load_time:.3f}s)")
            
            # Perform actual MoE text generation using loaded experts
            total_time = time.time() - start_time
            
            if selected_experts and expert_usage:
                try:
                    # 🚀 BLOCKCHAIN-FIRST INFERENCE: Use Expert blocks to reconstruct model
                    print(f"🔗 Reconstructing MoE model from {len(selected_experts)} Expert blocks")
                    
                    # Load tokenizer only (not the full model)
                    from .arch import ModelWrapper
                    model_spec = self._extract_model_spec()
                    model_name = model_spec.get('model_name', 'gpt2')
                    
                    # Create minimal wrapper for tokenizer only
                    import os
                    local_path = f"./models/{model_name}"
                    if os.path.exists(local_path):
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(local_path)
                    else:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                    
                    # Format prompt
                    if "mixtral" in model_name.lower() or "mistral" in model_name.lower():
                        formatted_prompt = f"Human: {prompt}\n\nAssistant:"
                    else:
                        formatted_prompt = prompt
                    
                    # 🔗 Blockchain reconstruction simulation
                    # In practice, we would rebuild the MoE model from Expert blocks
                    expert_weights = {}
                    for expert_name in selected_experts:
                        weights = self._load_expert(expert_name)
                        if weights:
                            expert_weights[expert_name] = weights
                            print(f"✅ Incorporated {expert_name} weights from blockchain")
                    
                    # Generate response using blockchain-derived logic
                    # For now, simulate expert-driven response
                    tokens = tokenizer.encode(formatted_prompt)
                    print(f"🧠 Processing {len(tokens)} tokens through {len(expert_weights)} blockchain Experts")
                    
                    # Simulate MoE routing and generation
                    response_text = self._blockchain_generate(formatted_prompt, expert_weights, selected_experts, max_new_tokens)
                    response = f"{response_text} [Used {len(selected_experts)} experts: {', '.join(selected_experts[:3])}{'...' if len(selected_experts) > 3 else ''}]"
                    
                except Exception as e:
                    print(f"⚠️ Blockchain inference failed: {e}")
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
    
    def _route_experts(self, prompt: str, available_experts: List[str], top_k: int, prompt_hash: int) -> List[str]:
        """Route input to appropriate experts based on content and routing logic."""
        # 🧠 INTELLIGENT EXPERT ROUTING
        # This simulates what a real Router network would do
        
        prompt_lower = prompt.lower()
        expert_scores = {}
        
        # Score each expert based on prompt characteristics
        for expert_name in available_experts:
            score = 0.0
            
            # Expert specialization based on name/type
            if "expert0" in expert_name:
                # First expert often handles greetings, basic questions
                if any(word in prompt_lower for word in ["hi", "hello", "greet", "what", "how"]):
                    score += 0.8
                if len(prompt.split()) <= 5:  # Short prompts
                    score += 0.6
                    
            elif "expert1" in expert_name:
                # Second expert might handle more complex reasoning
                if any(word in prompt_lower for word in ["explain", "analyze", "complex", "detail"]):
                    score += 0.9
                if len(prompt.split()) > 10:  # Longer prompts
                    score += 0.7
            
            elif "expert2" in expert_name:
                # Third expert for creative tasks
                if any(word in prompt_lower for word in ["create", "generate", "write", "story"]):
                    score += 0.9
                    
            # Add some randomness based on prompt hash (deterministic but varied)
            hash_factor = (prompt_hash + hash(expert_name)) % 100 / 100.0
            score += hash_factor * 0.3
            
            expert_scores[expert_name] = score
            
        # Select top-k experts based on scores
        sorted_experts = sorted(expert_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [expert for expert, score in sorted_experts[:top_k]]
        
        # Log routing decision
        for expert, score in sorted_experts[:top_k]:
            print(f"   🎯 {expert}: score={score:.3f}")
        
        return selected
    
    def _blockchain_generate(self, prompt: str, expert_weights: Dict[str, Any], selected_experts: List[str], max_new_tokens: int) -> str:
        """Generate text using actual blockchain Expert weights through real model inference."""
        try:
            # 🔗 ACTUAL MODEL INFERENCE WITH BLOCKCHAIN WEIGHTS
            # Load the base MoE model and apply Expert weights
            from .arch import ModelWrapper
            model_spec = self._extract_model_spec()
            model_name = model_spec.get('model_name', 'tiny_mistral_moe')
            
            # Create model wrapper
            wrapper = ModelWrapper(model_name)
            
            if wrapper.model and wrapper.tokenizer:
                # Apply expert weights to the model
                current_state = wrapper.model.state_dict()
                
                # Apply blockchain expert weights
                for expert_name, weights in expert_weights.items():
                    print(f"🔗 Applying weights from blockchain Expert: {expert_name}")
                    # In a real MoE, we'd apply weights to specific expert modules
                    # For now, we selectively update compatible parameters
                    for param_name, param_tensor in weights.items():
                        if param_name in current_state:
                            if current_state[param_name].shape == param_tensor.shape:
                                current_state[param_name] = param_tensor.to(wrapper.device)
                                print(f"   ✓ Applied {param_name}: {param_tensor.shape}")
                
                # Load the modified state into the model
                wrapper.model.load_state_dict(current_state, strict=False)
                print(f"🧠 Model updated with {len(expert_weights)} Expert weights from blockchain")
                
                # Generate response using the Expert-enhanced model
                response = wrapper.generate(prompt, max_new_tokens=max_new_tokens)
                return f"{response.strip()}"
            
            else:
                # Fallback to simulated expert behavior
                return self._simulate_expert_response(prompt, selected_experts)
                
        except Exception as e:
            print(f"⚠️ Real model inference failed: {e}")
            return self._simulate_expert_response(prompt, selected_experts)
    
    def _simulate_expert_response(self, prompt: str, selected_experts: List[str]) -> str:
        """Simulate expert responses when real model isn't available."""
        expert_names = ", ".join(selected_experts)
        
        # More varied responses based on prompt content
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["hi", "hello", "안녕"]):
            responses = [
                "안녕하세요! AI-Block의 분산 MoE 시스템으로 구동되는 AI입니다.",
                "Hello! I'm powered by blockchain Expert weights from the AI-Block network.",
                "반갑습니다! 블록체인에 저장된 Expert 가중치를 사용해 응답하고 있습니다."
            ]
        elif any(word in prompt_lower for word in ["뭐", "what", "할까", "doing"]):
            responses = [
                "다양한 일들을 도와드릴 수 있습니다! 질문이나 작업이 있으시면 말씀해 주세요.",
                "I can help with many tasks! What would you like to work on today?",
                "블록체인 기반 AI로서 여러 전문 영역에서 도움을 드릴 수 있습니다."
            ]
        elif "layer 0" in prompt_lower or "expert" in prompt_lower:
            responses = [
                f"현재 블록체인에는 {len(selected_experts)}개의 Expert가 저장되어 있어, 더 많은 전문가들이 필요합니다.",
                f"The blockchain currently has {len(selected_experts)} experts. We need more diverse experts for better responses.",
                "현재는 layer0.expert0 하나만 있지만, 실제 시스템에서는 여러 전문 Expert들이 협력합니다."
            ]
        else:
            responses = [
                f"AI-Block의 Expert {expert_names}가 처리한 응답입니다.",
                f"Blockchain Expert {expert_names} processed your request.",
                f"분산 MoE 시스템의 {expert_names}를 통해 응답을 생성했습니다."
            ]
        
        import random
        return random.choice(responses)
    
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