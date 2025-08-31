"""
Mixtral/Mistral Model Family Adapter
=====================================
Adapter for Mixtral and Mistral models.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any
from transformers import PretrainedConfig, AutoModelForCausalLM

from .base import ModelAdapter
from .registry import register_adapter

logger = logging.getLogger(__name__)


class MixtralAdapter(ModelAdapter):
    """Adapter for Mixtral/Mistral model family."""
    
    def build_empty_model(
        self, 
        config: PretrainedConfig, 
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ) -> nn.Module:
        """Build empty Mixtral/Mistral model structure."""
        # Create model on meta device first
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        
        # Move to target device
        model = model.to_empty(device=device)
        
        # Ensure correct dtype
        model = model.to(dtype=dtype)
        
        logger.info(f"Created Mixtral/Mistral model structure: {model.__class__.__name__}")
        return model
    
    def translate_key(
        self,
        layer_name: str,
        tensor_key: str,
        config: PretrainedConfig
    ) -> str:
        """Translate blockchain keys to Mixtral/Mistral state_dict keys."""
        # Handle special layers
        if layer_name == "embedding":
            if tensor_key == "weight":
                return "model.embed_tokens.weight"
            else:
                return f"model.embed_tokens.{tensor_key}"
        
        elif layer_name == "lm_head":
            if tensor_key == "weight":
                return "lm_head.weight"
            else:
                return f"lm_head.{tensor_key}"
        
        elif layer_name == "model_norm":
            if tensor_key == "weight":
                return "model.norm.weight"
            else:
                return f"model.norm.{tensor_key}"
        
        elif layer_name.startswith("layer_"):
            # Extract layer number
            layer_num = int(layer_name.split("_")[1])
            prefix = f"model.layers.{layer_num}"
            
            # Direct tensor
            if '.' not in tensor_key:
                return f"{prefix}.{tensor_key}"
            
            # Check if this is a MoE model (Mixtral) or dense (Mistral)
            is_moe = hasattr(config, 'num_local_experts') and config.num_local_experts > 1
            
            if is_moe:
                # Mixtral MoE structure
                if tensor_key.startswith('block_sparse_moe.'):
                    return f"{prefix}.{tensor_key}"
                elif tensor_key.startswith('self_attn.'):
                    return f"{prefix}.{tensor_key}"
                elif 'gate' in tensor_key:
                    return f"{prefix}.block_sparse_moe.{tensor_key}"
                elif 'experts' in tensor_key:
                    return f"{prefix}.block_sparse_moe.{tensor_key}"
            else:
                # Mistral dense structure (similar to Llama)
                if tensor_key.startswith('self_attn.'):
                    return f"{prefix}.{tensor_key}"
                elif tensor_key.startswith('mlp.'):
                    return f"{prefix}.{tensor_key}"
            
            # Layer norms
            if tensor_key in ['input_layernorm.weight', 'post_attention_layernorm.weight']:
                return f"{prefix}.{tensor_key}"
            
            # Fallback
            return f"{prefix}.{tensor_key}"
        
        # Other weights
        elif layer_name == "other_weights":
            return tensor_key
        
        # Unknown layer
        else:
            logger.warning(f"Unknown layer name: {layer_name}")
            return f"{layer_name}.{tensor_key}"
    
    def get_expected_keys(self, config: PretrainedConfig) -> Dict[str, List[str]]:
        """Get expected keys for Mixtral/Mistral models."""
        num_layers = config.num_hidden_layers
        is_moe = hasattr(config, 'num_local_experts') and config.num_local_experts > 1
        
        # Core weights that must be present
        core_weights = [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight"
        ]
        
        # Add all layer weights
        for i in range(num_layers):
            prefix = f"model.layers.{i}"
            
            # Attention weights (no biases)
            core_weights.extend([
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
            ])
            
            if is_moe:
                # Mixtral MoE structure
                core_weights.append(f"{prefix}.block_sparse_moe.gate.weight")
                
                # Expert weights
                num_experts = config.num_local_experts
                for expert_idx in range(num_experts):
                    core_weights.extend([
                        f"{prefix}.block_sparse_moe.experts.{expert_idx}.w1.weight",
                        f"{prefix}.block_sparse_moe.experts.{expert_idx}.w2.weight",
                        f"{prefix}.block_sparse_moe.experts.{expert_idx}.w3.weight",
                    ])
            else:
                # Mistral dense MLP
                core_weights.extend([
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                ])
            
            # Layer norms
            core_weights.extend([
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
            ])
        
        # Optional weights (not present)
        optional_weights = []
        for i in range(num_layers):
            prefix = f"model.layers.{i}"
            optional_weights.extend([
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.bias",
                f"{prefix}.self_attn.o_proj.bias",
                "lm_head.bias"
            ])
        
        # Expected patterns
        expected_patterns = []
        if is_moe:
            expected_patterns.extend(['experts', 'gate', 'block_sparse_moe'])
        
        return {
            'core_weights': core_weights,
            'optional_weights': optional_weights,
            'expected_patterns': expected_patterns
        }
    
    def validate_config(self, config: PretrainedConfig) -> bool:
        """Validate Mixtral/Mistral config."""
        # Check for Mixtral/Mistral-specific attributes
        if hasattr(config, 'model_type'):
            model_type = config.model_type.lower()
            if 'mixtral' in model_type or 'mistral' in model_type:
                return True
        
        # Check architectures
        if hasattr(config, 'architectures'):
            for arch in config.architectures:
                arch_lower = arch.lower()
                if 'mixtral' in arch_lower or 'mistral' in arch_lower:
                    return True
        
        # Check for sliding window (Mistral feature)
        if hasattr(config, 'sliding_window'):
            return True
        
        return False
    
    def get_critical_components(self, config: PretrainedConfig) -> List[str]:
        """Get critical components for Mixtral/Mistral."""
        return ["embedding", "lm_head", "model_norm"]


# Register the adapter
register_adapter("mixtral", MixtralAdapter)
register_adapter("mistral", MixtralAdapter)
register_adapter("mistralai", MixtralAdapter)