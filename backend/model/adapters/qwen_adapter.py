"""
Qwen Model Family Adapter
==========================
Adapter for Qwen/Qwen2/Qwen3 models.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any
from transformers import PretrainedConfig, AutoModelForCausalLM

from .base import ModelAdapter
from .registry import register_adapter

logger = logging.getLogger(__name__)


class QwenAdapter(ModelAdapter):
    """Adapter for Qwen model family."""
    
    def build_empty_model(
        self, 
        config: PretrainedConfig, 
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ) -> nn.Module:
        """Build empty Qwen model structure."""
        # Create model on meta device first
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        
        # Move to target device
        model = model.to_empty(device=device)
        
        # Ensure correct dtype
        model = model.to(dtype=dtype)
        
        logger.info(f"Created Qwen model structure: {model.__class__.__name__}")
        return model
    
    def translate_key(
        self,
        layer_name: str,
        tensor_key: str,
        config: PretrainedConfig
    ) -> str:
        """Translate blockchain keys to Qwen state_dict keys."""
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
            
            # Direct tensor (e.g., "weight" alone)
            if '.' not in tensor_key:
                return f"{prefix}.{tensor_key}"
            
            # Handle nested keys
            parts = tensor_key.split('.')
            
            # Attention weights
            if tensor_key.startswith('self_attn.'):
                return f"{prefix}.{tensor_key}"
            
            # MLP weights
            elif tensor_key.startswith('mlp.'):
                return f"{prefix}.{tensor_key}"
            
            # Layer norms
            elif tensor_key in ['input_layernorm.weight', 'post_attention_layernorm.weight']:
                return f"{prefix}.{tensor_key}"
            
            # Handle q_norm/k_norm for Qwen variants that have them
            elif 'q_norm' in tensor_key or 'k_norm' in tensor_key:
                # These go under self_attn
                return f"{prefix}.self_attn.{tensor_key}"
            
            # Fallback
            else:
                return f"{prefix}.{tensor_key}"
        
        # Other weights (if any)
        elif layer_name == "other_weights":
            return tensor_key
        
        # Unknown layer
        else:
            logger.warning(f"Unknown layer name: {layer_name}")
            return f"{layer_name}.{tensor_key}"
    
    def get_expected_keys(self, config: PretrainedConfig) -> Dict[str, List[str]]:
        """Get expected keys for Qwen models."""
        num_layers = config.num_hidden_layers
        
        # Core weights that must be present
        core_weights = [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight"
        ]
        
        # Add all layer weights
        for i in range(num_layers):
            prefix = f"model.layers.{i}"
            
            # Attention weights (no biases in Qwen)
            core_weights.extend([
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
            ])
            
            # MLP weights
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
        
        # Optional weights (Qwen typically doesn't have these)
        optional_weights = []
        for i in range(num_layers):
            prefix = f"model.layers.{i}"
            # Biases (not present in standard Qwen)
            optional_weights.extend([
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.bias",
                f"{prefix}.self_attn.o_proj.bias",
            ])
        
        # Some Qwen variants have q_norm/k_norm
        expected_patterns = ['q_norm', 'k_norm']
        
        return {
            'core_weights': core_weights,
            'optional_weights': optional_weights,
            'expected_patterns': expected_patterns
        }
    
    def validate_config(self, config: PretrainedConfig) -> bool:
        """Validate Qwen config."""
        # Check for Qwen-specific attributes
        if hasattr(config, 'model_type'):
            if 'qwen' in config.model_type.lower():
                return True
        
        # Check architectures
        if hasattr(config, 'architectures'):
            if any('qwen' in arch.lower() for arch in config.architectures):
                return True
        
        return False
    
    def get_critical_components(self, config: PretrainedConfig) -> List[str]:
        """Get critical components for Qwen."""
        return ["embedding", "lm_head", "model_norm"]


# Register the adapter
register_adapter("qwen", QwenAdapter)
register_adapter("qwen2", QwenAdapter)
register_adapter("qwen3", QwenAdapter)