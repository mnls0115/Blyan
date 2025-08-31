"""
Default Model Adapter
=====================
Fallback adapter for unknown model families.
Uses generic HuggingFace patterns.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Any
from transformers import PretrainedConfig, AutoModelForCausalLM

from .base import ModelAdapter
from .registry import register_adapter

logger = logging.getLogger(__name__)


class DefaultAdapter(ModelAdapter):
    """Default adapter for unknown model families."""
    
    def build_empty_model(
        self, 
        config: PretrainedConfig, 
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ) -> nn.Module:
        """Build empty model structure using AutoModel."""
        logger.info(f"Using default adapter for model type: {getattr(config, 'model_type', 'unknown')}")
        
        # Create model on meta device first
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        
        # Move to target device
        model = model.to_empty(device=device)
        
        # Ensure correct dtype
        model = model.to(dtype=dtype)
        
        logger.info(f"Created model structure: {model.__class__.__name__}")
        return model
    
    def translate_key(
        self,
        layer_name: str,
        tensor_key: str,
        config: PretrainedConfig
    ) -> str:
        """
        Translate blockchain keys using common patterns.
        
        This uses the most common HuggingFace patterns that work
        for most transformer models.
        """
        # Handle special layers
        if layer_name == "embedding":
            # Try common embedding layer names
            if hasattr(config, 'model_type'):
                model_type = config.model_type.lower()
                
                # GPT-style
                if 'gpt' in model_type:
                    return f"transformer.wte.{tensor_key}" if tensor_key != "weight" else "transformer.wte.weight"
                # BERT-style
                elif 'bert' in model_type:
                    return f"embeddings.word_embeddings.{tensor_key}" if tensor_key != "weight" else "embeddings.word_embeddings.weight"
            
            # Default to most common pattern
            if tensor_key == "weight":
                return "model.embed_tokens.weight"
            else:
                return f"model.embed_tokens.{tensor_key}"
        
        elif layer_name == "lm_head":
            # Most models use lm_head
            if tensor_key == "weight":
                return "lm_head.weight"
            else:
                return f"lm_head.{tensor_key}"
        
        elif layer_name == "model_norm":
            # Try to detect the right norm layer
            if hasattr(config, 'model_type'):
                model_type = config.model_type.lower()
                
                # GPT-style
                if 'gpt' in model_type:
                    return f"transformer.ln_f.{tensor_key}" if tensor_key != "weight" else "transformer.ln_f.weight"
            
            # Default pattern
            if tensor_key == "weight":
                return "model.norm.weight"
            else:
                return f"model.norm.{tensor_key}"
        
        elif layer_name.startswith("layer_"):
            # Extract layer number
            layer_num = int(layer_name.split("_")[1])
            
            # Detect model structure
            if hasattr(config, 'model_type'):
                model_type = config.model_type.lower()
                
                # GPT-style
                if 'gpt' in model_type:
                    prefix = f"transformer.h.{layer_num}"
                # BERT-style
                elif 'bert' in model_type:
                    prefix = f"encoder.layer.{layer_num}"
                else:
                    # Default transformer pattern
                    prefix = f"model.layers.{layer_num}"
            else:
                prefix = f"model.layers.{layer_num}"
            
            # Direct tensor
            if '.' not in tensor_key:
                return f"{prefix}.{tensor_key}"
            
            # Common patterns
            if tensor_key.startswith('self_attn.') or tensor_key.startswith('attention.'):
                return f"{prefix}.{tensor_key}"
            elif tensor_key.startswith('mlp.') or tensor_key.startswith('feed_forward.'):
                return f"{prefix}.{tensor_key}"
            elif 'layernorm' in tensor_key.lower() or 'layer_norm' in tensor_key.lower():
                return f"{prefix}.{tensor_key}"
            else:
                return f"{prefix}.{tensor_key}"
        
        # Other weights
        elif layer_name == "other_weights":
            return tensor_key
        
        # Unknown layer
        else:
            logger.warning(f"Unknown layer name: {layer_name}, using direct mapping")
            if tensor_key:
                return f"{layer_name}.{tensor_key}"
            return layer_name
    
    def get_expected_keys(self, config: PretrainedConfig) -> Dict[str, List[str]]:
        """
        Get expected keys using generic patterns.
        
        Since we don't know the exact model architecture, we use
        permissive patterns that accept most common structures.
        """
        num_layers = config.num_hidden_layers
        
        # We can't know exact keys, so we'll be permissive
        core_weights = []
        
        # Common embedding patterns
        core_weights.extend([
            "model.embed_tokens.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight"
        ])
        
        # Common output head patterns
        core_weights.extend([
            "lm_head.weight",
            "cls.predictions.decoder.weight"
        ])
        
        # Common final norm patterns
        core_weights.extend([
            "model.norm.weight",
            "transformer.ln_f.weight",
            "final_layer_norm.weight"
        ])
        
        # Since we don't know the exact structure, we'll rely on
        # the actual loaded keys rather than predicting them
        
        # Optional weights - be very permissive
        optional_weights = ['bias']  # Any key with 'bias' is optional
        
        # Expected patterns - accept various architectures
        expected_patterns = [
            'attention', 'self_attn', 'mlp', 'feed_forward',
            'layernorm', 'layer_norm', 'embed', 'head',
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
            'q_norm', 'k_norm'  # Accept these if present
        ]
        
        return {
            'core_weights': core_weights,
            'optional_weights': optional_weights,
            'expected_patterns': expected_patterns
        }
    
    def validate_config(self, config: PretrainedConfig) -> bool:
        """Default adapter accepts any config."""
        return True
    
    def get_critical_components(self, config: PretrainedConfig) -> List[str]:
        """Get critical components - use generic names."""
        return ["embedding", "lm_head", "model_norm"]
    
    def filter_missing_keys(
        self, 
        missing_keys: List[str],
        config: PretrainedConfig
    ) -> tuple[List[str], List[str]]:
        """
        Be more permissive with missing keys for unknown models.
        """
        critical = []
        ignorable = []
        
        for key in missing_keys:
            # Biases are almost always optional
            if 'bias' in key:
                ignorable.append(key)
            # Buffers are optional
            elif 'buffer' in key or 'mask' in key:
                ignorable.append(key)
            # Position embeddings might be optional
            elif 'position' in key and 'embed' in key:
                ignorable.append(key)
            else:
                # Check if it's a critical component
                if any(comp in key for comp in ['embed_tokens', 'lm_head', 'norm', 'ln_f']):
                    critical.append(key)
                else:
                    # For unknown models, be less strict
                    ignorable.append(key)
        
        return critical, ignorable
    
    def filter_unexpected_keys(
        self,
        unexpected_keys: List[str],
        config: PretrainedConfig
    ) -> tuple[List[str], List[str]]:
        """
        Be very permissive with unexpected keys for unknown models.
        """
        problematic = []
        acceptable = []
        
        # For default adapter, accept almost everything
        for key in unexpected_keys:
            # Accept any reasonable-looking key
            if any(pattern in key.lower() for pattern in [
                'weight', 'bias', 'norm', 'embed', 'attn', 'mlp',
                'proj', 'gate', 'layer', 'head', 'expert'
            ]):
                acceptable.append(key)
            else:
                # Even unknown keys are acceptable for default adapter
                acceptable.append(key)
                logger.debug(f"Accepting unexpected key in default adapter: {key}")
        
        return problematic, acceptable


# Register the default adapter
register_adapter("default", DefaultAdapter)
register_adapter("auto", DefaultAdapter)
register_adapter("unknown", DefaultAdapter)