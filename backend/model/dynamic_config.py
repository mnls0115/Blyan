"""
Dynamic Model Configuration
Automatically detects and configures model architecture
"""

import torch
from transformers import AutoConfig
from typing import Dict, Any, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

class DynamicModelConfig:
    """
    Dynamically detect and configure model architecture.
    Supports various transformer architectures without hardcoding.
    """
    
    def __init__(self, model_name_or_path: str = None):
        """
        Initialize dynamic configuration.
        
        Args:
            model_name_or_path: Model name or path (defaults to env var)
        """
        self.model_name = model_name_or_path or os.getenv('MODEL_NAME', 'Qwen/Qwen3-8B')
        self._config = None
        self._architecture_info = None
        self._load_config()
    
    def _load_config(self):
        """Load model configuration from HuggingFace or cache."""
        try:
            # Try to load from HuggingFace
            self._config = AutoConfig.from_pretrained(self.model_name)
            logger.info(f"Loaded config for {self.model_name}")
            
            # Extract architecture information
            self._parse_architecture()
        except Exception as e:
            logger.warning(f"Could not load config from HF: {e}")
            # Fall back to common defaults
            self._use_defaults()
    
    def _parse_architecture(self):
        """Parse architecture details from config."""
        config = self._config
        
        self._architecture_info = {
            # Model dimensions
            'hidden_size': getattr(config, 'hidden_size', 4096),
            'num_hidden_layers': getattr(config, 'num_hidden_layers', 32),
            'num_attention_heads': getattr(config, 'num_attention_heads', 32),
            'intermediate_size': getattr(config, 'intermediate_size', 11008),
            'vocab_size': getattr(config, 'vocab_size', 32000),
            
            # Architecture specifics
            'model_type': getattr(config, 'model_type', 'unknown'),
            'architectures': getattr(config, 'architectures', []),
            'max_position_embeddings': getattr(config, 'max_position_embeddings', 2048),
            
            # Layer names (varies by architecture)
            'embedding_layer': self._get_embedding_layer_name(),
            'layer_prefix': self._get_layer_prefix(),
            'lm_head_name': self._get_lm_head_name(),
            'norm_name': self._get_norm_name(),
            
            # Precision
            'torch_dtype': self._get_dtype(),
            
            # Special tokens
            'bos_token_id': getattr(config, 'bos_token_id', 1),
            'eos_token_id': getattr(config, 'eos_token_id', 2),
            'pad_token_id': getattr(config, 'pad_token_id', 0),
        }
    
    def _get_embedding_layer_name(self) -> str:
        """Get embedding layer name for architecture."""
        model_type = self._config.model_type if self._config else 'unknown'
        
        # Common patterns
        embedding_names = {
            'llama': 'model.embed_tokens',
            'qwen2': 'model.embed_tokens',
            'gpt2': 'transformer.wte',
            'gptj': 'transformer.wte',
            'opt': 'model.decoder.embed_tokens',
            'bloom': 'transformer.word_embeddings',
            'mistral': 'model.embed_tokens',
            'mixtral': 'model.embed_tokens',
        }
        
        return embedding_names.get(model_type, 'model.embed_tokens')
    
    def _get_layer_prefix(self) -> str:
        """Get transformer layer prefix for architecture."""
        model_type = self._config.model_type if self._config else 'unknown'
        
        # Common patterns
        layer_prefixes = {
            'llama': 'model.layers',
            'qwen2': 'model.layers',
            'gpt2': 'transformer.h',
            'gptj': 'transformer.h',
            'opt': 'model.decoder.layers',
            'bloom': 'transformer.h',
            'mistral': 'model.layers',
            'mixtral': 'model.layers',
        }
        
        return layer_prefixes.get(model_type, 'model.layers')
    
    def _get_lm_head_name(self) -> str:
        """Get LM head name for architecture."""
        model_type = self._config.model_type if self._config else 'unknown'
        
        # Common patterns
        lm_head_names = {
            'llama': 'lm_head',
            'qwen2': 'lm_head',
            'gpt2': 'lm_head',
            'gptj': 'lm_head',
            'opt': 'lm_head',
            'bloom': 'lm_head',
            'mistral': 'lm_head',
            'mixtral': 'lm_head',
        }
        
        return lm_head_names.get(model_type, 'lm_head')
    
    def _get_norm_name(self) -> str:
        """Get final norm layer name for architecture."""
        model_type = self._config.model_type if self._config else 'unknown'
        
        # Common patterns
        norm_names = {
            'llama': 'model.norm',
            'qwen2': 'model.norm',
            'gpt2': 'transformer.ln_f',
            'gptj': 'transformer.ln_f',
            'opt': 'model.decoder.final_layer_norm',
            'bloom': 'transformer.ln_f',
            'mistral': 'model.norm',
            'mixtral': 'model.norm',
        }
        
        return norm_names.get(model_type, 'model.norm')
    
    def _get_dtype(self) -> torch.dtype:
        """Get torch dtype for model."""
        if self._config and hasattr(self._config, 'torch_dtype'):
            dtype_str = str(self._config.torch_dtype)
            if 'bfloat16' in dtype_str:
                return torch.bfloat16
            elif 'float16' in dtype_str:
                return torch.float16
            elif 'float32' in dtype_str:
                return torch.float32
        
        # Default to BF16 as per requirements
        return torch.bfloat16
    
    def _use_defaults(self):
        """Use default configuration when auto-detection fails."""
        # Detect from model name
        if 'qwen' in self.model_name.lower():
            if '7b' in self.model_name.lower() or '8b' in self.model_name.lower():
                num_layers = 32
                hidden_size = 4096
            elif '14b' in self.model_name.lower():
                num_layers = 40
                hidden_size = 5120
            else:
                num_layers = 32
                hidden_size = 4096
        elif 'llama' in self.model_name.lower():
            if '7b' in self.model_name.lower():
                num_layers = 32
                hidden_size = 4096
            elif '13b' in self.model_name.lower():
                num_layers = 40
                hidden_size = 5120
            elif '70b' in self.model_name.lower():
                num_layers = 80
                hidden_size = 8192
            else:
                num_layers = 32
                hidden_size = 4096
        else:
            # Generic defaults
            num_layers = 32
            hidden_size = 4096
        
        self._architecture_info = {
            'hidden_size': hidden_size,
            'num_hidden_layers': num_layers,
            'num_attention_heads': hidden_size // 128,
            'intermediate_size': hidden_size * 2.75,
            'vocab_size': 32000,
            'model_type': 'unknown',
            'architectures': [],
            'max_position_embeddings': 2048,
            'embedding_layer': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'lm_head_name': 'lm_head',
            'norm_name': 'model.norm',
            'torch_dtype': torch.bfloat16,
            'bos_token_id': 1,
            'eos_token_id': 2,
            'pad_token_id': 0,
        }
    
    @property
    def num_layers(self) -> int:
        """Get number of transformer layers."""
        return self._architecture_info['num_hidden_layers']
    
    @property
    def hidden_size(self) -> int:
        """Get hidden dimension size."""
        return self._architecture_info['hidden_size']
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._architecture_info['vocab_size']
    
    @property
    def dtype(self) -> torch.dtype:
        """Get model dtype."""
        return self._architecture_info['torch_dtype']
    
    def get_layer_names(self) -> list[str]:
        """Get all layer names for the model."""
        layers = []
        
        # Embedding
        layers.append(self._architecture_info['embedding_layer'])
        
        # Transformer layers
        layer_prefix = self._architecture_info['layer_prefix']
        for i in range(self.num_layers):
            layers.append(f"{layer_prefix}.{i}")
        
        # Final norm and LM head
        layers.append(self._architecture_info['norm_name'])
        layers.append(self._architecture_info['lm_head_name'])
        
        return layers
    
    def get_pipeline_stages(self, num_gpus: int) -> list[Dict[str, Any]]:
        """
        Divide model into pipeline stages for multiple GPUs.
        
        Args:
            num_gpus: Number of GPUs available
            
        Returns:
            List of stage configurations
        """
        stages = []
        layers_per_gpu = self.num_layers // num_gpus
        remainder = self.num_layers % num_gpus
        
        current_layer = 0
        for gpu_id in range(num_gpus):
            # Distribute remainder layers
            gpu_layers = layers_per_gpu + (1 if gpu_id < remainder else 0)
            
            stage = {
                'stage_id': gpu_id,
                'layer_range': [current_layer, current_layer + gpu_layers],
                'has_embedding': gpu_id == 0,  # First GPU gets embedding
                'has_lm_head': gpu_id == num_gpus - 1,  # Last GPU gets LM head
                'device': f'cuda:{gpu_id}'
            }
            
            stages.append(stage)
            current_layer += gpu_layers
        
        return stages
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return self._architecture_info.copy()
    
    @classmethod
    def from_model(cls, model) -> 'DynamicModelConfig':
        """Create config from loaded model instance."""
        config = cls()
        
        # Update from actual model
        if hasattr(model, 'config'):
            config._config = model.config
            config._parse_architecture()
        
        # Count actual layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            config._architecture_info['num_hidden_layers'] = len(model.model.layers)
        
        return config

# Global config instance (singleton)
_global_config: Optional[DynamicModelConfig] = None

def get_model_config() -> DynamicModelConfig:
    """Get or create global model configuration."""
    global _global_config
    if _global_config is None:
        _global_config = DynamicModelConfig()
    return _global_config

def reset_model_config(model_name: str = None):
    """Reset global configuration with new model."""
    global _global_config
    _global_config = DynamicModelConfig(model_name)
    return _global_config

# Export
__all__ = [
    'DynamicModelConfig',
    'get_model_config',
    'reset_model_config'
]