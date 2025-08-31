"""
Model Adapter Registry
======================
Registry for model family adapters with auto-detection.
"""

import logging
from typing import Dict, Optional, Type
from transformers import PretrainedConfig

from .base import ModelAdapter

logger = logging.getLogger(__name__)

# Global registry
_ADAPTER_REGISTRY: Dict[str, Type[ModelAdapter]] = {}

# Sentinel for auto-detection
AUTO_DETECT = "auto"


def register_adapter(name: str, adapter_class: Type[ModelAdapter]) -> None:
    """
    Register an adapter class for a model family.
    
    Args:
        name: Model family name (e.g., "qwen", "llama", "mixtral")
        adapter_class: Adapter class
    """
    _ADAPTER_REGISTRY[name.lower()] = adapter_class
    logger.debug(f"Registered adapter for {name}")


def detect_model_family(config: PretrainedConfig) -> str:
    """
    Auto-detect model family from config.
    
    Args:
        config: Model configuration
        
    Returns:
        Model family name
    """
    # Check architectures field
    architectures = getattr(config, 'architectures', [])
    if architectures:
        arch_name = architectures[0].lower()
        
        # Map architecture names to families
        if 'qwen' in arch_name:
            return 'qwen'
        elif 'llama' in arch_name:
            return 'llama'
        elif 'mixtral' in arch_name or 'mistral' in arch_name:
            return 'mixtral'
        elif 'opt' in arch_name:
            return 'opt'
        elif 'gpt' in arch_name:
            return 'gpt'
    
    # Check model_type field
    model_type = getattr(config, 'model_type', '').lower()
    if model_type:
        if 'qwen' in model_type:
            return 'qwen'
        elif 'llama' in model_type:
            return 'llama'
        elif 'mixtral' in model_type or 'mistral' in model_type:
            return 'mixtral'
    
    # Check for specific config attributes
    if hasattr(config, 'rope_theta') and hasattr(config, 'sliding_window'):
        # Likely Mistral/Mixtral family
        return 'mixtral'
    elif hasattr(config, 'rope_scaling'):
        # Could be Llama or Qwen with RoPE scaling
        if hasattr(config, 'num_key_value_heads'):
            # GQA suggests newer model
            return 'llama'  # Default to Llama for GQA models
    
    # Default fallback
    logger.warning(f"Could not detect model family for {config}, using default adapter")
    return 'default'


def get_adapter(
    model_name_or_config: str | PretrainedConfig,
    auto_detect: bool = True
) -> ModelAdapter:
    """
    Get adapter for a model.
    
    Args:
        model_name_or_config: Model name, family name, or config object
        auto_detect: Whether to auto-detect family from config
        
    Returns:
        Model adapter instance
    """
    # If it's a config object and auto-detect is enabled
    if isinstance(model_name_or_config, PretrainedConfig) and auto_detect:
        family = detect_model_family(model_name_or_config)
    elif isinstance(model_name_or_config, str):
        # Check if it's a registered family name
        family = model_name_or_config.lower()
        
        # If not registered, try to extract family from model ID
        if family not in _ADAPTER_REGISTRY:
            # Extract family from model ID (e.g., "Qwen/Qwen3-8B" -> "qwen")
            if '/' in family:
                family = family.split('/')[0].lower()
            elif '-' in family:
                family = family.split('-')[0].lower()
            elif '_' in family:
                family = family.split('_')[0].lower()
            
            # Map common variations
            if 'qwen' in family:
                family = 'qwen'
            elif 'llama' in family:
                family = 'llama'
            elif 'mixtral' in family or 'mistral' in family:
                family = 'mixtral'
    else:
        family = 'default'
    
    # Get adapter class
    adapter_class = _ADAPTER_REGISTRY.get(family)
    
    if adapter_class is None:
        logger.warning(f"No adapter found for {family}, using default")
        adapter_class = _ADAPTER_REGISTRY.get('default')
        
        if adapter_class is None:
            raise ValueError(f"No adapter found for {family} and no default adapter available")
    
    # Return instance
    return adapter_class()