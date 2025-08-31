"""
Base Model Adapter Interface
=============================
Defines the interface for model-specific adapters.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from transformers import PretrainedConfig


class ModelAdapter(ABC):
    """Base adapter interface for model family-specific handling."""
    
    @abstractmethod
    def build_empty_model(
        self, 
        config: PretrainedConfig, 
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ) -> nn.Module:
        """
        Build empty model structure from config.
        
        Args:
            config: HuggingFace config object
            device: Target device
            dtype: Target dtype (should be BF16)
            
        Returns:
            Empty model on target device with correct dtype
        """
        pass
    
    @abstractmethod
    def translate_key(
        self,
        layer_name: str,
        tensor_key: str,
        config: PretrainedConfig
    ) -> str:
        """
        Translate blockchain layer/tensor names to model state_dict keys.
        
        Args:
            layer_name: Blockchain layer name (e.g., "embedding", "layer_0", "lm_head")
            tensor_key: Tensor name within layer (e.g., "weight", "self_attn.q_proj.weight")
            config: Model configuration
            
        Returns:
            Final state_dict key (e.g., "model.embed_tokens.weight")
        """
        pass
    
    @abstractmethod
    def get_expected_keys(self, config: PretrainedConfig) -> Dict[str, List[str]]:
        """
        Get expected keys for this model architecture.
        
        Args:
            config: Model configuration
            
        Returns:
            Dict with:
                - core_weights: Critical weights that must be present
                - optional_weights: Optional weights (e.g., biases)
                - expected_patterns: Patterns that should be present
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: PretrainedConfig) -> bool:
        """
        Validate that config is compatible with this adapter.
        
        Args:
            config: Model configuration
            
        Returns:
            True if compatible
        """
        pass
    
    def get_layer_count(self, config: PretrainedConfig) -> int:
        """Get number of transformer layers from config."""
        return config.num_hidden_layers
    
    def get_critical_components(self, config: PretrainedConfig) -> List[str]:
        """
        Get list of critical component names that must be present.
        
        Default implementation returns standard components.
        Override for model-specific requirements.
        """
        return ["embedding", "lm_head", "model_norm"]
    
    def filter_missing_keys(
        self, 
        missing_keys: List[str],
        config: PretrainedConfig
    ) -> Tuple[List[str], List[str]]:
        """
        Filter missing keys into critical and ignorable.
        
        Args:
            missing_keys: List of missing state_dict keys
            config: Model configuration
            
        Returns:
            (critical_missing, ignorable_missing)
        """
        critical = []
        ignorable = []
        
        expected = self.get_expected_keys(config)
        
        for key in missing_keys:
            # Check if it's a critical weight
            if any(core in key for core in expected.get('core_weights', [])):
                critical.append(key)
            # Check if it's an optional weight (e.g., bias that doesn't exist)
            elif any(opt in key for opt in expected.get('optional_weights', [])):
                ignorable.append(key)
            else:
                # Default to critical if unknown
                critical.append(key)
        
        return critical, ignorable
    
    def filter_unexpected_keys(
        self,
        unexpected_keys: List[str],
        config: PretrainedConfig
    ) -> Tuple[List[str], List[str]]:
        """
        Filter unexpected keys into problematic and acceptable.
        
        Args:
            unexpected_keys: List of unexpected state_dict keys
            config: Model configuration
            
        Returns:
            (problematic_keys, acceptable_keys)
        """
        problematic = []
        acceptable = []
        
        expected = self.get_expected_keys(config)
        
        for key in unexpected_keys:
            # Check against expected patterns
            if any(pattern in key for pattern in expected.get('expected_patterns', [])):
                acceptable.append(key)
            else:
                problematic.append(key)
        
        return problematic, acceptable