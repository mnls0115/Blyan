"""
Central model configuration for the DNAI project.
Change the model here to update it across the entire project.
"""

# Primary model configuration
DEFAULT_MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
DEFAULT_MODEL_ARCHITECTURE = "mixture-of-experts"
DEFAULT_MODEL_LAYERS = 28  # Qwen has 28 layers
DEFAULT_MODEL_EXPERTS = 16
DEFAULT_MODEL_ACTIVE_PARAMS = "2.7B"  # Active parameters
DEFAULT_MODEL_TOTAL_PARAMS = "14.3B"  # Total parameters

# Fallback models for testing
FALLBACK_MODELS = [
    "EleutherAI/gpt-j-6b",
    "bigscience/bloom-7b1",
]

# Model-specific settings
MODEL_CONFIGS = {
    "Qwen/Qwen1.5-MoE-A2.7B": {
        "architecture": "mixture-of-experts",
        "num_layers": 28,
        "num_experts": 16,
        "active_params": "2.7B",
        "total_params": "14.3B",
        "precision": "fp16",
        "max_sequence_length": 32768,
    },
    "openai/gpt-oss-20b": {
        "architecture": "gpt-neox",
        "num_layers": 24,
        "num_experts": 16,
        "active_params": "20B",
        "total_params": "20B",
        "precision": "fp16",
        "max_sequence_length": 2048,
    },
    "EleutherAI/gpt-j-6b": {
        "architecture": "gpt-j",
        "num_layers": 28,
        "num_experts": 1,
        "active_params": "6B",
        "total_params": "6B",
        "precision": "fp16",
        "max_sequence_length": 2048,
    },
}

def get_model_config(model_name: str = None):
    """Get configuration for a specific model."""
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
    
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    
    # Return default config for unknown models
    return {
        "architecture": "unknown",
        "num_layers": 24,
        "num_experts": 1,
        "active_params": "unknown",
        "total_params": "unknown",
        "precision": "fp16",
        "max_sequence_length": 2048,
    }