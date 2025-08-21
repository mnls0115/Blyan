"""
Central model configuration for the DNAI project.
Change the model here to update it across the entire project.
"""

# Primary model configuration
DEFAULT_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
DEFAULT_MODEL_ARCHITECTURE = "mixture-of-experts"
DEFAULT_MODEL_LAYERS = 48  # Qwen3-30B has 48 layers
DEFAULT_MODEL_EXPERTS = 128  # 128 experts total
DEFAULT_MODEL_ACTIVE_PARAMS = "3.3B"  # 3.3B activated parameters
DEFAULT_MODEL_TOTAL_PARAMS = "30.5B"  # 30.5B total parameters

# Fallback models for testing
FALLBACK_MODELS = [
    "EleutherAI/gpt-j-6b",
    "bigscience/bloom-7b1",
]

# Model-specific settings
MODEL_CONFIGS = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8": {
        "architecture": "mixture-of-experts",
        "num_layers": 48,
        "num_experts": 128,
        "activated_experts": 8,
        "active_params": "3.3B",  # 3.3B activated per token
        "total_params": "30.5B",  # 30.5B total parameters
        "non_embedding_params": "29.9B",
        "attention_heads_q": 32,  # Query heads
        "attention_heads_kv": 4,  # Key-Value heads (GQA)
        "precision": "fp8",
        "max_sequence_length": 262144,  # 256K context length native
        "requires_transformers": ">=4.51.0",
        "torch_dtype": "auto",
        "device_map": "auto",
        "quantization": "fp8",
        "thinking_mode": False,  # Non-thinking mode only
        "enable_thinking": False
    },
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