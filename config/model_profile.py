"""
Model Profile Configuration
===========================
Centralized configuration for the current active model.
Update this file when switching to a different model.

Current Model: Qwen3-30B-A3B-Instruct-2507
"""

# Model identifier
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
MODEL_NAME = "Qwen3-30B-A3B-Instruct"
MODEL_VERSION = "2507"

# Architecture details
ARCHITECTURE = {
    "type": "MoE",  # Mixture of Experts
    "family": "Qwen3",
    "total_params": "30.5B",
    "active_params": "3.3B",  # Per token
    "non_embedding_params": "29.9B"
}

# Layer configuration
LAYERS = {
    "num_layers": 48,
    "num_attention_heads": 32,  # For Q
    "num_kv_heads": 4,  # For KV (GQA)
    "hidden_size": 4096,  # Adjust based on actual model
    "intermediate_size": 11008,  # Adjust based on actual model
}

# MoE configuration
MOE = {
    "num_experts": 128,  # Total experts per layer
    "num_activated_experts": 8,  # Experts activated per token
    "routing_strategy": "top-k",
    "expert_capacity": 1.25,  # Load balancing factor
}

# Context configuration
CONTEXT = {
    "max_length": 262144,  # 256K native context
    "chunk_size": 8192,  # For processing long contexts
    "sliding_window": None,  # No sliding window for this model
}

# Precision and memory
PRECISION = {
    "default": "fp8",  # Model native precision
    "supported": ["fp8", "fp16", "int8", "int4"],
    "memory_per_billion": {
        "fp32": 4.0,  # GB per billion params
        "fp16": 2.0,
        "fp8": 1.0,
        "int8": 1.0,
        "int4": 0.5
    }
}

# Compute requirements
COMPUTE = {
    "min_gpu_memory": 32,  # GB minimum for fp8
    "recommended_gpu_memory": 48,  # GB recommended
    "optimal_batch_size": {
        "fp8": 8,
        "fp16": 4,
        "int8": 16,
        "int4": 32
    }
}

# Training configuration
TRAINING = {
    "base_learning_rate": 1e-5,
    "warmup_steps": 1000,
    "gradient_accumulation_steps": 4,
    "mixed_precision": True,
    "lora_rank": 64,  # For LoRA fine-tuning
    "lora_alpha": 128,
}

# Inference configuration  
INFERENCE = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "max_new_tokens": 2048,
    "stream": True,
    "use_cache": True,
}

# Blockchain configuration
BLOCKCHAIN = {
    "chain_id": "B",  # Parameter chain for experts
    "block_type": "expert",
    "upload_batch_size": 4,  # Experts to upload in parallel
    "compression": "pickle",  # Serialization format
    "max_block_size": 2 * 1024 * 1024 * 1024,  # 2GB max per block
}

# Helper functions
def get_total_experts():
    """Get total number of experts in the model."""
    return LAYERS["num_layers"] * MOE["num_experts"]

def get_active_params_per_forward():
    """Get number of parameters activated per forward pass."""
    # Base model params + activated experts
    return ARCHITECTURE["active_params"]

def get_memory_requirement(precision="fp8", batch_size=1):
    """Calculate memory requirement in GB."""
    total_params_b = float(ARCHITECTURE["total_params"].rstrip("B"))
    gb_per_b = PRECISION["memory_per_billion"][precision]
    base_memory = total_params_b * gb_per_b
    
    # Add overhead for activations (roughly 20% for batch_size=1)
    activation_overhead = 0.2 * batch_size
    return base_memory * (1 + activation_overhead)

def get_expert_naming(layer_idx, expert_idx):
    """Generate consistent expert naming."""
    return f"layer{layer_idx}.expert{expert_idx}"

def get_model_config(model_name=None):
    """Get complete model configuration as dict.
    
    Args:
        model_name: Optional model name (for compatibility, ignored as we use the profile)
    """
    return {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "architecture": ARCHITECTURE,
        "layers": LAYERS,
        "moe": MOE,
        "context": CONTEXT,
        "precision": PRECISION,
        "compute": COMPUTE,
        "training": TRAINING,
        "inference": INFERENCE,
        "blockchain": BLOCKCHAIN,
    }

# Validation
def validate_config():
    """Validate configuration consistency."""
    checks = []
    
    # Check expert count makes sense
    total_experts = get_total_experts()
    if total_experts != LAYERS["num_layers"] * MOE["num_experts"]:
        checks.append(f"Expert count mismatch: {total_experts}")
    
    # Check memory requirements
    min_memory = get_memory_requirement("fp8", 1)
    if min_memory > COMPUTE["min_gpu_memory"]:
        checks.append(f"Memory requirement {min_memory}GB exceeds minimum {COMPUTE['min_gpu_memory']}GB")
    
    # Check activated experts <= total experts
    if MOE["num_activated_experts"] > MOE["num_experts"]:
        checks.append(f"Activated experts {MOE['num_activated_experts']} > total {MOE['num_experts']}")
    
    return checks

# Run validation on import
_validation_errors = validate_config()
if _validation_errors:
    print(f"⚠️ Model profile validation warnings: {_validation_errors}")