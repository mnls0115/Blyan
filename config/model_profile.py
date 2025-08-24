"""
Model Profile Configuration - Dense Model
==========================================
Centralized configuration for the current active model.
This file defines the dense model architecture and partitioning strategy.

Current Model: Qwen3-8B (Dense)
"""

# Model identifier
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_NAME = "Qwen3-8B"
MODEL_VERSION = "latest"

# Architecture details
ARCHITECTURE = {
    "type": "dense",  # Dense transformer (not MoE)
    "family": "Qwen3",
    "total_params": "8.2B",
    "non_embedding_params": "6.95B",
    "embedding_params": "1.25B"
}

# Layer configuration
LAYERS = {
    "num_layers": 36,
    "num_attention_heads": 32,  # For Q
    "num_kv_heads": 8,  # For KV (GQA)
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "head_dim": 112,  # hidden_size / num_attention_heads
    "vocab_size": 152064
}

# Context configuration
CONTEXT = {
    "max_length": 32768,  # 32K native context
    "extended_length": 131072,  # 128K with YaRN
    "chunk_size": 2048,  # For processing contexts
    "sliding_window": None
}

# Precision and memory
PRECISION = {
    "default": "fp16",
    "supported": ["fp32", "fp16", "int8", "int4"],
    "memory_per_billion": {
        "fp32": 4.0,  # GB per billion params
        "fp16": 2.0,
        "int8": 1.0,
        "int4": 0.5
    }
}

# Per-layer memory requirements (GB)
LAYER_MEMORY = {
    "fp32": {
        "per_layer": 0.772,  # 6.95B / 36 / 1e9 * 4 bytes
        "embedding": 1.090,
        "lm_head": 1.090
    },
    "fp16": {
        "per_layer": 0.386,  # 6.95B / 36 / 1e9 * 2 bytes
        "embedding": 0.545,
        "lm_head": 0.545
    },
    "int8": {
        "per_layer": 0.193,
        "embedding": 0.273,
        "lm_head": 0.273
    },
    "int4": {
        "per_layer": 0.097,
        "embedding": 0.136,
        "lm_head": 0.136
    }
}

# Partitioning configuration for distributed inference
PARTITION = {
    # Default profile for small GPUs
    "default": {
        "target_vram_gb": 4.0,
        "reserved_headroom_gb": 1.0,  # For CUDA context, temp buffers
        "kv_cache_budget_gb": 0.6,
        "runtime_buffer_gb": 0.2,
        "weight_precision": "int4",
        "activation_checkpointing": True,
        "microbatch_size": 1
    },
    
    # GPU profiles
    "profiles": {
        "small_gpu": {  # 4GB VRAM
            "target_vram_gb": 4.0,
            "usable_vram_gb": 2.2,  # 4.0 - 1.0 - 0.6 - 0.2
            "max_layers_per_stage": 22,  # 2.2 / 0.097 (int4)
            "weight_precision": "int4"
        },
        "medium_gpu": {  # 8GB VRAM
            "target_vram_gb": 8.0,
            "usable_vram_gb": 5.5,  # 8.0 - 1.5 - 0.8 - 0.2
            "max_layers_per_stage": 28,  # 5.5 / 0.193 (int8)
            "weight_precision": "int8"
        },
        "large_gpu": {  # 16GB VRAM
            "target_vram_gb": 16.0,
            "usable_vram_gb": 12.0,  # 16.0 - 2.0 - 1.5 - 0.5
            "max_layers_per_stage": 36,  # All layers fit
            "weight_precision": "fp16"
        }
    }
}

# Distribution strategies for 36-layer model
DISTRIBUTION_STRATEGIES = {
    "single": {  # 1 GPU - all 36 layers
        "num_stages": 1,
        "layers_per_stage": [36],
        "min_vram_gb": 16.0,
        "precision": "fp16"
    },
    "dual": {  # 2 GPUs - 18 layers each
        "num_stages": 2,
        "layers_per_stage": [18, 18],
        "min_vram_gb": 8.0,
        "precision": "int8"
    },
    "triple": {  # 3 GPUs - 12 layers each
        "num_stages": 3,
        "layers_per_stage": [12, 12, 12],
        "min_vram_gb": 4.0,
        "precision": "int4"
    },
    "quad": {  # 4 GPUs - 9 layers each
        "num_stages": 4,
        "layers_per_stage": [9, 9, 9, 9],
        "min_vram_gb": 3.0,
        "precision": "int4"
    },
    "hexa": {  # 6 GPUs - 6 layers each
        "num_stages": 6,
        "layers_per_stage": [6, 6, 6, 6, 6, 6],
        "min_vram_gb": 2.0,
        "precision": "int4"
    },
    "per_layer": {  # 36 GPUs - 1 layer each
        "num_stages": 36,
        "layers_per_stage": [1] * 36,
        "min_vram_gb": 1.0,
        "precision": "int4"
    }
}

# Compute requirements
COMPUTE = {
    "min_gpu_memory": 4,  # GB minimum with int4
    "recommended_gpu_memory": 16,  # GB for full model
    "optimal_batch_size": {
        "fp16": 4,
        "int8": 8,
        "int4": 16
    }
}

# Training configuration
TRAINING = {
    "base_learning_rate": 1e-5,
    "warmup_steps": 1000,
    "gradient_accumulation_steps": 4,
    "mixed_precision": True,
    "lora_rank": 64,
    "lora_alpha": 128,
}

# Inference configuration  
INFERENCE = {
    "temperature": 0.6,  # For thinking mode
    "top_p": 0.95,
    "top_k": 20,
    "repetition_penalty": 1.1,
    "max_new_tokens": 32768,
    "stream": True,
    "use_cache": True,
    "enable_thinking": True
}

# Blockchain configuration
BLOCKCHAIN = {
    "chain_id": "B",  # Parameter chain
    "block_type": "layer",  # Store by layer
    "upload_batch_size": 4,  # Layers to upload in parallel
    "compression": "pickle",
    "max_block_size": 500 * 1024 * 1024,  # 500MB max per block (per layer)
}

# Helper functions
def get_memory_requirement(precision="fp16", num_layers=36):
    """Calculate memory requirement in GB for given layers."""
    layer_mem = LAYER_MEMORY[precision]["per_layer"] * num_layers
    embed_mem = LAYER_MEMORY[precision]["embedding"]
    lm_head_mem = LAYER_MEMORY[precision]["lm_head"]
    return layer_mem + embed_mem + lm_head_mem

def get_kv_cache_size(batch_size=1, seq_len=2048, num_layers=36):
    """Calculate KV cache size in GB."""
    # 2 (K+V) * batch * seq_len * num_layers * hidden_size * num_kv_heads / num_heads * 2 bytes
    kv_bytes = 2 * batch_size * seq_len * num_layers * LAYERS["hidden_size"] * LAYERS["num_kv_heads"] / LAYERS["num_attention_heads"] * 2
    return kv_bytes / (1024**3)

def calculate_partition_plan(target_vram_gb=4.0, precision="int4"):
    """Calculate how to partition model across GPUs."""
    profile = PARTITION["profiles"].get("small_gpu")
    if target_vram_gb >= 8:
        profile = PARTITION["profiles"]["medium_gpu"]
    if target_vram_gb >= 16:
        profile = PARTITION["profiles"]["large_gpu"]
    
    usable_vram = profile["usable_vram_gb"]
    layer_size = LAYER_MEMORY[precision]["per_layer"]
    embed_size = LAYER_MEMORY[precision]["embedding"]
    
    # Calculate layers per stage
    max_layers = int(usable_vram / layer_size)
    max_layers = min(max_layers, 36)
    
    # Create partition plan
    num_stages = (36 + max_layers - 1) // max_layers
    layers_per_stage = []
    remaining = 36
    for _ in range(num_stages):
        chunk = min(max_layers, remaining)
        layers_per_stage.append(chunk)
        remaining -= chunk
    
    return {
        "num_stages": num_stages,
        "layers_per_stage": layers_per_stage,
        "precision": precision,
        "memory_per_stage": [n * layer_size for n in layers_per_stage],
        "total_memory": get_memory_requirement(precision)
    }

def get_layer_range(stage_idx, layers_per_stage):
    """Get the layer range for a given stage."""
    start = sum(layers_per_stage[:stage_idx])
    end = start + layers_per_stage[stage_idx]
    return list(range(start, end))

def get_model_config():
    """Get complete model configuration as dict."""
    return {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "architecture": ARCHITECTURE,
        "layers": LAYERS,
        "context": CONTEXT,
        "precision": PRECISION,
        "layer_memory": LAYER_MEMORY,
        "partition": PARTITION,
        "distribution": DISTRIBUTION_STRATEGIES,
        "compute": COMPUTE,
        "training": TRAINING,
        "inference": INFERENCE,
        "blockchain": BLOCKCHAIN,
    }

# Validation
def validate_config():
    """Validate configuration consistency."""
    checks = []
    
    # Check memory calculations
    total_mem_fp16 = get_memory_requirement("fp16")
    expected = 8.2 * 2  # 8.2B params * 2 bytes
    if abs(total_mem_fp16 - expected) > 1:
        checks.append(f"Memory calculation mismatch: {total_mem_fp16:.1f}GB vs expected {expected:.1f}GB")
    
    # Check partition feasibility
    for strategy_name, strategy in DISTRIBUTION_STRATEGIES.items():
        total_layers = sum(strategy["layers_per_stage"])
        if total_layers != 36:
            checks.append(f"Strategy {strategy_name} has {total_layers} layers, expected 36")
    
    return checks

# Run validation on import
_validation_errors = validate_config()
if _validation_errors:
    print(f"⚠️ Model profile validation warnings: {_validation_errors}")