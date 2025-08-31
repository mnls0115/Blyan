"""
Model Profile Configuration - Scalable Dense Models
====================================================
Dynamic configuration system supporting models from 8B to 70B+.
Automatically derives partition plans from model config and GPU VRAM.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# Environment variable overrides
MODEL_SIZE = os.getenv("MODEL_SIZE", "8B")  # 8B, 32B, 70B
WEIGHT_PRECISION = os.getenv("WEIGHT_PRECISION", "bf16")  # bf16 default per CLAUDE.md
KV_CACHE_BUDGET_GB = float(os.getenv("KV_CACHE_BUDGET_GB", "0.6"))
MICROBATCH_SIZE = int(os.getenv("MICROBATCH_SIZE", "1"))
JOB_CAPACITY = int(os.getenv("JOB_CAPACITY", "4"))

# Model configurations for different sizes
MODEL_CONFIGS = {
    "8B": {
        "model_id": "Qwen/Qwen3-8B",
        "model_name": "Qwen3-8B",
        "architecture": {
            "type": "dense",
            "family": "Qwen3",
            "total_params_b": 8.2,
            "non_embedding_params_b": 6.95,
            "embedding_params_b": 1.25
        },
        "layers": {
            "num_hidden_layers": 36,  # Standard HF config name
            "num_attention_heads": 32,
            "num_key_value_heads": 8,  # GQA
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "vocab_size": 152064
        }
    },
    "32B": {
        "model_id": "Qwen/Qwen2.5-32B",
        "model_name": "Qwen2.5-32B",
        "architecture": {
            "type": "dense",
            "family": "Qwen2.5",
            "total_params_b": 32.5,
            "non_embedding_params_b": 31.0,
            "embedding_params_b": 1.5
        },
        "layers": {
            "num_hidden_layers": 64,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "hidden_size": 5120,
            "intermediate_size": 27648,
            "vocab_size": 152064
        }
    },
    "70B": {
        "model_id": "Qwen/Qwen2.5-72B",
        "model_name": "Qwen2.5-72B",
        "architecture": {
            "type": "dense",
            "family": "Qwen2.5",
            "total_params_b": 72.7,
            "non_embedding_params_b": 70.0,
            "embedding_params_b": 2.7
        },
        "layers": {
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "hidden_size": 8192,
            "intermediate_size": 29568,
            "vocab_size": 152064
        }
    }
}

# Select active model configuration
ACTIVE_CONFIG = MODEL_CONFIGS.get(MODEL_SIZE, MODEL_CONFIGS["8B"])

# Export top-level config for compatibility
MODEL_ID = ACTIVE_CONFIG["model_id"]
MODEL_NAME = ACTIVE_CONFIG["model_name"]
MODEL_VERSION = "latest"
ARCHITECTURE = ACTIVE_CONFIG["architecture"]
LAYERS = ACTIVE_CONFIG["layers"]

# Add computed fields for compatibility
LAYERS["num_layers"] = LAYERS["num_hidden_layers"]  # Alias for backward compat
LAYERS["num_kv_heads"] = LAYERS["num_key_value_heads"]  # Alias
LAYERS["head_dim"] = LAYERS["hidden_size"] // LAYERS["num_attention_heads"]

# Context configuration
CONTEXT = {
    "max_length": 32768,  # 32K native context
    "extended_length": 131072,  # 128K with YaRN
    "chunk_size": 2048,
    "sliding_window": None
}

# Precision configurations with BF16 as default
PRECISION = {
    "default": WEIGHT_PRECISION,
    "supported": ["fp32", "bf16", "fp16", "int8", "int4"],
    "memory_per_billion": {
        "fp32": 4.0,  # GB per billion params
        "bf16": 2.0,  # BF16 for numerical consistency
        "fp16": 2.0,
        "int8": 1.0,
        "int4": 0.5
    }
}

def calculate_layer_memory(precision: str = None) -> Dict[str, Dict[str, float]]:
    """
    Dynamically calculate per-layer memory requirements.
    
    Args:
        precision: Weight precision (uses default if None)
        
    Returns:
        Memory requirements by component
    """
    if precision is None:
        precision = WEIGHT_PRECISION
    
    bytes_per_param = PRECISION["memory_per_billion"][precision]
    arch = ACTIVE_CONFIG["architecture"]
    layers_config = ACTIVE_CONFIG["layers"]
    
    # Calculate per-layer memory
    per_layer_params_b = arch["non_embedding_params_b"] / layers_config["num_hidden_layers"]
    per_layer_gb = per_layer_params_b * bytes_per_param
    
    # Calculate embedding/lm_head memory
    embedding_gb = arch["embedding_params_b"] * bytes_per_param
    
    return {
        precision: {
            "per_layer": per_layer_gb,
            "embedding": embedding_gb / 2,  # Split between embed and lm_head
            "lm_head": embedding_gb / 2
        }
    }

# Generate layer memory for all precisions
LAYER_MEMORY = {}
for prec in PRECISION["supported"]:
    LAYER_MEMORY.update(calculate_layer_memory(prec))

# Partition configuration with environment overrides
PARTITION = {
    "default": {
        "target_vram_gb": float(os.getenv("TARGET_VRAM_GB", "4.0")),
        "reserved_headroom_gb": float(os.getenv("RESERVED_HEADROOM_GB", "1.0")),
        "kv_cache_budget_gb": KV_CACHE_BUDGET_GB,
        "runtime_buffer_gb": float(os.getenv("RUNTIME_BUFFER_GB", "0.2")),
        "weight_precision": WEIGHT_PRECISION,
        "activation_checkpointing": True,
        "microbatch_size": MICROBATCH_SIZE
    }
}

def calculate_kv_cache_size(
    batch_size: int = 1,
    seq_len: int = 2048,
    num_layers: Optional[int] = None,
    precision: str = "bf16"
) -> float:
    """
    Calculate KV cache size in GB.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_layers: Number of layers (uses model config if None)
        precision: Data type for KV cache
        
    Returns:
        KV cache size in GB
    """
    if num_layers is None:
        num_layers = LAYERS["num_hidden_layers"]
    
    bytes_per_element = 2 if precision in ["bf16", "fp16"] else 4
    
    # 2 (K+V) * batch * seq_len * num_layers * hidden_size * num_kv_heads / num_heads * bytes
    kv_elements = (
        2 * batch_size * seq_len * num_layers *
        LAYERS["hidden_size"] * LAYERS["num_key_value_heads"] / LAYERS["num_attention_heads"]
    )
    kv_bytes = kv_elements * bytes_per_element
    
    return kv_bytes / (1024**3)

def get_memory_requirement(
    precision: Optional[str] = None,
    num_layers: Optional[int] = None
) -> float:
    """
    Calculate total memory requirement.
    
    Args:
        precision: Weight precision (uses default if None)
        num_layers: Number of layers (uses all if None)
        
    Returns:
        Memory requirement in GB
    """
    if precision is None:
        precision = WEIGHT_PRECISION
    if num_layers is None:
        num_layers = LAYERS["num_hidden_layers"]
    
    layer_mem = LAYER_MEMORY[precision]["per_layer"] * num_layers
    embed_mem = LAYER_MEMORY[precision]["embedding"]
    lm_head_mem = LAYER_MEMORY[precision]["lm_head"]
    
    return layer_mem + embed_mem + lm_head_mem

def calculate_dynamic_partition_plan(
    target_vram_gb: Optional[float] = None,
    precision: Optional[str] = None,
    reserved_headroom_gb: Optional[float] = None,
    kv_cache_budget_gb: Optional[float] = None,
    runtime_buffer_gb: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate partition plan based on available VRAM and model config.
    
    Args:
        target_vram_gb: GPU VRAM (uses env/default if None)
        precision: Weight precision (uses env/default if None)
        reserved_headroom_gb: Reserved for CUDA context
        kv_cache_budget_gb: Reserved for KV cache
        runtime_buffer_gb: Reserved for temp buffers
        
    Returns:
        Partition plan with validation
    """
    # Use environment variables or defaults
    if target_vram_gb is None:
        target_vram_gb = PARTITION["default"]["target_vram_gb"]
    if precision is None:
        precision = WEIGHT_PRECISION
    if reserved_headroom_gb is None:
        reserved_headroom_gb = PARTITION["default"]["reserved_headroom_gb"]
    if kv_cache_budget_gb is None:
        kv_cache_budget_gb = KV_CACHE_BUDGET_GB
    if runtime_buffer_gb is None:
        runtime_buffer_gb = PARTITION["default"]["runtime_buffer_gb"]
    
    # Calculate usable VRAM
    usable_vram = target_vram_gb - reserved_headroom_gb - kv_cache_budget_gb - runtime_buffer_gb
    
    if usable_vram <= 0:
        return {
            "feasible": False,
            "error": f"No usable VRAM! Target: {target_vram_gb}GB, Reserved: {target_vram_gb - usable_vram}GB",
            "guidance": "Increase TARGET_VRAM_GB or reduce KV_CACHE_BUDGET_GB"
        }
    
    # Get layer and embedding sizes
    layer_size = LAYER_MEMORY[precision]["per_layer"]
    embed_size = LAYER_MEMORY[precision]["embedding"]
    lm_head_size = LAYER_MEMORY[precision]["lm_head"]
    total_layers = LAYERS["num_hidden_layers"]
    
    # Check if single embedding/lm_head fits
    if embed_size > usable_vram or lm_head_size > usable_vram:
        return {
            "feasible": False,
            "error": f"Embedding ({embed_size:.2f}GB) or LM head ({lm_head_size:.2f}GB) exceeds usable VRAM ({usable_vram:.2f}GB)",
            "guidance": f"Need at least {max(embed_size, lm_head_size):.2f}GB usable VRAM. Try: 1) Increase GPU memory, 2) Use lower precision (int8/int4), 3) Reduce reserved memory"
        }
    
    # Calculate maximum layers per stage
    # Account for embedding on first stage and lm_head on last stage
    max_layers_with_embed = int((usable_vram - embed_size) / layer_size)
    max_layers_with_lm = int((usable_vram - lm_head_size) / layer_size)
    max_layers_middle = int(usable_vram / layer_size)
    
    # Determine optimal partitioning strategy
    if max_layers_with_embed >= total_layers:
        # Everything fits on one GPU
        num_stages = 1
        layers_per_stage = [total_layers]
    else:
        # Need multiple stages
        # Try to balance stages while respecting constraints
        min_stages = 2  # At least 2 if doesn't fit on one
        max_stages = total_layers  # At most one layer per stage
        
        best_plan = None
        best_balance = float('inf')
        
        for n_stages in range(min_stages, min(max_stages + 1, 20)):  # Cap at 20 for performance
            # Distribute layers as evenly as possible
            base_layers = total_layers // n_stages
            extra_layers = total_layers % n_stages
            
            stage_layers = []
            for i in range(n_stages):
                n_layers = base_layers + (1 if i < extra_layers else 0)
                stage_layers.append(n_layers)
            
            # Validate this configuration
            valid = True
            stage_memories = []
            
            for i, n_layers in enumerate(stage_layers):
                stage_mem = n_layers * layer_size
                if i == 0:  # First stage has embedding
                    stage_mem += embed_size
                if i == n_stages - 1:  # Last stage has lm_head
                    stage_mem += lm_head_size
                
                stage_memories.append(stage_mem)
                
                if stage_mem > usable_vram:
                    valid = False
                    break
            
            if valid:
                # Calculate balance (variance in memory usage)
                avg_mem = sum(stage_memories) / len(stage_memories)
                balance = sum((m - avg_mem) ** 2 for m in stage_memories)
                
                if balance < best_balance:
                    best_balance = balance
                    best_plan = {
                        "num_stages": n_stages,
                        "layers_per_stage": stage_layers,
                        "memory_per_stage": stage_memories
                    }
        
        if best_plan:
            num_stages = best_plan["num_stages"]
            layers_per_stage = best_plan["layers_per_stage"]
        else:
            # No feasible plan found
            min_vram_needed = max(
                embed_size + layer_size,  # First stage
                lm_head_size + layer_size,  # Last stage
                layer_size  # Middle stages
            )
            return {
                "feasible": False,
                "error": f"Cannot partition {MODEL_NAME} ({total_layers} layers) with {target_vram_gb}GB VRAM",
                "guidance": f"Need at least {min_vram_needed:.2f}GB usable VRAM per GPU. Options: "
                           f"1) Use {total_layers} GPUs for single-layer stages, "
                           f"2) Increase VRAM to {min_vram_needed + reserved_headroom_gb + kv_cache_budget_gb:.1f}GB, "
                           f"3) Use lower precision (current: {precision})"
            }
    
    # Calculate detailed memory breakdown
    memory_breakdown = []
    for i in range(num_stages):
        stage_mem = layers_per_stage[i] * layer_size
        components = [f"{layers_per_stage[i]} layers"]
        
        if i == 0:
            stage_mem += embed_size
            components.insert(0, "Embedding")
        if i == num_stages - 1:
            stage_mem += lm_head_size
            components.append("LM Head")
        
        kv_cache = calculate_kv_cache_size(
            batch_size=MICROBATCH_SIZE,
            num_layers=layers_per_stage[i],
            precision=precision
        )
        
        memory_breakdown.append({
            "stage": i,
            "components": components,
            "weight_memory_gb": stage_mem,
            "kv_cache_gb": kv_cache,
            "total_gb": stage_mem + kv_cache + reserved_headroom_gb + runtime_buffer_gb,
            "headroom_gb": target_vram_gb - (stage_mem + kv_cache + reserved_headroom_gb + runtime_buffer_gb)
        })
    
    return {
        "feasible": True,
        "num_stages": num_stages,
        "layers_per_stage": layers_per_stage,
        "precision": precision,
        "total_model_memory_gb": get_memory_requirement(precision),
        "target_vram_gb": target_vram_gb,
        "usable_vram_gb": usable_vram,
        "memory_breakdown": memory_breakdown,
        "config": {
            "model": MODEL_NAME,
            "total_layers": total_layers,
            "microbatch_size": MICROBATCH_SIZE,
            "job_capacity": JOB_CAPACITY
        }
    }

def get_layer_range(stage_idx: int, layers_per_stage: List[int]) -> List[int]:
    """Get the layer indices for a given stage."""
    start = sum(layers_per_stage[:stage_idx])
    end = start + layers_per_stage[stage_idx]
    return list(range(start, end))

def validate_config() -> List[str]:
    """Validate configuration consistency."""
    checks = []
    
    # Validate active model config
    if MODEL_SIZE not in MODEL_CONFIGS:
        checks.append(f"Unknown MODEL_SIZE: {MODEL_SIZE}, using 8B")
    
    # Validate precision
    if WEIGHT_PRECISION not in PRECISION["supported"]:
        checks.append(f"Unsupported precision: {WEIGHT_PRECISION}")
    
    # Check if BF16 is being used (recommended)
    if WEIGHT_PRECISION != "bf16":
        logger.warning(f"Using {WEIGHT_PRECISION} instead of recommended BF16")
    
    # Validate partition feasibility
    plan = calculate_dynamic_partition_plan()
    if not plan["feasible"]:
        checks.append(f"Partition plan not feasible: {plan['error']}")
    
    return checks

def get_model_config() -> Dict[str, Any]:
    """Get complete model configuration as dict."""
    return {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "model_size": MODEL_SIZE,
        "architecture": ARCHITECTURE,
        "layers": LAYERS,
        "context": CONTEXT,
        "precision": PRECISION,
        "layer_memory": LAYER_MEMORY,
        "partition": PARTITION,
        "active_precision": WEIGHT_PRECISION,
        "kv_cache_budget_gb": KV_CACHE_BUDGET_GB,
        "microbatch_size": MICROBATCH_SIZE,
        "job_capacity": JOB_CAPACITY
    }

# Legacy distribution strategies (for compatibility)
DISTRIBUTION_STRATEGIES = {}

def generate_distribution_strategies():
    """Generate distribution strategies based on current model."""
    total_layers = LAYERS["num_hidden_layers"]
    strategies = {}
    
    # Common GPU configurations
    gpu_configs = [
        (1, "single", 16.0),
        (2, "dual", 8.0),
        (3, "triple", 6.0),
        (4, "quad", 4.0),
        (6, "hexa", 3.0),
        (8, "octa", 2.5),
        (12, "dodec", 2.0),
    ]
    
    for num_gpus, name, min_vram in gpu_configs:
        if total_layers % num_gpus == 0:
            layers_per_gpu = total_layers // num_gpus
            strategies[name] = {
                "num_stages": num_gpus,
                "layers_per_stage": [layers_per_gpu] * num_gpus,
                "min_vram_gb": min_vram,
                "precision": WEIGHT_PRECISION
            }
    
    # Add per-layer strategy
    strategies["per_layer"] = {
        "num_stages": total_layers,
        "layers_per_stage": [1] * total_layers,
        "min_vram_gb": 1.0,
        "precision": WEIGHT_PRECISION
    }
    
    return strategies

DISTRIBUTION_STRATEGIES = generate_distribution_strategies()

# Compute requirements
COMPUTE = {
    "min_gpu_memory": 4,  # GB minimum
    "recommended_gpu_memory": 16,  # GB for full model
    "optimal_batch_size": {
        "bf16": 4,
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
    "temperature": 0.6,
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
    "chain_id": "B",
    "block_type": "layer",
    "upload_batch_size": 4,
    "compression": "pickle",
    "max_block_size": 500 * 1024 * 1024,  # 500MB max per block
}

# Run validation on import
_validation_errors = validate_config()
if _validation_errors:
    logger.warning(f"Model profile validation warnings: {_validation_errors}")

# Log active configuration
logger.info(f"Active model: {MODEL_NAME} ({MODEL_SIZE}) with {WEIGHT_PRECISION} precision")
logger.info(f"Layers: {LAYERS['num_hidden_layers']}, Hidden size: {LAYERS['hidden_size']}")
logger.info(f"Environment overrides: WEIGHT_PRECISION={WEIGHT_PRECISION}, KV_CACHE_BUDGET_GB={KV_CACHE_BUDGET_GB}")