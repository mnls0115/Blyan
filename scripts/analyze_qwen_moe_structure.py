#!/usr/bin/env python3
"""
Analyze the structure of Qwen3-30B MoE model to understand expert organization.
This helps us properly extract all 128 experts per layer.
"""

import sys
import torch
from pathlib import Path
from typing import Dict, Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_profile import MODEL_ID, LAYERS, MOE

def analyze_model_structure():
    """Analyze the MoE model structure without loading full weights."""
    print(f"Analyzing model: {MODEL_ID}")
    print("=" * 60)
    
    try:
        from transformers import AutoConfig
        
        # Load just the config to understand structure
        config = AutoConfig.from_pretrained(MODEL_ID)
        
        print("Model Configuration:")
        print(f"  Model type: {config.model_type}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Num layers: {config.num_hidden_layers}")
        
        # Check for MoE specific attributes
        moe_attrs = [
            'num_experts', 'num_experts_per_tok', 'expert_interval',
            'num_local_experts', 'router_aux_loss_coef', 'moe_intermediate_size',
            'shared_expert_intermediate_size', 'norm_topk_prob'
        ]
        
        print("\nMoE Configuration:")
        for attr in moe_attrs:
            if hasattr(config, attr):
                print(f"  {attr}: {getattr(config, attr)}")
        
        # Expected structure for Qwen MoE
        print("\n" + "=" * 60)
        print("Expected Layer Structure for Qwen3-30B:")
        print("  model.layers[0..47]")
        print("    ├── self_attn (attention mechanism)")
        print("    ├── mlp (MoE block)")
        print("    │   ├── gate (router)")
        print("    │   ├── experts[0..127] (individual experts)")
        print("    │   │   ├── gate_proj")
        print("    │   │   ├── up_proj")
        print("    │   │   └── down_proj")
        print("    │   └── shared_expert (if present)")
        print("    └── layer norm layers")
        
        return config
        
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def create_expert_extraction_code():
    """Generate the code needed to properly extract experts."""
    
    code = '''
def extract_all_experts_from_layer(layer, layer_idx):
    """Extract all individual experts from a MoE layer."""
    experts_data = {}
    
    # Check if layer has MoE structure
    if not hasattr(layer, 'mlp'):
        return experts_data
    
    mlp = layer.mlp
    
    # Extract gate/router weights
    if hasattr(mlp, 'gate'):
        router_name = f"layer{layer_idx}.router"
        experts_data[router_name] = {
            'weight': mlp.gate.weight.detach().cpu().contiguous()
        }
    
    # Extract individual experts
    if hasattr(mlp, 'experts'):
        num_experts = len(mlp.experts) if hasattr(mlp.experts, '__len__') else 0
        
        for expert_idx in range(num_experts):
            expert = mlp.experts[expert_idx]
            expert_name = f"layer{layer_idx}.expert{expert_idx}"
            
            # Extract expert weights (gate_proj, up_proj, down_proj)
            expert_state = {}
            if hasattr(expert, 'gate_proj'):
                expert_state['gate_proj.weight'] = expert.gate_proj.weight.detach().cpu().contiguous()
            if hasattr(expert, 'up_proj'):
                expert_state['up_proj.weight'] = expert.up_proj.weight.detach().cpu().contiguous()
            if hasattr(expert, 'down_proj'):
                expert_state['down_proj.weight'] = expert.down_proj.weight.detach().cpu().contiguous()
            
            if expert_state:
                experts_data[expert_name] = expert_state
    
    # Extract shared expert if present
    if hasattr(mlp, 'shared_expert'):
        shared_name = f"layer{layer_idx}.shared_expert"
        shared_state = {}
        
        expert = mlp.shared_expert
        if hasattr(expert, 'gate_proj'):
            shared_state['gate_proj.weight'] = expert.gate_proj.weight.detach().cpu().contiguous()
        if hasattr(expert, 'up_proj'):
            shared_state['up_proj.weight'] = expert.up_proj.weight.detach().cpu().contiguous()
        if hasattr(expert, 'down_proj'):
            shared_state['down_proj.weight'] = expert.down_proj.weight.detach().cpu().contiguous()
        
        if shared_state:
            experts_data[shared_name] = shared_state
    
    return experts_data
'''
    
    print("\n" + "=" * 60)
    print("Expert Extraction Code:")
    print(code)
    
    return code

def estimate_storage_requirements():
    """Estimate blockchain storage for all experts."""
    
    print("\n" + "=" * 60)
    print("Storage Estimates for Full Model:")
    
    # Rough estimates based on Qwen3-30B architecture
    # Each expert has gate_proj, up_proj, down_proj
    hidden_size = 4096  # Typical for 30B model
    intermediate_size = 11008  # Typical MoE intermediate size
    
    # Weight sizes per expert
    gate_proj_params = hidden_size * intermediate_size
    up_proj_params = hidden_size * intermediate_size  
    down_proj_params = intermediate_size * hidden_size
    params_per_expert = gate_proj_params + up_proj_params + down_proj_params
    
    # Total experts
    total_experts = LAYERS["num_layers"] * MOE["num_experts"]
    total_params = total_experts * params_per_expert
    
    # Storage in different precisions
    storage_fp32 = (total_params * 4) / (1024**3)  # GB
    storage_fp16 = (total_params * 2) / (1024**3)  # GB
    storage_fp8 = (total_params * 1) / (1024**3)   # GB
    storage_int8 = (total_params * 1) / (1024**3)  # GB
    
    print(f"  Experts: {total_experts:,} ({LAYERS['num_layers']} layers × {MOE['num_experts']} experts)")
    print(f"  Parameters per expert: ~{params_per_expert/1e6:.1f}M")
    print(f"  Total expert parameters: ~{total_params/1e9:.1f}B")
    print(f"\nStorage Requirements:")
    print(f"  FP32: {storage_fp32:.1f} GB")
    print(f"  FP16: {storage_fp16:.1f} GB")
    print(f"  FP8:  {storage_fp8:.1f} GB")
    print(f"  INT8: {storage_int8:.1f} GB")
    
    print(f"\nBlockchain Blocks:")
    print(f"  Blocks needed: {total_experts:,} expert blocks + {LAYERS['num_layers']} router blocks")
    print(f"  Average block size (FP8): {storage_fp8*1024/total_experts:.1f} MB per expert")

if __name__ == "__main__":
    print("Qwen3-30B MoE Structure Analysis")
    print("=" * 60)
    
    # Analyze model config
    config = analyze_model_structure()
    
    # Generate extraction code
    extraction_code = create_expert_extraction_code()
    
    # Estimate storage
    estimate_storage_requirements()
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("\nNext steps:")
    print("1. Update run_gpu_node.py to use the extraction code above")
    print("2. Modify upload logic to handle all experts")
    print("3. Implement batch uploading for efficiency")