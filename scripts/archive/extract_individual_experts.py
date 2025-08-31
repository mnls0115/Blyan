#!/usr/bin/env python3
"""Extract individual experts from MoE model and create separate blocks."""

import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.chain import Chain
from backend.model.arch import state_dict_to_bytes
from transformers import AutoModelForCausalLM

def extract_experts_from_moe(model_path: str):
    """Extract individual experts from MoE model."""
    print(f"🔍 Loading MoE model from {model_path}")
    
    try:
        # Load the MoE model
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        state_dict = model.state_dict()
        
        print(f"📊 Model loaded with {len(state_dict)} parameters")
        
        # Group parameters by expert
        experts = {}
        shared_params = {}
        
        for param_name, param_tensor in state_dict.items():
            print(f"   Analyzing: {param_name} - {param_tensor.shape}")
            
            # Check if this is an expert-specific parameter
            if "block_sparse_moe.experts" in param_name:
                # Extract expert number: model.layers.0.block_sparse_moe.experts.2.w1.weight
                parts = param_name.split(".")
                layer_idx = None
                expert_idx = None
                
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        layer_idx = parts[i + 1]
                    elif part == "experts" and i + 1 < len(parts):
                        expert_idx = parts[i + 1]
                        
                if layer_idx is not None and expert_idx is not None:
                    expert_key = f"layer{layer_idx}.expert{expert_idx}"
                    if expert_key not in experts:
                        experts[expert_key] = {}
                    
                    # Store the parameter with a simplified name
                    param_key = param_name.split("experts.")[1].split(".", 1)[1]  # Remove expert number
                    experts[expert_key][param_key] = param_tensor
                    print(f"     → Added to {expert_key}: {param_key}")
            
            # Check for router/gate parameters
            elif "block_sparse_moe.gate" in param_name:
                parts = param_name.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        layer_idx = parts[i + 1]
                        router_key = f"layer{layer_idx}.router"
                        if router_key not in experts:
                            experts[router_key] = {}
                        experts[router_key]["gate.weight"] = param_tensor
                        print(f"     → Added router: {router_key}")
                        break
            
            # Shared parameters (embeddings, layer norms, etc.)
            else:
                shared_params[param_name] = param_tensor
        
        print(f"\n📈 Extraction Results:")
        print(f"   • Found {len(experts)} individual experts/routers")
        print(f"   • Found {len(shared_params)} shared parameters")
        
        for expert_name in sorted(experts.keys()):
            param_count = len(experts[expert_name])
            total_params = sum(p.numel() for p in experts[expert_name].values())
            print(f"   • {expert_name}: {param_count} tensors, {total_params:,} parameters")
        
        return experts, shared_params
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return {}, {}

def create_expert_blocks(experts: dict):
    """Create blockchain blocks for each expert."""
    root_dir = Path("./data")
    param_chain = Chain(root_dir, "B")
    
    print(f"\n🔗 Creating blockchain blocks for {len(experts)} experts...")
    
    for expert_name, expert_params in experts.items():
        if not expert_params:
            print(f"⚠️  Skipping {expert_name} - no parameters")
            continue
            
        try:
            # Convert tensors to bytes
            print(f"📦 Creating block for {expert_name}...")
            payload = state_dict_to_bytes(expert_params)
            
            # Determine block type and layer info
            if "router" in expert_name:
                block_type = "router" 
                layer_id = expert_name.split(".")[0]  # layer0, layer1, etc.
                expert_id = None
            else:
                block_type = "expert"
                parts = expert_name.split(".")
                layer_id = parts[0]  # layer0, layer1, etc.  
                expert_id = expert_name
            
            # Add block to chain
            param_chain.add_block(
                payload,
                block_type=block_type,
                expert_name=expert_id,
                layer_id=layer_id
            )
            
            print(f"✅ Created {block_type} block: {expert_name}")
            print(f"   Size: {len(payload):,} bytes")
            print(f"   Parameters: {len(expert_params)}")
            
        except Exception as e:
            print(f"❌ Failed to create block for {expert_name}: {e}")
    
    print(f"\n🎉 Successfully created blocks for MoE experts!")

def main():
    """Main function."""
    model_path = "./models/Qwen/Qwen3-8B-FP8"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("🚀 Extracting individual experts from MoE model...")
    print("=" * 60)
    
    # Extract experts from model
    experts, _ = extract_experts_from_moe(model_path)
    
    if not experts:
        print("❌ No experts found in model")
        return
    
    # Create blockchain blocks
    create_expert_blocks(experts)
    
    print("\n" + "=" * 60)
    print("✅ Expert extraction complete!")
    print(f"   • Extracted {len(experts)} experts/routers")
    print(f"   • Created blockchain blocks for each expert")
    print(f"   • Ready for diverse MoE routing!")

if __name__ == "__main__":
    main()