#!/usr/bin/env python
"""Direct MoE expert upload bypassing API server."""

import torch
import io
from pathlib import Path
from backend.core.chain import Chain

def extract_simple_experts(model_path: str, num_experts: int = 4):
    """Extract a few experts from model for testing."""
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
    except Exception:
        print("❌ Failed to load model")
        return {}
    
    experts = {}
    layer_keys = list(state_dict.keys())[:10]  # Take first 10 keys
    
    # Group keys into experts
    expert_size = len(layer_keys) // num_experts
    for i in range(num_experts):
        expert_name = f"layer{i//2}.expert{i%2}"
        start_idx = i * expert_size
        end_idx = start_idx + expert_size if i < num_experts-1 else len(layer_keys)
        
        expert_tensors = {}
        for key in layer_keys[start_idx:end_idx]:
            expert_tensors[key] = state_dict[key]
        
        if expert_tensors:  # Only add non-empty experts
            experts[expert_name] = expert_tensors
    
    return experts

def main():
    print("🚀 Direct MoE Expert Upload")
    
    # Setup chains
    root_dir = Path("./data")
    param_chain = Chain(root_dir, "B")
    
    # Extract experts
    model_path = "test_data/microsoft_DialoGPT-small.pt"
    print(f"📥 Extracting experts from {model_path}...")
    experts = extract_simple_experts(model_path, num_experts=4)
    
    if not experts:
        print("❌ No experts extracted")
        return
    
    print(f"✓ Extracted {len(experts)} experts")
    
    # Upload each expert
    uploaded = 0
    for expert_name, tensors in experts.items():
        print(f"📤 Uploading {expert_name}...")
        
        # Serialize tensors
        buffer = io.BytesIO()
        torch.save(tensors, buffer)
        tensor_bytes = buffer.getvalue()
        
        try:
            block = param_chain.add_block(
                payload=tensor_bytes,
                block_type='expert',
                expert_name=expert_name,
                layer_id=expert_name.split('.')[0],
                depends_on=[],  # No dependencies
                miner_pub=f'alice_pub_{uploaded}'
            )
            print(f"✓ {expert_name}: {block.compute_hash()[:16]}...")
            uploaded += 1
        except Exception as e:
            print(f"❌ Failed to upload {expert_name}: {e}")
    
    print(f"\n🎉 Successfully uploaded {uploaded}/{len(experts)} experts!")
    
    # Verify chain state
    all_blocks = list(param_chain.storage.iter_blocks())
    print(f"📊 Parameter chain now has {len(all_blocks)} blocks")
    
    # Show expert blocks
    expert_blocks = [b for b in all_blocks if getattr(b.header, 'block_type', None) == 'expert']
    print(f"🧠 Expert blocks: {len(expert_blocks)}")
    for block in expert_blocks:
        print(f"  - {getattr(block.header, 'expert_name', 'unknown')}")

if __name__ == "__main__":
    main()