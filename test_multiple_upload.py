#!/usr/bin/env python
"""Test uploading multiple experts in sequence."""

import torch
import io
from pathlib import Path
from backend.core.chain import Chain

def main():
    print("🚀 Testing Multiple Expert Upload")
    
    # Setup
    root_dir = Path("./data")
    param_chain = Chain(root_dir, "B")
    
    # Load model
    print("1. Loading model...")
    model_path = "test_data/microsoft_DialoGPT-small.pt"
    state_dict = torch.load(model_path, map_location='cpu')
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    print(f"✓ Model loaded with {len(state_dict)} keys")
    
    # Create experts
    print("2. Creating experts...")
    experts = {}
    all_keys = list(state_dict.keys())
    num_experts = 24  # Create many small experts
    expert_size = len(all_keys) // num_experts
    
    for i in range(num_experts):
        expert_name = f"layer{i//2}.expert{i%2}"
        start_idx = i * expert_size
        end_idx = start_idx + expert_size if i < num_experts-1 else len(all_keys)
        
        expert_tensors = {}
        for key in all_keys[start_idx:end_idx]:
            expert_tensors[key] = state_dict[key]
        
        if expert_tensors:
            experts[expert_name] = expert_tensors
    
    print(f"✓ Created {len(experts)} experts")
    
    # Upload experts one by one
    print("3. Uploading experts...")
    for i, (expert_name, tensors) in enumerate(experts.items()):
        print(f"   [{i+1}/{len(experts)}] Uploading {expert_name}...")
        
        # Serialize
        buffer = io.BytesIO()
        torch.save(tensors, buffer)
        tensor_bytes = buffer.getvalue()
        
        try:
            block = param_chain.add_block(
                payload=tensor_bytes,
                block_type='expert',
                expert_name=expert_name,
                layer_id=expert_name.split('.')[0],
                depends_on=[],
                miner_pub=f'alice_{i}'
            )
            print(f"   ✓ {expert_name}: {block.compute_hash()[:16]}...")
        except Exception as e:
            print(f"   ❌ Failed {expert_name}: {e}")
            break
    
    # Final check
    all_blocks = list(param_chain.storage.iter_blocks())
    expert_blocks = [b for b in all_blocks if getattr(b.header, 'block_type', None) == 'expert']
    print(f"\n✅ Uploaded {len(expert_blocks)} expert blocks!")

if __name__ == "__main__":
    main()