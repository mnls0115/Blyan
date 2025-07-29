#!/usr/bin/env python
"""Upload just a few experts to test the flow."""

import torch
import io
from pathlib import Path
from backend.core.chain import Chain

def main():
    print("🚀 Uploading Representative MoE Experts")
    
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
    
    # Create a few representative experts
    print("2. Creating representative experts...")
    all_keys = list(state_dict.keys())
    experts = {}
    
    # Create 3 experts with different layer components
    expert_configs = [
        ("layer0.expert0", all_keys[0:15]),    # First 15 keys
        ("layer1.expert0", all_keys[15:30]),   # Next 15 keys  
        ("layer2.expert0", all_keys[30:45]),   # Next 15 keys
    ]
    
    for expert_name, keys in expert_configs:
        expert_tensors = {}
        for key in keys:
            if key in state_dict:
                expert_tensors[key] = state_dict[key]
        if expert_tensors:
            experts[expert_name] = expert_tensors
    
    print(f"✓ Created {len(experts)} representative experts")
    
    # Upload experts
    print("3. Uploading experts...")
    for i, (expert_name, tensors) in enumerate(experts.items()):
        buffer = io.BytesIO()
        torch.save(tensors, buffer)
        tensor_bytes = buffer.getvalue()
        size_mb = len(tensor_bytes) / (1024*1024)
        
        print(f"   [{i+1}] {expert_name} ({size_mb:.1f}MB, {len(tensors)} tensors)...")
        
        try:
            block = param_chain.add_block(
                payload=tensor_bytes,
                block_type='expert',
                expert_name=expert_name,
                layer_id=expert_name.split('.')[0],
                depends_on=[],
                miner_pub=f'alice_{i}'
            )
            print(f"       ✓ Block: {block.compute_hash()[:16]}...")
        except Exception as e:
            print(f"       ❌ Failed: {e}")
            break
    
    # Verify results
    all_blocks = list(param_chain.storage.iter_blocks())
    expert_blocks = [b for b in all_blocks if getattr(b.header, 'block_type', None) == 'expert']
    print(f"\n✅ Successfully uploaded {len(expert_blocks)} expert blocks!")
    print(f"📊 Parameter chain now has {len(all_blocks)} total blocks")
    
    # List expert blocks
    print("\n🧠 Expert blocks:")
    for block in expert_blocks:
        expert_name = getattr(block.header, 'expert_name', 'unknown')
        size_mb = len(block.payload) / (1024*1024)
        print(f"  - {expert_name}: {size_mb:.1f}MB")

if __name__ == "__main__":
    main()