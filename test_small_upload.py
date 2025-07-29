#!/usr/bin/env python
"""Test uploading many very small experts."""

import torch
import io
from pathlib import Path
from backend.core.chain import Chain

def main():
    print("🚀 Testing Small Expert Upload")
    
    # Setup
    root_dir = Path("./data")
    param_chain = Chain(root_dir, "B")
    
    print("1. Creating tiny experts...")
    experts = {}
    for i in range(10):  # 10 small experts
        expert_name = f"tiny.expert{i}"
        # Very small tensors
        tensors = {
            'weight': torch.randn(2, 2),
            'bias': torch.randn(2)
        }
        experts[expert_name] = tensors
    
    print(f"✓ Created {len(experts)} tiny experts")
    
    # Upload experts
    print("2. Uploading experts...")
    for i, (expert_name, tensors) in enumerate(experts.items()):
        print(f"   [{i+1}] {expert_name}...")
        
        # Serialize
        buffer = io.BytesIO()
        torch.save(tensors, buffer)
        tensor_bytes = buffer.getvalue()
        print(f"       Size: {len(tensor_bytes)} bytes")
        
        try:
            block = param_chain.add_block(
                payload=tensor_bytes,
                block_type='expert',
                expert_name=expert_name,
                layer_id='tiny',
                depends_on=[],
                miner_pub=f'alice_{i}'
            )
            print(f"       ✓ {block.compute_hash()[:8]}...")
        except Exception as e:
            print(f"       ❌ Failed: {e}")
            break
    
    # Final check
    all_blocks = list(param_chain.storage.iter_blocks())
    print(f"\n✅ Chain has {len(all_blocks)} total blocks")

if __name__ == "__main__":
    main()