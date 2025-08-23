#!/usr/bin/env python3
"""Verify which experts are actually in the blockchain."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from backend.core.chain import Chain
from collections import defaultdict

def verify_experts():
    """Check which experts are in the blockchain."""
    data_dir = Path("./data")
    
    # Load parameter chain
    param_chain = Chain(data_dir, "B")
    blocks = param_chain.get_all_blocks()
    
    print(f"Total blocks in chain B: {len(blocks)}")
    
    # Count experts by layer
    experts_by_layer = defaultdict(set)
    total_experts = 0
    
    for i, block in enumerate(blocks):
        if hasattr(block.header, 'block_type') and block.header.block_type == 'expert':
            expert_name = block.header.expert_name
            if expert_name and 'layer' in expert_name:
                parts = expert_name.split('.')
                if len(parts) == 2:
                    layer_part = parts[0]  # e.g., "layer0"
                    expert_part = parts[1]  # e.g., "expert0"
                    experts_by_layer[layer_part].add(expert_part)
                    total_experts += 1
        
        # Show progress
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} blocks...")
    
    print(f"\nTotal expert blocks found: {total_experts}")
    print(f"Layers with experts: {len(experts_by_layer)}")
    
    # Show distribution
    print("\nExperts per layer:")
    for layer in sorted(experts_by_layer.keys(), key=lambda x: int(x.replace('layer', ''))):
        num_experts = len(experts_by_layer[layer])
        print(f"  {layer}: {num_experts} experts")
        if num_experts < 128:
            print(f"    WARNING: Expected 128 experts, found {num_experts}")
    
    # Check for missing layers
    print("\nMissing layers:")
    num_layers = LAYERS["num_layers"] if PROFILE_AVAILABLE else 48
    for i in range(num_layers):
        layer_name = f"layer{i}"
        if layer_name not in experts_by_layer:
            print(f"  {layer_name}: MISSING!")
    
    return experts_by_layer

if __name__ == "__main__":
    verify_experts()