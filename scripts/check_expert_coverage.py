#!/usr/bin/env python3
"""Quick check of expert coverage in blockchain."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from backend.core.chain import Chain

try:
    from config.model_profile import LAYERS, MOE, get_total_experts
    PROFILE_AVAILABLE = True
except ImportError:
    PROFILE_AVAILABLE = False
    # Fallback values
    LAYERS = {"num_layers": 48}
    MOE = {"num_experts": 128}
    def get_total_experts():
        return 6144

def check_coverage():
    """Quick check of what experts we have."""
    data_dir = Path("./data")
    param_chain = Chain(data_dir, "B")
    
    print(f"Total blocks: {len(param_chain.get_all_blocks())}")
    
    # Count expert blocks
    expert_count = 0
    expert_names = set()
    layers = {}
    
    for block in param_chain.get_all_blocks():
        if hasattr(block.header, 'block_type') and block.header.block_type == 'expert':
            expert_count += 1
            if block.header.expert_name:
                expert_names.add(block.header.expert_name)
                # Extract layer
                if 'layer' in block.header.expert_name:
                    layer = block.header.expert_name.split('.')[0]
                    if layer not in layers:
                        layers[layer] = set()
                    layers[layer].add(block.header.expert_name)
    
    print(f"Expert blocks: {expert_count}")
    print(f"Unique experts: {len(expert_names)}")
    print(f"Layers with experts: {len(layers)}")
    
    # Check coverage per layer
    print("\nExperts per layer:")
    missing_layers = []
    incomplete_layers = []
    
    num_layers = LAYERS["num_layers"] if PROFILE_AVAILABLE else 48
    num_experts = MOE["num_experts"] if PROFILE_AVAILABLE else 128
    
    for i in range(num_layers):
        layer_name = f"layer{i}"
        if layer_name in layers:
            count = len(layers[layer_name])
            if count < num_experts:
                print(f"  {layer_name}: {count}/{num_experts} experts ⚠️")
                incomplete_layers.append(layer_name)
            else:
                print(f"  {layer_name}: {count}/{num_experts} experts ✅")
        else:
            print(f"  {layer_name}: 0/{num_experts} experts ❌")
            missing_layers.append(layer_name)
    
    print(f"\nSummary:")
    print(f"  Complete layers ({num_experts} experts): {num_layers - len(missing_layers) - len(incomplete_layers)}")
    print(f"  Incomplete layers: {len(incomplete_layers)}")
    print(f"  Missing layers: {len(missing_layers)}")
    
    if missing_layers:
        print(f"\nMissing layers: {missing_layers[:10]}...")
    if incomplete_layers:
        print(f"Incomplete layers: {incomplete_layers[:10]}...")

if __name__ == "__main__":
    check_coverage()