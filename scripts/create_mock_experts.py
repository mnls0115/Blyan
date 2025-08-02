#!/usr/bin/env python3
"""Create mock expert blocks for testing diverse expert routing."""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.chain import Chain
from backend.core.block import BlockHeader
from backend.model.arch import state_dict_to_bytes
import torch

def create_mock_expert_weights(expert_id: int) -> dict:
    """Create mock expert weights with slight variations."""
    # Create different weights for each expert to simulate specialization
    base_size = 1024
    hidden_size = 512
    
    weights = {
        f"expert_{expert_id}.w1.weight": torch.randn(hidden_size, base_size) * (0.8 + expert_id * 0.1),
        f"expert_{expert_id}.w2.weight": torch.randn(base_size, hidden_size) * (0.9 + expert_id * 0.05),
        f"expert_{expert_id}.w3.weight": torch.randn(hidden_size, base_size) * (1.0 + expert_id * 0.1),
        f"expert_{expert_id}.gate.weight": torch.randn(1, hidden_size) * (0.7 + expert_id * 0.15),
    }
    
    return weights

def main():
    """Create mock experts for testing."""
    root_dir = Path("./data")
    param_chain = Chain(root_dir, "B")
    
    expert_names = [
        "layer0.expert1",  # Reasoning expert
        "layer0.expert2",  # Creative expert  
        "layer1.expert0",  # Analysis expert
        "layer1.expert1",  # Language expert
    ]
    
    for i, expert_name in enumerate(expert_names):
        print(f"Creating mock expert: {expert_name}")
        
        # Create mock weights
        weights = create_mock_expert_weights(i + 1)
        payload = state_dict_to_bytes(weights)
        
        # Add to chain (Chain.add_block will create the header)
        layer_id, expert_id = expert_name.split(".")
        param_chain.add_block(
            payload, 
            block_type="expert",
            expert_name=expert_name,
            layer_id=layer_id
        )
        print(f"✅ Added {expert_name} to blockchain")
    
    print(f"\n🎉 Created {len(expert_names)} mock experts for testing!")
    print("Now the system has diverse experts for routing:")
    for name in expert_names:
        print(f"  - {name}")

if __name__ == "__main__":
    main()