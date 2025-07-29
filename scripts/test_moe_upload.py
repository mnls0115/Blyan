#!/usr/bin/env python
"""Test script for MoE parameter upload functionality.

This creates a mock MoE model and tests the expert extraction and upload process.
"""

import json
import torch
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.chain import Chain
from backend.core.block import BlockHeader


def create_mock_moe_model() -> dict:
    """Create a mock MoE model state dict with realistic structure."""
    
    state_dict = {}
    
    # Base model weights (non-expert)
    state_dict['embeddings.weight'] = torch.randn(1000, 512)
    state_dict['norm.weight'] = torch.randn(512)
    state_dict['norm.bias'] = torch.randn(512)
    
    # Create 4 layers with 8 experts each
    for layer_idx in range(4):
        # Router weights for this layer
        state_dict[f'model.layers.{layer_idx}.mlp.router.weight'] = torch.randn(512, 8)
        state_dict[f'model.layers.{layer_idx}.mlp.router.bias'] = torch.randn(8)
        
        # Expert weights for this layer
        for expert_idx in range(8):
            # Each expert has typical MLP structure
            state_dict[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.w1.weight'] = torch.randn(2048, 512)
            state_dict[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.w2.weight'] = torch.randn(512, 2048)
            state_dict[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.w3.weight'] = torch.randn(2048, 512)
        
        # Attention weights (non-expert)
        state_dict[f'model.layers.{layer_idx}.attention.q_proj.weight'] = torch.randn(512, 512)
        state_dict[f'model.layers.{layer_idx}.attention.k_proj.weight'] = torch.randn(512, 512)
        state_dict[f'model.layers.{layer_idx}.attention.v_proj.weight'] = torch.randn(512, 512)
        state_dict[f'model.layers.{layer_idx}.attention.o_proj.weight'] = torch.randn(512, 512)
    
    # Output layer
    state_dict['lm_head.weight'] = torch.randn(1000, 512)
    
    return state_dict


def test_expert_extraction():
    """Test the expert extraction functionality."""
    print("Testing MoE expert extraction...")
    
    # Create mock model
    mock_state_dict = create_mock_moe_model()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(mock_state_dict, f.name)
        temp_model_path = f.name
    
    try:
        # Import the extractor
        from miner.upload_moe_parameters import MoEExpertExtractor
        
        # Extract experts
        extractor = MoEExpertExtractor(temp_model_path)
        extracted_data = extractor.extract_experts()
        
        # Validate extraction
        experts = extracted_data['experts']
        routers = extracted_data['routers']
        base = extracted_data['base']
        
        print(f"✓ Extracted {len(experts)} experts")
        print(f"✓ Extracted {len(routers)} routers")
        print(f"✓ Extracted {len(base)} base parameters")
        
        # Check expert structure
        expected_experts = 4 * 8  # 4 layers × 8 experts
        expected_routers = 4      # 4 layers
        
        assert len(experts) == expected_experts, f"Expected {expected_experts} experts, got {len(experts)}"
        assert len(routers) == expected_routers, f"Expected {expected_routers} routers, got {len(routers)}"
        
        # Check first expert structure
        first_expert_name = list(experts.keys())[0]
        first_expert = experts[first_expert_name]
        
        assert 'tensors' in first_expert
        assert 'layer_id' in first_expert
        assert 'expert_idx' in first_expert
        assert len(first_expert['tensors']) == 3  # w1, w2, w3
        
        print(f"✓ Expert structure validation passed")
        print(f"  First expert: {first_expert_name}")
        print(f"  Tensor count: {len(first_expert['tensors'])}")
        print(f"  Layer ID: {first_expert['layer_id']}")
        
        return True
        
    finally:
        # Clean up
        os.unlink(temp_model_path)


def test_dag_block_creation():
    """Test DAG block creation with expert metadata."""
    print("\nTesting DAG block creation...")
    
    # Create temporary chain
    with tempfile.TemporaryDirectory() as temp_dir:
        chain = Chain(Path(temp_dir), "TEST", difficulty=1)  # Low difficulty for testing
        
        # Create meta block first
        meta_payload = json.dumps({
            "model_name": "test-moe-model",
            "architecture": "mixture-of-experts",
            "num_layers": 4,
            "num_experts": 8
        }).encode()
        
        meta_block = chain.add_block(
            payload=meta_payload,
            block_type='meta'
        )
        meta_hash = meta_block.compute_hash()
        print(f"✓ Created meta block: {meta_hash[:16]}...")
        
        # Create expert block
        expert_payload = torch.save({'w1.weight': torch.randn(100, 50)}, 
                                  io := __import__('io').BytesIO()).getvalue() or io.getvalue()
        
        expert_block = chain.add_block(
            payload=expert_payload,
            depends_on=[meta_hash],
            block_type='expert',
            expert_name='layer0.expert0',
            layer_id='layer0'
        )
        
        expert_hash = expert_block.compute_hash()
        print(f"✓ Created expert block: {expert_hash[:16]}...")
        
        # Verify DAG structure
        assert chain.verify_dag(), "DAG verification failed"
        print("✓ DAG structure is valid")
        
        # Test block retrieval by type
        expert_blocks = chain.get_blocks_by_type('expert')
        assert len(expert_blocks) == 1, f"Expected 1 expert block, got {len(expert_blocks)}"
        
        meta_blocks = chain.get_blocks_by_type('meta')
        assert len(meta_blocks) == 1, f"Expected 1 meta block, got {len(meta_blocks)}"
        
        print("✓ Block type filtering works correctly")
        
        # Test expert-specific retrieval
        layer0_experts = chain.get_expert_blocks('layer0.expert0')
        assert len(layer0_experts) == 1, "Expert retrieval failed"
        
        layer0_blocks = chain.get_blocks_by_layer('layer0')
        assert len(layer0_blocks) == 1, "Layer-based retrieval failed"
        
        print("✓ Expert and layer filtering works correctly")
        
        return True


def main():
    """Run all tests."""
    print("=== MoE DAG Blockchain Test Suite ===\n")
    
    try:
        # Test 1: Expert extraction
        if not test_expert_extraction():
            print("❌ Expert extraction test failed")
            return 1
        
        # Test 2: DAG block creation
        if not test_dag_block_creation():
            print("❌ DAG block creation test failed")
            return 1
        
        print("\n🎉 All tests passed! MoE DAG blockchain is working correctly.")
        print("\nNext steps:")
        print("1. Start the API server: uvicorn api.server:app --reload")
        print("2. Initialize meta chain with MoE model spec")
        print("3. Upload MoE parameters: python miner/upload_moe_parameters.py --help")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())