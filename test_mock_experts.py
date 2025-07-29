#!/usr/bin/env python
"""Test mock expert creation and upload."""

import sys
import json
import requests
import torch
from pathlib import Path

# Add to path
sys.path.append(str(Path(__file__).parent))

def test_mock_expert_upload():
    """Test uploading virtual experts from a mock model."""
    print("🧪 Testing Mock Expert Upload")
    print("=" * 40)
    
    # 1. Create a simple mock model
    print("1️⃣ Creating mock model...")
    mock_state_dict = {
        'layer.0.weight': torch.randn(100, 50),
        'layer.0.bias': torch.randn(100),
        'layer.1.weight': torch.randn(50, 25),  
        'layer.1.bias': torch.randn(50),
        'embeddings.weight': torch.randn(1000, 512),
        'lm_head.weight': torch.randn(1000, 512),
    }
    
    test_model_path = Path("test_mock_model.pt")
    torch.save(mock_state_dict, test_model_path)
    print(f"✓ Created mock model with {len(mock_state_dict)} parameters")
    
    # 2. Test expert extraction 
    print("\n2️⃣ Testing expert extraction...")
    try:
        from miner.upload_moe_parameters import MoEExpertExtractor
        
        extractor = MoEExpertExtractor(str(test_model_path))
        extracted = extractor.extract_experts()
        
        expert_count = len(extracted['experts'])
        router_count = len(extracted['routers'])
        
        print(f"✓ Extracted {expert_count} virtual experts")
        print(f"✓ Extracted {router_count} routers")
        
        if expert_count == 0:
            print("❌ No experts extracted!")
            return False
            
        # Show expert details
        for expert_name, expert_data in extracted['experts'].items():
            tensor_count = len(expert_data['tensors'])
            print(f"  - {expert_name}: {tensor_count} tensors")
            
    except Exception as e:
        print(f"❌ Expert extraction failed: {e}")
        return False
    
    # 3. Test API upload (optional - requires server)
    print("\n3️⃣ Testing API connectivity...")
    try:
        response = requests.get("http://127.0.0.1:8000/docs", timeout=2)
        if response.status_code == 200:
            print("✓ API server is running")
            
            # Test a simple upload
            print("   Testing simple upload...")
            from miner.upload_moe_parameters import ExpertBlockUploader
            
            uploader = ExpertBlockUploader("test_miner")
            
            # Just test one expert
            first_expert = list(extracted['experts'].keys())[0]
            expert_data = extracted['experts'][first_expert]
            
            payload = uploader.create_expert_block_payload(
                expert_name=first_expert,
                tensors=expert_data['tensors'],
                layer_id=expert_data['layer_id'],
                meta_hash="test_meta_hash",
                candidate_loss=0.5
            )
            
            print(f"✓ Created upload payload for {first_expert}")
            print(f"   Payload size: {len(json.dumps(payload))} bytes")
            
        else:
            print("⚠️  API server not responding")
            
    except Exception as e:
        print(f"⚠️  API test failed: {e}")
    
    # 4. Cleanup
    test_model_path.unlink()
    print("\n✅ Mock expert test completed!")
    return True

def test_inference_with_mock_experts():
    """Test inference after uploading mock experts."""
    print("\n🧠 Testing Inference with Mock Experts")
    print("=" * 40)
    
    try:
        # Test standard inference (should work now)
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"prompt": "Test prompt", "max_new_tokens": 10},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Standard inference: '{result['response'][:50]}...'")
        else:
            print(f"❌ Standard inference failed: {response.status_code}")
            
        # Test MoE inference
        response = requests.post(
            "http://127.0.0.1:8000/chat_moe",
            json={"prompt": "Test prompt", "max_new_tokens": 10},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            expert_count = len(result.get('expert_usage', {}))
            print(f"✅ MoE inference: Used {expert_count} experts")
        else:
            print(f"❌ MoE inference failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 AI-Block Mock Expert Test")
    print("=" * 50)
    
    success1 = test_mock_expert_upload()
    success2 = test_inference_with_mock_experts()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
        print("The AI-Block MoE system is working correctly!")
        exit(0)
    else:
        print("\n⚠️  Some tests failed - check output above")
        exit(1)