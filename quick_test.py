#!/usr/bin/env python
"""Quick test to check if the fixes work."""

import sys
from pathlib import Path

# Add to path
sys.path.append(str(Path(__file__).parent))

def test_model_wrapper():
    """Test ModelWrapper with fallback."""
    print("🧪 Testing ModelWrapper...")
    
    try:
        from backend.model.arch import ModelWrapper
        
        # This should fallback to mock model
        wrapper = ModelWrapper("non-existent-model")
        
        # Test generation
        result = wrapper.generate("Hello world", max_new_tokens=10)
        print(f"✅ Generation result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ ModelWrapper test failed: {e}")
        return False

def test_expert_extraction():
    """Test virtual expert creation."""
    print("🧪 Testing Expert Extraction...")
    
    try:
        from miner.upload_moe_parameters import MoEExpertExtractor
        import torch
        
        # Create mock model file
        mock_state_dict = {
            'layer.0.weight': torch.randn(100, 50),
            'layer.0.bias': torch.randn(100),
            'layer.1.weight': torch.randn(50, 25),
            'layer.1.bias': torch.randn(50),
        }
        
        test_file = Path("test_mock_model.pt")
        torch.save(mock_state_dict, test_file)
        
        # Test extraction
        extractor = MoEExpertExtractor(str(test_file))
        extracted = extractor.extract_experts()
        
        experts = extracted['experts']
        print(f"✅ Extracted {len(experts)} virtual experts")
        
        # Cleanup
        test_file.unlink()
        
        return len(experts) > 0
        
    except Exception as e:
        print(f"❌ Expert extraction test failed: {e}")
        return False

def test_api_imports():
    """Test API server imports."""
    print("🧪 Testing API imports...")
    
    try:
        import api.server
        print("✅ API server imports successfully")
        return True
        
    except Exception as e:
        print(f"❌ API import failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Running Quick Tests")
    print("=" * 30)
    
    tests = [
        test_api_imports,
        test_model_wrapper,  
        test_expert_extraction,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System should work now.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    exit(0 if passed == total else 1)