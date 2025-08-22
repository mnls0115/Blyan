#!/usr/bin/env python3
"""
Fix GPU inference - Test script to verify GPU model loading
"""
import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_gpu_inference():
    """Test if GPU inference is working."""
    print("🔍 Testing GPU Inference Setup")
    print("="*50)
    
    # 1. Check GPU
    print("\n1. GPU Check:")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ❌ No GPU detected!")
        return False
    
    # 2. Test model loading
    print("\n2. Testing Model Loading:")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Use small model for quick test
        model_name = "gpt2"
        print(f"   Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        
        print("   ✅ Model loaded on GPU")
        
        # Test inference
        print("\n3. Testing Inference:")
        inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"   Input: 'Hello world'")
        print(f"   Output: '{result}'")
        print("   ✅ GPU inference working!")
        
        # Check memory usage
        mem_used = torch.cuda.memory_allocated() / 1e9
        print(f"\n4. GPU Memory Used: {mem_used:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def fix_blockchain_mode():
    """Fix the blockchain-only mode issue."""
    print("\n" + "="*50)
    print("📝 Fix Instructions:")
    print("="*50)
    print("""
The issue is that your node is in 'blockchain-only' mode but can't load experts properly.

To fix this, you need to either:

1. QUICK FIX - Disable blockchain-only mode:
   export BLOCKCHAIN_ONLY=false
   export USE_SMALL_MODEL=true
   
   Then restart the node:
   pkill -f run_gpu_node
   screen -S blyan bash -c "BLOCKCHAIN_ONLY=false USE_SMALL_MODEL=true python run_gpu_node.py"

2. OR - Load actual model (not from blockchain):
   screen -S blyan bash -c "SKIP_BLOCKCHAIN=true python run_gpu_node.py"

3. OR - Use the fast test mode:
   Create a simple FastAPI server that actually uses GPU:
""")

if __name__ == "__main__":
    if test_gpu_inference():
        fix_blockchain_mode()
    else:
        print("\n❌ GPU not working properly!")
        print("Check your CUDA installation and GPU availability")