#!/usr/bin/env python3
"""
RunPod GPU Test - Minimal test for GPU availability and model loading
"""
import os
import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gpu():
    """Test GPU availability and capabilities."""
    print("\n" + "="*60)
    print("🔍 GPU Detection Test")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available!")
        print("   This means PyTorch cannot see any GPUs")
        return False
    
    print("✅ CUDA is available!")
    
    # Count GPUs
    num_gpus = torch.cuda.device_count()
    print(f"📊 Number of GPUs detected: {num_gpus}")
    
    if num_gpus == 0:
        print("❌ No GPUs detected by PyTorch!")
        return False
    
    # Show GPU details
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"\n🎮 GPU {i}: {props.name}")
        print(f"   Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"   Compute Capability: {props.major}.{props.minor}")
        
        # Test memory allocation
        try:
            # Try to allocate 1GB
            test_tensor = torch.zeros(256, 1024, 1024, dtype=torch.float16, device=f'cuda:{i}')
            print(f"   ✅ Can allocate memory on GPU {i}")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ❌ Cannot allocate memory on GPU {i}: {e}")
            return False
    
    print("\n✅ All GPU tests passed!")
    return True

def test_model_loading():
    """Test minimal model loading on GPU."""
    print("\n" + "="*60)
    print("🤖 Model Loading Test")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ Skipping model test - no GPU available")
        return False
    
    try:
        # Test with a tiny model first
        print("Testing with small GPT-2 model (124M params)...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "gpt2"  # Small 124M model for testing
        
        print(f"Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Loading model to GPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Test inference
        print("Testing inference...")
        inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: {result}")
        
        # Check memory usage
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\n📊 GPU Memory Usage:")
        print(f"   Allocated: {mem_allocated:.2f} GB")
        print(f"   Reserved: {mem_reserved:.2f} GB")
        
        # Cleanup
        del model
        del inputs
        del outputs
        torch.cuda.empty_cache()
        
        print("\n✅ Model loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n🚀 RunPod GPU Test Suite")
    print("="*60)
    
    # Show environment
    print("\n📋 Environment:")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"   CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Run tests
    gpu_ok = test_gpu()
    
    if gpu_ok:
        model_ok = test_model_loading()
        
        if model_ok:
            print("\n✅ ✅ ✅ ALL TESTS PASSED! ✅ ✅ ✅")
            print("Your RunPod GPU setup is working correctly!")
        else:
            print("\n⚠️ GPU works but model loading failed")
            print("Check your transformers installation")
    else:
        print("\n❌ GPU detection failed!")
        print("\nTroubleshooting steps:")
        print("1. Check if you selected a GPU pod (not CPU)")
        print("2. Try restarting the pod")
        print("3. Check CUDA installation with: nvidia-smi")
        print("4. Ensure PyTorch is installed with CUDA support:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    main()