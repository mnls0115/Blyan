#!/usr/bin/env python3
"""
RunPod Qwen3-30B GPU Node
Optimized for loading and serving the Qwen3-30B-A3B-Instruct model
"""
import os
import sys
import torch
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Model configuration
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
MODEL_REQUIREMENTS = {
    "min_gpu_memory_gb": 32,
    "recommended_gpu_memory_gb": 48,
    "precision": "FP8",
    "total_params": "30.5B",
    "active_params": "3.3B",
    "num_experts": 128,
    "num_layers": 48
}

def check_gpu_capability():
    """Check if GPU meets requirements for Qwen3-30B."""
    print("\n" + "="*60)
    print("🔍 Checking GPU for Qwen3-30B Model")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Cannot run this model without GPU.")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ Found {num_gpus} GPU(s)")
    
    total_memory = 0
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        total_memory += memory_gb
        
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {memory_gb:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Check compute capability for FP8 support
        if props.major >= 9:  # Ada Lovelace or newer
            print(f"  ✅ Native FP8 support")
        elif props.major == 8 and props.minor >= 9:  # Some Ampere
            print(f"  ⚠️  Limited FP8 support")
        else:
            print(f"  ❌ No FP8 support (will use FP16 fallback)")
    
    print(f"\n📊 Total GPU Memory: {total_memory:.1f} GB")
    print(f"📋 Model Requirements: {MODEL_REQUIREMENTS['min_gpu_memory_gb']} GB minimum")
    
    if total_memory < MODEL_REQUIREMENTS['min_gpu_memory_gb']:
        print(f"❌ Insufficient GPU memory! Need at least {MODEL_REQUIREMENTS['min_gpu_memory_gb']} GB")
        print("   Consider using quantization or a smaller model")
        return False
    elif total_memory < MODEL_REQUIREMENTS['recommended_gpu_memory_gb']:
        print(f"⚠️  GPU memory below recommended {MODEL_REQUIREMENTS['recommended_gpu_memory_gb']} GB")
        print("   Model will load but may be slow or require offloading")
    else:
        print(f"✅ Sufficient GPU memory for optimal performance")
    
    return True

def test_fp8_support():
    """Test if PyTorch supports FP8 operations."""
    print("\n" + "="*60)
    print("🧪 Testing FP8 Support")
    print("="*60)
    
    try:
        # Check if torch has FP8 dtype (requires recent version)
        if hasattr(torch, 'float8_e4m3fn'):
            print("✅ PyTorch has FP8 E4M3 support")
            
            # Try creating FP8 tensor
            test_tensor = torch.randn(10, 10, dtype=torch.float8_e4m3fn, device='cuda')
            print("✅ Can create FP8 tensors")
            return True
        else:
            print("⚠️  PyTorch version doesn't have native FP8")
            print("   Will use auto dtype for model loading")
            return False
    except Exception as e:
        print(f"⚠️  FP8 test failed: {e}")
        print("   Will fallback to FP16 precision")
        return False

def load_model_with_progress():
    """Load Qwen3-30B model with progress tracking."""
    print("\n" + "="*60)
    print("🚀 Loading Qwen3-30B Model")
    print("="*60)
    
    print(f"Model: {MODEL_ID}")
    print(f"Total Parameters: {MODEL_REQUIREMENTS['total_params']}")
    print(f"Active Parameters: {MODEL_REQUIREMENTS['active_params']} per token")
    print(f"Experts: {MODEL_REQUIREMENTS['num_experts']} per layer × {MODEL_REQUIREMENTS['num_layers']} layers")
    print("")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Set cache directory
        cache_dir = os.environ.get("HF_HOME", "./models")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"📁 Cache directory: {cache_dir}")
        
        # Load tokenizer first (quick)
        print("\n⏳ Loading tokenizer...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print(f"✅ Tokenizer loaded in {time.time() - start_time:.1f}s")
        
        # Load model (this will take time)
        print("\n⏳ Loading model weights (this may take 5-10 minutes)...")
        print("   Downloading if not cached...")
        start_time = time.time()
        
        # Use auto dtype to preserve FP8 if available
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",  # Let transformers decide based on model config
            device_map="auto",   # Distribute across all GPUs
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            trust_remote_code=True,
            offload_folder="offload",  # Offload to CPU if needed
            offload_state_dict=True    # Offload state dict during loading
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f}s ({load_time/60:.1f} minutes)")
        
        # Check memory usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / 1e9
                mem_reserved = torch.cuda.memory_reserved(i) / 1e9
                print(f"\nGPU {i} Memory:")
                print(f"  Allocated: {mem_allocated:.2f} GB")
                print(f"  Reserved: {mem_reserved:.2f} GB")
        
        # Test generation
        print("\n🧪 Testing inference...")
        inputs = tokenizer("Hello, I am", return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True
            )
            gen_time = time.time() - start_time
            
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Generated in {gen_time:.2f}s: {result}")
        
        # Calculate tokens per second
        num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        tokens_per_sec = num_tokens / gen_time
        print(f"⚡ Speed: {tokens_per_sec:.1f} tokens/second")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("🚀 Qwen3-30B RunPod Setup")
    print("="*60)
    
    # Check environment
    print("\n📋 Environment:")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
    
    # Check GPU capability
    if not check_gpu_capability():
        print("\n❌ GPU requirements not met!")
        print("\nOptions:")
        print("1. Use a RunPod instance with more GPU memory (48GB+ recommended)")
        print("2. Use quantization (INT8 or INT4) to reduce memory usage")
        print("3. Use a smaller model for testing")
        return
    
    # Test FP8 support
    has_fp8 = test_fp8_support()
    
    # Ask user before loading
    print("\n" + "="*60)
    print("⚠️  WARNING: Loading Qwen3-30B will:")
    print("  - Download ~30GB if not cached")
    print("  - Use most of your GPU memory")
    print("  - Take 5-10 minutes to load")
    print("="*60)
    
    response = input("\nProceed with loading? (yes/no): ").strip().lower()
    if response != "yes":
        print("Aborted by user")
        return
    
    # Load model
    model, tokenizer = load_model_with_progress()
    
    if model is not None:
        print("\n" + "="*60)
        print("✅ SUCCESS! Qwen3-30B is ready!")
        print("="*60)
        print("\nModel is loaded and ready for inference.")
        print("You can now integrate this with your API server.")
    else:
        print("\n" + "="*60)
        print("❌ FAILED to load model")
        print("="*60)
        print("\nTroubleshooting:")
        print("1. Check GPU memory with: nvidia-smi")
        print("2. Try restarting with: torch.cuda.empty_cache()")
        print("3. Use smaller batch size or quantization")

if __name__ == "__main__":
    main()