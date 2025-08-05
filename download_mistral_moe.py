#!/usr/bin/env python3
"""
Download TinyMistral-6x248M-Instruct MoE model for Blyan distributed MoE system.
This model has 6 experts (1.48B total, 248M active) - perfect for blockchain distribution!
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

def download_tiny_mistral_moe():
    """Download and save TinyMistral-6x248M-Instruct model."""
    
    print("⚡ Downloading TinyMistral-6x248M-Instruct...")
    print("📊 Model specs: 6 experts × 248M = 1.48B total, 248M active")
    print("💾 Expected size: ~1.5GB (still great for MacBook Air M4!)")
    print("🎯 Instruct-tuned for better conversation quality!")
    
    model_name = "M4-ai/TinyMistral-6x248M-Instruct"
    save_path = "./models/tiny_mistral_moe"
    
    # Create models directory
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # Download tokenizer
        print("\n📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        print("✅ Tokenizer saved!")
        
        # Download model (no quantization needed - already lightweight!)
        print("\n📥 Downloading model (6 MoE experts)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Half precision for efficiency
            low_cpu_mem_usage=True,
            device_map="auto"  # Let transformers handle device placement
        )
        
        # Save model
        model.save_pretrained(save_path)
        print("✅ Model saved!")
        
        # Display model info
        print(f"\n🎯 Model successfully downloaded to: {save_path}")
        print(f"📋 Config: {model.config}")
        
        # Check for MoE-specific attributes
        config = model.config
        if hasattr(config, 'num_experts'):
            print(f"🔥 Number of experts: {config.num_experts}")
        if hasattr(config, 'num_experts_per_tok'):
            print(f"🎲 Experts per token: {config.num_experts_per_tok}")
        if hasattr(config, 'router_aux_loss_coef'):
            print(f"⚖️ Router loss coefficient: {config.router_aux_loss_coef}")
            
        print("\n🚀 Ready for Blyan blockchain distribution!")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check internet connection")
        print("2. Verify Hugging Face access")
        print("3. Ensure sufficient disk space (~1.5GB)")
        return False
    
    return True

if __name__ == "__main__":
    success = download_tiny_mistral_moe()
    if success:
        print("\n✨ Next steps:")
        print("1. Initialize meta-chain with TinyMistral MoE specs")
        print("2. Extract 6 experts to blockchain blocks")
        print("3. Test distributed MoE inference!")
    else:
        print("\n💡 Try running with: python download_mistral_moe.py")