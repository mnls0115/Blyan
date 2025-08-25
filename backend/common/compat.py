#!/usr/bin/env python3
"""
Universal compatibility layer for transformers, bitsandbytes, and torch.
Handles version differences, quantization fallbacks, and common issues.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
import warnings

logger = logging.getLogger(__name__)

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_hf_cache(cache_dir: Optional[Path] = None) -> Path:
    """
    Setup HuggingFace cache directory and optimize downloads.
    
    Args:
        cache_dir: Optional cache directory path
        
    Returns:
        Path to cache directory
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface"
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for HF
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    
    # Enable faster downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    # Disable telemetry
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    logger.info(f"HF cache configured at: {cache_dir}")
    return cache_dir

# ============================================================================
# TRITON COMPATIBILITY SHIMS
# ============================================================================

def _setup_triton_shims():
    """
    Fix all known triton-related import issues across transformers versions.
    """
    try:
        import transformers
        import transformers.utils
        
        # Common triton stub
        def _triton_stub(*args, **kwargs):
            return False
        
        # Fix typo variations
        if not hasattr(transformers.utils, 'is_triton_kernels_availalble'):
            transformers.utils.is_triton_kernels_availalble = _triton_stub
            
        if not hasattr(transformers.utils, 'is_triton_kernels_available'):
            transformers.utils.is_triton_kernels_available = _triton_stub
            
        if not hasattr(transformers.utils, 'is_triton_available'):
            transformers.utils.is_triton_available = _triton_stub
            
        # Also patch import_utils if it exists
        try:
            import transformers.utils.import_utils
            if not hasattr(transformers.utils.import_utils, 'is_triton_kernels_availalble'):
                transformers.utils.import_utils.is_triton_kernels_availalble = _triton_stub
            if not hasattr(transformers.utils.import_utils, 'is_triton_kernels_available'):
                transformers.utils.import_utils.is_triton_kernels_available = _triton_stub
            if not hasattr(transformers.utils.import_utils, 'is_triton_available'):
                transformers.utils.import_utils.is_triton_available = _triton_stub
        except ImportError:
            pass
            
    except Exception as e:
        logger.debug(f"Triton shim setup skipped: {e}")

# Apply shims on import
_setup_triton_shims()

# ============================================================================
# BITSANDBYTES COMPATIBILITY
# ============================================================================

def _setup_bnb_compat():
    """
    Add compatibility methods for BitsAndBytesConfig across versions.
    """
    try:
        from transformers import BitsAndBytesConfig
        
        # Add get_loading_attributes if missing (older transformers versions)
        if not hasattr(BitsAndBytesConfig, 'get_loading_attributes'):
            def _get_loading_attributes(self):
                """Compatibility method for older transformers."""
                attrs = {}
                if getattr(self, 'load_in_8bit', False):
                    attrs['load_in_8bit'] = True
                if getattr(self, 'load_in_4bit', False):
                    attrs['load_in_4bit'] = True
                if hasattr(self, 'llm_int8_enable_fp32_cpu_offload'):
                    attrs['llm_int8_enable_fp32_cpu_offload'] = self.llm_int8_enable_fp32_cpu_offload
                if hasattr(self, 'bnb_4bit_quant_type'):
                    attrs['bnb_4bit_quant_type'] = self.bnb_4bit_quant_type
                if hasattr(self, 'bnb_4bit_use_double_quant'):
                    attrs['bnb_4bit_use_double_quant'] = self.bnb_4bit_use_double_quant
                if hasattr(self, 'bnb_4bit_compute_dtype'):
                    attrs['bnb_4bit_compute_dtype'] = self.bnb_4bit_compute_dtype
                return attrs
            
            BitsAndBytesConfig.get_loading_attributes = _get_loading_attributes
            
    except Exception as e:
        logger.debug(f"BnB compat setup skipped: {e}")

# Apply BnB compat on import
_setup_bnb_compat()

# ============================================================================
# CAPABILITY DETECTION
# ============================================================================

def detect_capabilities() -> Dict[str, Any]:
    """
    Detect available hardware and software capabilities.
    
    Returns:
        Dictionary with capability flags
    """
    caps = {
        'cuda': False,
        'cuda_version': None,
        'gpu_name': None,
        'gpu_memory_gb': 0,
        'bitsandbytes': False,
        'bnb_cuda': False,
        'torch_version': None,
        'transformers_version': None,
        'supports_bf16': False,
        'supports_flash_attn': False,
    }
    
    # Check PyTorch and CUDA
    try:
        import torch
        caps['torch_version'] = torch.__version__
        caps['cuda'] = torch.cuda.is_available()
        
        if caps['cuda']:
            caps['cuda_version'] = torch.version.cuda
            caps['gpu_name'] = torch.cuda.get_device_name(0)
            caps['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            caps['supports_bf16'] = torch.cuda.is_bf16_supported()
            
    except ImportError:
        logger.warning("PyTorch not installed")
    
    # Check transformers
    try:
        import transformers
        caps['transformers_version'] = transformers.__version__
    except ImportError:
        logger.warning("Transformers not installed")
    
    # Check bitsandbytes
    try:
        import bitsandbytes as bnb
        caps['bitsandbytes'] = True
        # Check if BnB was compiled with CUDA support
        try:
            import torch
            if torch.cuda.is_available():
                # Try to create a simple quantized tensor to verify CUDA support
                test_tensor = torch.randn(10, 10, device='cuda')
                _ = bnb.functional.quantize_4bit(test_tensor)
                caps['bnb_cuda'] = True
        except Exception:
            caps['bnb_cuda'] = False
            logger.warning("BitsAndBytes installed but CUDA support not available")
    except ImportError:
        logger.debug("BitsAndBytes not installed")
    
    # Check flash attention
    try:
        from flash_attn import flash_attn_func
        caps['supports_flash_attn'] = True
    except ImportError:
        pass
    
    return caps

# ============================================================================
# TOKENIZER LOADING
# ============================================================================

def load_tokenizer(model_name: str, **kwargs) -> Any:
    """
    Load tokenizer robustly - try fast first, then slow if available.
    
    Args:
        model_name: Model name or path
        **kwargs: Additional tokenizer arguments
        
    Returns:
        Loaded tokenizer
    """
    from transformers import AutoTokenizer
    
    # Add trust_remote_code by default for compatibility
    if 'trust_remote_code' not in kwargs:
        kwargs['trust_remote_code'] = True
    
    # Try fast tokenizer first
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            **kwargs
        )
        logger.info(f"Loaded fast tokenizer for {model_name}")
        return tokenizer
    except Exception as e_fast:
        logger.warning(f"Fast tokenizer failed: {e_fast}, trying slow tokenizer")
        
        # Try slow tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                **kwargs
            )
            logger.info(f"Loaded slow tokenizer for {model_name}")
            return tokenizer
        except Exception as e_slow:
            error_msg = (
                f"Failed to load tokenizer for {model_name}.\n"
                f"Fast error: {e_fast}\n"
                f"Slow error: {e_slow}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e_slow

# ============================================================================
# MODEL LOADING WITH FALLBACKS
# ============================================================================

def load_model_any_precision(
    model_name: str,
    prefer_quant: Optional[str] = None,
    device_pref: Optional[str] = "auto",
    max_memory_gb: Optional[float] = None,
    **extra_kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load model with BF16/FP16 precision (no quantization).
    
    Args:
        model_name: Model name or path
        prefer_quant: Ignored - no quantization used
        device_pref: Device preference ("auto", "cuda", "cpu")
        max_memory_gb: Maximum GPU memory to use
        **extra_kwargs: Additional model loading arguments
        
    Returns:
        Tuple of (model, info_dict) where info_dict contains loading details
    """
    from transformers import AutoModelForCausalLM
    import torch
    
    caps = detect_capabilities()
    info = {
        'quantization': None,
        'dtype': None,
        'device': None,
        'method': None,
        'fallback_reason': None
    }
    
    # BF16 ONLY - no fallbacks
    if not caps['cuda']:
        raise RuntimeError("CUDA required for BF16 inference. CPU mode not supported.")
    
    if not caps['supports_bf16']:
        raise RuntimeError("GPU does not support BF16. Minimum compute capability 8.0 required (Ampere or newer).")
    
    # Only use BF16 - fail fast if not supported
    device = "cuda"  # BF16 requires CUDA
    
    logger.info("Loading model with BF16 precision (REQUIRED - no fallbacks)...")
    
    try:
        model = _load_bf16(
            model_name,
            device=device,
            max_memory_gb=max_memory_gb,
            **extra_kwargs
        )
        
        info['method'] = 'bf16'
        info['device'] = device
        info['dtype'] = 'bf16'
        
        logger.info("✅ Successfully loaded model with BF16 precision")
        return model, info
        
    except Exception as e:
        logger.error(f"Failed to load model with BF16: {e}")
        raise RuntimeError(f"BF16 loading failed (no fallbacks allowed): {e}")

# ============================================================================
# SPECIFIC LOADING FUNCTIONS
# ============================================================================

# BF16 ONLY - no quantization, no fallbacks

def _load_bf16(model_name: str, device: str, max_memory_gb: Optional[float] = None, **kwargs):
    """Load model with bfloat16 precision - REQUIRED, no fallbacks."""
    from transformers import AutoModelForCausalLM
    import torch
    
    # Verify BF16 support
    if device != 'cuda':
        raise RuntimeError("BF16 requires CUDA. CPU mode not supported.")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. BF16 requires GPU.")
    
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError(f"GPU compute capability {torch.cuda.get_device_capability()} does not support BF16. Minimum 8.0 required (Ampere or newer).")
    
    load_kwargs = {
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': True,
        'low_cpu_mem_usage': True,
        'device_map': 'auto',
        **kwargs
    }
    
    if max_memory_gb:
        load_kwargs['max_memory'] = {0: f"{int(max_memory_gb)}GB"}
    
    logger.info("Loading model in BF16 precision (REQUIRED - no fallbacks)")
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    
    return model

# Removed FP16 and FP32 functions - BF16 only

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _get_transformers_version() -> Tuple[int, int, int]:
    """Get transformers version as tuple for comparison."""
    try:
        import transformers
        version_str = transformers.__version__
        # Handle versions like "4.30.0.dev0"
        version_parts = version_str.split('.')[:3]
        version_parts = [p.split('dev')[0] for p in version_parts]  # Remove dev suffix
        return tuple(int(p) for p in version_parts if p.isdigit())
    except Exception:
        return (0, 0, 0)

def print_capabilities():
    """Print detected capabilities for debugging."""
    caps = detect_capabilities()
    
    print("\n" + "="*60)
    print("SYSTEM CAPABILITIES")
    print("="*60)
    
    print(f"PyTorch: {caps.get('torch_version', 'Not installed')}")
    print(f"Transformers: {caps.get('transformers_version', 'Not installed')}")
    print(f"CUDA Available: {caps['cuda']}")
    
    if caps['cuda']:
        print(f"CUDA Version: {caps['cuda_version']}")
        print(f"GPU: {caps['gpu_name']}")
        print(f"GPU Memory: {caps['gpu_memory_gb']:.2f} GB")
        print(f"BF16 Support: {caps['supports_bf16']}")
    
    print(f"BitsAndBytes: {caps['bitsandbytes']}")
    if caps['bitsandbytes']:
        print(f"BitsAndBytes CUDA: {caps['bnb_cuda']}")
    
    print(f"Flash Attention: {caps['supports_flash_attn']}")
    print("="*60 + "\n")

# ============================================================================
# MAIN ENTRY POINT FOR TESTING
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Print capabilities
    print_capabilities()
    
    # Test loading if model name provided
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        print(f"\nTesting model loading for: {model_name}")
        
        try:
            # Setup cache
            setup_hf_cache()
            
            # Load tokenizer
            print("Loading tokenizer...")
            tokenizer = load_tokenizer(model_name)
            print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
            
            # Load model
            print("Loading model...")
            model, info = load_model_any_precision(
                model_name,
                prefer_quant="auto"
            )
            print(f"✅ Model loaded successfully!")
            print(f"   Method: {info['method']}")
            print(f"   Device: {info['device']}")
            print(f"   Quantization: {info.get('quantization', 'None')}")
            print(f"   Dtype: {info.get('dtype', 'N/A')}")
            
            if info.get('fallback_reason'):
                print(f"   Fallback reason: {info['fallback_reason']}")
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()