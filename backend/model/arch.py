from __future__ import annotations

from typing import Optional
import os
import io

# Third-party libraries are optional at runtime; silence type checkers if missing
try:
    import torch  # type: ignore
except ImportError:
    print("⚠️  PyTorch not available")
    torch = None
try:
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
except ImportError:
    print("⚠️  Transformers not available")
    AutoModelForCausalLM = None
    AutoTokenizer = None
    PreTrainedModel = None  
    PreTrainedTokenizerBase = None


def state_dict_to_bytes(state_dict: dict) -> bytes:
    """Serialize a PyTorch state_dict to bytes."""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def bytes_to_state_dict(data: bytes) -> dict:
    """Deserialize bytes into a PyTorch state_dict."""
    buffer = io.BytesIO(data)
    # Ensure tensors are loaded onto the CPU first
    return torch.load(buffer, map_location=torch.device('cpu'))


class ModelWrapper:
    """A thin convenience wrapper around Hugging Face causal language models."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device_map: Optional[str] = None,
        max_memory: Optional[dict] = None,
        allow_mock_fallback: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.allow_mock_fallback = allow_mock_fallback
        
        # Check if we're in blockchain-only mode
        blockchain_only = os.getenv('BLOCKCHAIN_ONLY', 'true').lower() == 'true'
        
        if blockchain_only:
            print("🔗 Blockchain-only mode: Skipping local model loading")
            print("   Models must be loaded from blockchain expert blocks")
            # Don't try to load any local models
            return
        
        # Determine if we need quantization for large models
        is_large_model = "20b" in model_name.lower() or "neox-20b" in model_name.lower()
        # Allow disabling auto quantization via env
        auto_quant_env = os.getenv("MODEL_AUTO_QUANTIZE") or os.getenv("AUTO_QUANTIZE")
        auto_quantize = True if auto_quant_env is None else auto_quant_env.lower() not in {"0", "false", "no", "off"}
        if is_large_model and not (load_in_8bit or load_in_4bit) and auto_quantize:
            print(f"⚠️ Large model detected ({model_name}), enabling INT8 quantization (can disable with MODEL_AUTO_QUANTIZE=0)")
            load_in_8bit = True
        
        try:
            # Check if it's a local model path (only in non-blockchain mode)
            local_path = f"./models/{model_name}" if not blockchain_only else None
            
            # Prepare loading kwargs
            load_kwargs = {}
            
            # For large models or multi-GPU, use device_map
            if device_map or is_large_model:
                load_kwargs['device_map'] = device_map or "auto"
                print(f"🚀 Using device_map={load_kwargs['device_map']} for model distribution")
            
            # Add quantization options
            if load_in_8bit:
                load_kwargs['load_in_8bit'] = True
                print("📦 Loading model in INT8 (8-bit quantization)")
            elif load_in_4bit:
                load_kwargs['load_in_4bit'] = True
                load_kwargs['bnb_4bit_compute_dtype'] = torch.float16
                print("📦 Loading model in INT4 (4-bit quantization)")
            else:
                load_kwargs['torch_dtype'] = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # Add memory limits if specified
            if max_memory:
                load_kwargs['max_memory'] = max_memory
                print(f"💾 Memory limits: {max_memory}")
            
            # Some models need this (and unknown community models may too)
            if (
                "neox" in model_name.lower()
                or "pythia" in model_name.lower()
                or "mpt" in model_name.lower()
                or "llama" in model_name.lower()
                or "gpt-oss" in model_name.lower()
            ):
                load_kwargs['trust_remote_code'] = True

            # Authentication token (for gated/private models)
            # Support both legacy and new parameter names
            hf_token = (
                os.getenv("HF_TOKEN")
                or os.getenv("HUGGINGFACEHUB_API_TOKEN")
                or os.getenv("HUGGING_FACE_HUB_TOKEN")
            )

            # Optional force re-download to bypass corrupted cache
            force_download_env = os.getenv("HF_FORCE_DOWNLOAD")
            force_download = False
            if force_download_env and force_download_env.lower() not in {"0", "false", "no", "off"}:
                force_download = True
            
            if local_path and os.path.exists(local_path):
                print(f"🔍 Loading local model from {local_path}")
                # Try fast tokenizer, then fall back to slow
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=True)
                except Exception as e_tok:
                    print(f"⚠️  Fast tokenizer failed ({type(e_tok).__name__}): {e_tok}. Falling back to slow tokenizer...")
                    self.tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=False)
                self.model = AutoModelForCausalLM.from_pretrained(local_path, **load_kwargs)
            else:
                # Try loading from HuggingFace
                print(f"🌐 Loading model from HuggingFace: {model_name}")
                tokenizer_kwargs = {}
                # Token argument handling for different versions of Transformers/HF Hub
                if hf_token:
                    try:
                        tokenizer_kwargs['token'] = hf_token  # new API
                    except Exception:
                        tokenizer_kwargs['use_auth_token'] = hf_token  # legacy
                if force_download:
                    tokenizer_kwargs['force_download'] = True
                
                # Prefer native tokenizer for GPT-OSS-20B; fall back to generic AutoTokenizer or GPT2
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name, use_fast=True, **tokenizer_kwargs
                    )
                except Exception as e_tok:
                    print(f"⚠️  Fast tokenizer failed ({type(e_tok).__name__}): {e_tok}. Falling back to slow tokenizer...")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name, use_fast=False, **tokenizer_kwargs
                        )
                    except Exception as e_slow:
                        print(f"⚠️ Slow tokenizer also failed: {e_slow}")
                        from transformers import GPT2TokenizerFast
                        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
                        print("✅ Using GPT2 tokenizer as ultimate fallback")
                model_kwargs = dict(load_kwargs)
                if hf_token:
                    # Add token for model download too
                    # Transformers supports 'token' (new) or 'use_auth_token' (legacy)
                    model_kwargs.setdefault('token', hf_token)
                if force_download:
                    model_kwargs['force_download'] = True
                
                # Try to load the model
                model_loaded = False
                
                # Always load requested repo (no neox fallback)
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                    model_loaded = True
                except TypeError:
                    model_kwargs.pop('token', None)
                    model_kwargs['use_auth_token'] = hf_token
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                    model_loaded = True
                
                if not model_loaded:
                    raise RuntimeError(f"Failed to load model: {model_name}")
            
            # Set padding token if not set (required for batch generation)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Only move to device if not using device_map (device_map handles placement)
            if not device_map and not load_kwargs.get('device_map'):
                self.model.to(self.device)
            
            self.model.eval()
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters()) / 1e9
            print(f"✓ Loaded model {model_name} ({total_params:.1f}B params)")
            
            if hasattr(self.model, 'hf_device_map'):
                print(f"📍 Device placement: {self.model.hf_device_map}")
                
        except Exception as e:
            print(f"⚠️  Could not load {model_name}: {e}")
            # Provide a hint for common tokenizers JSON errors
            if "ModelWrapper" in str(e) and "untagged enum" in str(e):
                print(
                    "💡 Hint: Your installed 'tokenizers' may not support this tokenizer.json."
                    " Try setting use_fast=False (now auto-falling back) or upgrade tokenizers."
                )
                print(
                    "   If the repo is not a Transformers model (e.g., GGUF), use a compatible HF repo."
                )
            if not self.allow_mock_fallback:
                raise
            print("Using mock model for testing...")
            self._create_mock_model()

    def apply_weights(self, state_dict: dict):
        """Load a state_dict into the model."""
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)  # Ensure model is on the correct device after loading

    # ------------------------------------------------------------------
    # Generation helper
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 64, **gen_kwargs) -> str:
        if self.model is None or self.tokenizer is None:
            return f"Mock AI response to: '{prompt}' (using Blyan MoE system)"
        
        try:
            # Format prompt for Mistral/Mixtral models
            if "mixtral" in self.model_name.lower() or "mistral" in self.model_name.lower():
                # Try a simple instruction format
                formatted_prompt = f"Human: {prompt}\n\nAssistant:"
            else:
                formatted_prompt = prompt
                
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            input_length = inputs['input_ids'].shape[1]
            
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=gen_kwargs.get("temperature", 0.8),
                top_p=gen_kwargs.get("top_p", 0.95),
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            # Only decode the newly generated tokens (exclude input)
            new_tokens = output_ids[0][input_length:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return generated_text
        except Exception as e:
            print(f"⚠️  Generation failed: {e}")
            return f"Blyan MoE response to: '{prompt}' [Note: Using fallback due to model error]"
    
    def _create_mock_model(self):
        """Create a mock model that returns unavailable message."""
        # Don't load any fallback models - just use mock that shows unavailable
        self.tokenizer = MockTokenizer()
        self.model = MockModel()
        print("⚠️ Model not available - inference disabled")


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.eos_token = "<|endoftext|>"
        self.pad_token = self.eos_token
    
    def __call__(self, text, return_tensors=None):
        # Return mock tensor format
        import torch
        return {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),  # Mock token IDs
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
    
    def decode(self, token_ids, skip_special_tokens=True):
        # Return unavailable message
        return "Model inference is not available at this time. Please try again later."


class MockModel:
    """Mock model for testing."""
    
    def __init__(self):
        self.device = "cpu"
    
    def to(self, device):
        self.device = device
        return self
    
    def eval(self):
        return self
    
    def load_state_dict(self, state_dict):
        print(f"✓ Mock model loaded {len(state_dict)} parameters")
    
    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=64, **kwargs):
        # Return mock generation
        import torch
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        seq_len = input_ids.shape[1] if input_ids is not None else 5
        
        # Generate mock token IDs
        new_tokens = torch.randint(1, 100, (batch_size, max_new_tokens))
        if input_ids is not None:
            return torch.cat([input_ids, new_tokens], dim=1)
        else:
            return new_tokens 