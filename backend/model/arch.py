from __future__ import annotations

from typing import Optional
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
        max_memory: Optional[dict] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        # Determine if we need quantization for large models
        is_large_model = "20b" in model_name.lower() or "neox-20b" in model_name.lower()
        if is_large_model and not (load_in_8bit or load_in_4bit):
            print(f"⚠️ Large model detected ({model_name}), enabling INT8 quantization")
            load_in_8bit = True
        
        try:
            # Check if it's a local model path
            import os
            local_path = f"./models/{model_name}"
            
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
            
            # Some models need this
            if "neox" in model_name.lower() or "pythia" in model_name.lower():
                load_kwargs['trust_remote_code'] = True
            
            if os.path.exists(local_path):
                print(f"🔍 Loading local model from {local_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_path)
                self.model = AutoModelForCausalLM.from_pretrained(local_path, **load_kwargs)
            else:
                # Try loading from HuggingFace
                print(f"🌐 Loading model from HuggingFace: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            
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
        """Create a mock model and tokenizer for testing when real model fails."""
        try:
            # Try to create a simple tokenizer
            from transformers import GPT2TokenizerFast
            self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except:
            # Create minimal mock tokenizer
            self.tokenizer = MockTokenizer()
        
        # Mock model that just returns test responses
        self.model = MockModel()
        print("✓ Created mock model for testing")


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
        # Return mock decoded text
        return "This is a mock response from Blyan MoE blockchain system."


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