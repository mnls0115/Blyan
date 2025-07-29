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

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Loaded model {model_name}")
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
            return f"Mock AI response to: '{prompt}' (using AI-Block MoE system)"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=gen_kwargs.get("temperature", 0.8),
                top_p=gen_kwargs.get("top_p", 0.95),
            )
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"⚠️  Generation failed: {e}")
            return f"AI-Block MoE response to: '{prompt}' [Note: Using fallback due to model error]"
    
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
        return "This is a mock response from AI-Block MoE blockchain system."


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