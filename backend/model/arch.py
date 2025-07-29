from __future__ import annotations

from typing import Optional
import io

# Third-party libraries are optional at runtime; silence type checkers if missing
import torch  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


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
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def apply_weights(self, state_dict: dict):
        """Load a state_dict into the model."""
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)  # Ensure model is on the correct device after loading

    # ------------------------------------------------------------------
    # Generation helper
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 64, **gen_kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=gen_kwargs.get("temperature", 0.8),
            top_p=gen_kwargs.get("top_p", 0.95),
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True) 