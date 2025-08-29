#!/usr/bin/env python3
"""Check what keys are in the Qwen3-8B model to identify missing components."""

from transformers import AutoModelForCausalLM
import torch

# Load model structure (weights can be random, we just need keys)
print("Loading Qwen3-8B model structure...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# Get all keys
state_dict_keys = list(model.state_dict().keys())
print(f"\nTotal keys in model: {len(state_dict_keys)}")

# Categorize keys
embedding_keys = []
layer_keys = []
lm_head_keys = []
norm_keys = []
other_keys = []

for key in state_dict_keys:
    if 'embed_tokens' in key:
        embedding_keys.append(key)
    elif 'layers.' in key:
        layer_keys.append(key)
    elif 'lm_head' in key:
        lm_head_keys.append(key)
    elif 'norm' in key:
        norm_keys.append(key)
    else:
        other_keys.append(key)

print(f"\nKey categories:")
print(f"  Embedding keys ({len(embedding_keys)}): {embedding_keys[:3]}...")
print(f"  Layer keys ({len(layer_keys)}): {layer_keys[:3]}...")
print(f"  LM head keys ({len(lm_head_keys)}): {lm_head_keys}")
print(f"  Norm keys ({len(norm_keys)}): {norm_keys}")
print(f"  OTHER KEYS ({len(other_keys)}): {other_keys}")

print("\n⚠️ MISSING FROM UPLOAD:")
print(f"  - Norm keys: {norm_keys}")
print(f"  - Other keys: {other_keys}")

# Check for rotary embeddings
rotary_keys = [k for k in state_dict_keys if 'rotary' in k.lower() or 'rope' in k.lower()]
if rotary_keys:
    print(f"\n  - Rotary/RoPE keys: {rotary_keys}")

print("\nThese missing keys are why the model produces garbled output!")