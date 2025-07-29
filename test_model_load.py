#!/usr/bin/env python
"""Test model loading and expert extraction."""

import torch
from pathlib import Path

print("1. Loading model...")
model_path = "test_data/microsoft_DialoGPT-small.pt"

try:
    state_dict = torch.load(model_path, map_location='cpu')
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    print(f"✓ Model loaded with {len(state_dict)} keys")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

print("2. Creating virtual experts...")
experts = {}
all_keys = list(state_dict.keys())
num_experts = 4

# Simple grouping
expert_size = len(all_keys) // num_experts
for i in range(num_experts):
    expert_name = f"layer{i//2}.expert{i%2}"
    start_idx = i * expert_size
    end_idx = start_idx + expert_size if i < num_experts-1 else len(all_keys)
    
    expert_tensors = {}
    for key in all_keys[start_idx:end_idx]:
        expert_tensors[key] = state_dict[key]
    
    if expert_tensors:
        experts[expert_name] = expert_tensors

print(f"✓ Created {len(experts)} virtual experts")

print("3. Testing tensor serialization...")
import io
for expert_name, tensors in list(experts.items())[:1]:  # Test first expert only
    buffer = io.BytesIO()
    torch.save(tensors, buffer)
    size = len(buffer.getvalue())
    print(f"✓ {expert_name}: {len(tensors)} tensors, {size} bytes")

print("✅ All model tests passed!")