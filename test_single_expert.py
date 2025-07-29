#!/usr/bin/env python
"""Test uploading a single expert via API."""

import requests
import torch
import base64
import io
import json

# Create synthetic tensors
dummy_tensors = {
    'weight': torch.randn(64, 32),
    'bias': torch.randn(64)
}

# Serialize
buffer = io.BytesIO()
torch.save(dummy_tensors, buffer)
tensor_bytes = buffer.getvalue()
tensor_b64 = base64.b64encode(tensor_bytes).decode()

# Prepare request
payload = {
    "miner_address": "alice",
    "expert_name": "test.expert0", 
    "layer_id": "test",
    "block_type": "expert",
    "depends_on": [],  # No dependencies
    "tensor_data_b64": tensor_b64,
    "candidate_loss": 0.8,
    "previous_loss": None,
    "miner_pub": "dummy_pub",
    "payload_sig": "dummy_sig"
}

print("Sending request...")
try:
    response = requests.post(
        "http://127.0.0.1:8000/upload_moe_experts",
        json=payload,
        timeout=10
    )
    print(f"Response: {response.status_code}")
    print(f"Content: {response.text}")
except requests.exceptions.Timeout:
    print("❌ Request timed out")
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to server")
except Exception as e:
    print(f"❌ Error: {e}")