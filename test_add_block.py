#!/usr/bin/env python
"""Test adding a single block."""

import torch
import io
from pathlib import Path
from backend.core.chain import Chain

print("1. Setting up chains...")
root_dir = Path("./data")
param_chain = Chain(root_dir, "B")
print("✓ Chains ready")

print("2. Creating synthetic tensors...")
dummy_tensors = {
    'weight': torch.randn(32, 16),
    'bias': torch.randn(32)
}
print("✓ Tensors created")

print("3. Serializing tensors...")
buffer = io.BytesIO()
torch.save(dummy_tensors, buffer)
tensor_bytes = buffer.getvalue()
print(f"✓ Serialized to {len(tensor_bytes)} bytes")

print("4. Adding block to chain...")
try:
    block = param_chain.add_block(
        payload=tensor_bytes,
        block_type='expert',
        expert_name='test.expert0',
        layer_id='test',
        depends_on=[],
        miner_pub='alice_test'
    )
    print(f"✓ Block added: {block.compute_hash()[:16]}...")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("5. Verifying chain state...")
blocks = list(param_chain.storage.iter_blocks())
print(f"✓ Chain has {len(blocks)} blocks")

print("✅ Test completed!")