#!/usr/bin/env python
"""Simple test to upload one expert block directly."""

import torch
import json
import base64
from pathlib import Path
from backend.core.chain import Chain

# Get meta hash
root_dir = Path("./data")
meta_chain = Chain(root_dir, "A")
param_chain = Chain(root_dir, "B")

# Get the meta block hash
meta_blocks = list(meta_chain.storage.iter_blocks())
if not meta_blocks:
    print("❌ No meta blocks found")
    exit(1)

meta_hash = meta_blocks[0].compute_hash()
print(f"✓ Using meta hash: {meta_hash}")

# Create a simple synthetic expert
dummy_tensors = {
    'weight': torch.randn(128, 64),
    'bias': torch.randn(128)
}

# Serialize tensors
import io
buffer = io.BytesIO()
torch.save(dummy_tensors, buffer)
tensor_bytes = buffer.getvalue()

print(f"✓ Created synthetic expert with {len(tensor_bytes)} bytes")

# Try to add directly to param chain
try:
    block = param_chain.add_block(
        payload=tensor_bytes,
        block_type='expert',
        expert_name='test.expert0',
        layer_id='test',
        depends_on=[],
        miner_pub='alice_pub'
    )
    print(f"✓ Successfully added block: {block.compute_hash()}")
    
    # Verify it exists
    param_blocks = list(param_chain.storage.iter_blocks())
    print(f"✓ Parameter chain now has {len(param_blocks)} blocks")
    
except Exception as e:
    print(f"❌ Failed to add block: {e}")
    import traceback
    traceback.print_exc()