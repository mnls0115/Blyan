#!/usr/bin/env python
"""Test chain initialization."""

from pathlib import Path
from backend.core.chain import Chain

print("1. Importing Chain class... ✓")

print("2. Creating Path object...")
root_dir = Path("./data")
print(f"   Root dir: {root_dir}")

print("3. Initializing meta chain...")
meta_chain = Chain(root_dir, "A")
print("   Meta chain initialized ✓")

print("4. Initializing param chain...")
param_chain = Chain(root_dir, "B")
print("   Param chain initialized ✓")

print("5. Counting meta blocks...")
meta_blocks = list(meta_chain.storage.iter_blocks())
print(f"   Meta blocks: {len(meta_blocks)}")

print("6. Counting param blocks...")
param_blocks = list(param_chain.storage.iter_blocks())
print(f"   Param blocks: {len(param_blocks)}")

print("✅ All tests passed!")