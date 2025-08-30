#!/usr/bin/env python3
"""
Check Weight Mapping Script
Diagnoses weight mapping issues between param_index and Qwen model structure.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def load_param_index(data_dir: Path) -> Dict:
    """Load param_index.json file."""
    param_index_path = data_dir / "param_index.json"
    if not param_index_path.exists():
        print(f"❌ No param_index.json found at {param_index_path}")
        return {}
    
    with open(param_index_path, 'r') as f:
        return json.load(f)

def get_expected_qwen_keys() -> List[Tuple[str, str]]:
    """Get expected Qwen3-8B model structure."""
    expected = []
    
    # Embedding layer
    expected.append(("model.embed_tokens.weight", "embedding"))
    
    # 32 transformer layers
    for i in range(32):
        layer_prefix = f"model.layers.{i}"
        param_prefix = f"layer_{i}"
        
        # Self-attention
        expected.extend([
            (f"{layer_prefix}.self_attn.q_proj.weight", f"{param_prefix}.self_attn.q_proj"),
            (f"{layer_prefix}.self_attn.k_proj.weight", f"{param_prefix}.self_attn.k_proj"),
            (f"{layer_prefix}.self_attn.v_proj.weight", f"{param_prefix}.self_attn.v_proj"),
            (f"{layer_prefix}.self_attn.o_proj.weight", f"{param_prefix}.self_attn.o_proj"),
        ])
        
        # MLP
        expected.extend([
            (f"{layer_prefix}.mlp.gate_proj.weight", f"{param_prefix}.mlp.gate_proj"),
            (f"{layer_prefix}.mlp.up_proj.weight", f"{param_prefix}.mlp.up_proj"),
            (f"{layer_prefix}.mlp.down_proj.weight", f"{param_prefix}.mlp.down_proj"),
        ])
        
        # Layer norms
        expected.extend([
            (f"{layer_prefix}.input_layernorm.weight", f"{param_prefix}.input_layernorm"),
            (f"{layer_prefix}.post_attention_layernorm.weight", f"{param_prefix}.post_attention_layernorm"),
        ])
    
    # Final norm and output
    expected.append(("model.norm.weight", "model_norm"))
    expected.append(("lm_head.weight", "lm_head"))
    
    return expected

def check_mapping(param_index: Dict) -> None:
    """Check weight mapping between param_index and expected structure."""
    expected_keys = get_expected_qwen_keys()
    
    print("\n📊 Weight Mapping Analysis")
    print("=" * 60)
    
    # Check what's in param_index
    param_layers = list(param_index.keys())
    print(f"\n📦 Param Index Contents:")
    print(f"  Total entries: {len(param_layers)}")
    
    if param_layers:
        print(f"  Sample entries:")
        for layer in param_layers[:5]:
            print(f"    - {layer} -> block {param_index[layer]}")
        if len(param_layers) > 5:
            print(f"    ... and {len(param_layers) - 5} more")
    
    # Check mapping
    print(f"\n🔍 Checking Expected Mappings:")
    missing_count = 0
    found_count = 0
    
    for qwen_key, param_key in expected_keys:
        if param_key in param_index:
            found_count += 1
            if found_count <= 3:  # Show first few matches
                print(f"  ✅ {param_key} -> {qwen_key}")
        else:
            missing_count += 1
            if missing_count <= 5:  # Show first few missing
                print(f"  ❌ Missing: {param_key} (needs {qwen_key})")
    
    if missing_count > 5:
        print(f"  ... and {missing_count - 5} more missing")
    
    # Summary
    print(f"\n📈 Summary:")
    print(f"  Expected keys: {len(expected_keys)}")
    print(f"  Found: {found_count}")
    print(f"  Missing: {missing_count}")
    print(f"  Coverage: {found_count/len(expected_keys)*100:.1f}%")
    
    # Check for unexpected keys
    expected_param_keys = {pk for _, pk in expected_keys}
    unexpected = set(param_layers) - expected_param_keys
    if unexpected:
        print(f"\n⚠️ Unexpected keys in param_index:")
        for key in list(unexpected)[:5]:
            print(f"    - {key}")
        if len(unexpected) > 5:
            print(f"    ... and {len(unexpected) - 5} more")

def main():
    """Main function."""
    data_dir = Path("./data")
    
    print("🔍 Weight Mapping Diagnostic Tool")
    print("=" * 60)
    
    # Load param_index
    param_index = load_param_index(data_dir)
    
    if not param_index:
        print("\n⚠️ No param_index to check. Run model upload first.")
        sys.exit(1)
    
    # Check mapping
    check_mapping(param_index)
    
    print("\n✅ Diagnostic complete")

if __name__ == "__main__":
    main()