# Weight Mapping Validation Runbook

## Quick Validation Steps

### 1. Enable Diagnostic Mode on GPU Node

```bash
# SSH to your RunPod instance
cd /workspace/blyan  # or your repo directory

# Pull latest changes
git pull

# Enable diagnostic logging and restart GPU node
export DIAG_MODEL_LOAD=1
python run_gpu_node.py
```

### 2. Check Diagnostic Logs

Look for the following in the logs:

```
[DIAG] Final load summary:
[DIAG]   Loaded tensors: XXX
[DIAG]   Missing keys: 0      # Should be 0 or very small
[DIAG]   Unexpected keys: 0   # Should be 0 or very small
```

**Success Indicators:**
- ✅ Missing keys: 0 (or < 10 for non-critical buffers)
- ✅ Unexpected keys: 0 (or < 10 for optional components)
- ✅ No "[DIAG] Missing:" entries for critical components (embedding, layers, lm_head)

**Problem Indicators:**
- ❌ Large number of missing keys (> 100)
- ❌ Missing critical components like "model.embed_tokens.weight" or "lm_head.weight"
- ❌ Unexpected keys suggesting wrong model structure

### 3. Deterministic Generation Test

Test with zero temperature for reproducible output:

```bash
# From local machine or another terminal
curl -X POST https://blyan.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2?",
    "max_new_tokens": 16,
    "temperature": 0.0,
    "top_p": 1.0
  }'
```

**Expected output:** Should be coherent text like "2+2 equals 4" or "The answer is 4"

**Gibberish indicators:**
- Repetitive characters: "aaaaaaa" or "!!!!!!!!"
- Random symbols: "�������"
- Nonsensical text: "xyz xyz xyz"

### 4. Layer-Specific Diagnostics

If still seeing issues, check specific layer mappings:

```bash
# On the GPU node
python scripts/check_weight_mapping.py
```

Focus on these critical layers:
- `embedding` → Should map to `model.embed_tokens`
- `layer_0` → Should map to `model.layers.0`
- `lm_head` → Should stay as `lm_head`
- `model_norm` → Should map to `model.norm`

### 5. Advanced Debugging

If mappings look correct but output is still wrong:

```bash
# Check actual tensor shapes in diagnostic mode
DIAG_MODEL_LOAD=1 python -c "
from pathlib import Path
from backend.model.manager import UnifiedModelManager

manager = UnifiedModelManager(root_dir=Path('./data'))
manager.load_from_blockchain()

# This will show all mapping translations
"
```

Look for:
- `[DIAG] Mapping layer_0.self_attn.q_proj.weight -> model.layers.0.self_attn.q_proj.weight`
- Similar translations for all layers

### 6. Quick Fixes

If you find mapping issues:

1. **Wrong layer names in param_index:**
   - Check `data/param_index.json` 
   - Layer names should be: `embedding`, `layer_0`, `layer_1`, ..., `lm_head`, `model_norm`

2. **Missing translations:**
   - Check `backend/model/manager.py` has `_translate_layer_prefix()` method
   - Verify `_build_tensor_key()` is being called in both loaders

3. **Clear caches and retry:**
   ```bash
   rm -rf data/models/fused/*.safetensors  # Clear snapshots
   rm -rf data/gpu_cache/*                 # Clear GPU cache
   python run_gpu_node.py                  # Restart
   ```

## Summary Checklist

- [ ] Diagnostic mode enabled (`DIAG_MODEL_LOAD=1`)
- [ ] Missing keys count is 0 or very small
- [ ] Unexpected keys count is 0 or very small
- [ ] Deterministic test produces coherent text
- [ ] Critical layers (embedding, layer_0, lm_head) are mapped correctly
- [ ] No repetitive or gibberish output

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Many missing keys | Layer prefix translation not working - check `_translate_layer_prefix()` |
| Unexpected keys | Wrong model structure - verify using Qwen3-8B |
| Gibberish output | Weight mapping mismatch - run `check_weight_mapping.py` |
| No logs showing | Ensure `DIAG_MODEL_LOAD=1` is exported |