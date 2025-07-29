# 🧪 AI-Block MoE Testing Guide

Complete guide for testing the MoE blockchain system's end-to-end functionality.

## 🎯 Testing Objectives

We're validating the complete flow:
**MoE Model Upload → Expert Block Splitting → Inference Request → Router-based Expert Selection → Selective Inference**

## 🚀 Quick Start Testing

### Option 1: Full End-to-End Test
```bash
# Complete automated test (recommended)
python scripts/demo_full_moe_flow.py
```

### Option 2: Inference Testing Only
```bash
# Test inference assuming blockchain is set up
python scripts/test_inference_only.py

# Quick debug test
python scripts/test_inference_only.py debug

# Check expert status
python scripts/test_inference_only.py status
```

### Option 3: Manual Step-by-Step

#### Step 1: Start API Server
```bash
uvicorn api.server:app --reload
```

#### Step 2: Initialize Blockchain
```bash
python - <<'PY'
import json
from pathlib import Path
from backend.core.chain import Chain

root_dir = Path("./data")
meta_chain = Chain(root_dir, "A")
spec = {
    "model_name": "mock-moe-model",
    "architecture": "mixture-of-experts",
    "num_layers": 3,
    "num_experts": 4,
    "routing_strategy": "top2"
}
meta_chain.add_block(json.dumps(spec).encode(), block_type='meta')
print("✅ Meta chain initialized.")
PY
```

#### Step 3: Create Mock MoE Model
```bash
python scripts/demo_full_moe_flow.py  # Will create mock model automatically
```

#### Step 4: Upload Experts
```bash
# Get meta hash first
curl -X GET "http://127.0.0.1:8000/chain/A/blocks?limit=1"

# Upload with actual meta hash
python miner/upload_moe_parameters.py \
  --address test_miner \
  --model-file test_data/mock_moe_model.pt \
  --meta-hash YOUR_META_HASH_HERE \
  --candidate-loss 0.85
```

#### Step 5: Test Inference
```bash
# MoE inference
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "use_moe": true, "top_k_experts": 2}'

# Check expert usage
curl -X GET "http://127.0.0.1:8000/experts/top?limit=5"
```

## ✅ What to Look For

### Success Indicators

1. **Expert Extraction**:
   ```
   ✓ Extracted 12 experts, 3 routers, 15 base parameters
   ✓ Expert layer0.expert0 uploaded: abc123def...
   ```

2. **Selective Loading**:
   ```
   ✓ Loaded expert layer0.expert1 (0.123s)
   ✓ Loaded expert layer1.expert0 (0.087s)
   ```

3. **Inference Response**:
   ```json
   {
     "response": "Machine learning is...",
     "expert_usage": {
       "layer0.expert1": 0.123,
       "layer1.expert0": 0.087
     },
     "inference_time": 0.534
   }
   ```

4. **Expert Analytics**:
   ```
   - layer0.expert1: 5 calls, 0.123s avg, 1.25x reward
   - layer1.expert0: 3 calls, 0.087s avg, 1.45x reward
   ```

### Failure Scenarios & Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **API Server Not Running** | Connection refused | `uvicorn api.server:app --reload` |
| **Missing Dependencies** | Import errors | `pip install -r requirements.txt` |
| **No Meta Chain** | "Meta chain empty" | Initialize genesis block |
| **Expert Upload Fails** | Signature verification error | Use valid meta hash and fix upload script |
| **No Expert Usage** | Empty expert_usage dict | Check if experts were uploaded correctly |
| **Slow Inference** | >5s response time | Use smaller model or CPU-only mode |

## 🔧 Debugging Tips

### Enable Verbose Logging
Add to API server startup:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check System State
```bash
# Blockchain state
curl -X GET "http://127.0.0.1:8000/chain/A/blocks"  # Meta chain
curl -X GET "http://127.0.0.1:8000/chain/B/blocks"  # Parameter chain

# Expert analytics
curl -X GET "http://127.0.0.1:8000/experts/top"

# P2P network
curl -X GET "http://127.0.0.1:8000/p2p/nodes"
```

### Common Error Messages

- **"No expert nodes available"**: Start P2P nodes or register mock nodes
- **"Expert not found"**: Check if expert upload completed successfully
- **"Invalid base64 tensor data"**: Signature verification failed
- **"PoL failed"**: Candidate loss not better than previous
- **"DAG cycle detected"**: Fix depends_on relationships

## 🎯 Success Criteria

For the system to be considered **fully operational**:

1. ✅ **Model Loading**: MoE model successfully extracted into experts
2. ✅ **Block Creation**: Expert blocks stored with proper DAG dependencies
3. ✅ **Selective Inference**: Only required experts loaded for each query
4. ✅ **Router Logic**: Router successfully selects appropriate experts
5. ✅ **Performance Tracking**: Expert usage recorded and rewards calculated
6. ✅ **Distributed Computing**: P2P expert nodes can be registered and used

## 🌟 Next Steps After Success

Once all tests pass:

1. **Real MoE Models**: Try with actual HuggingFace MoE models
2. **Distributed Setup**: Set up multiple expert nodes on different machines
3. **Performance Optimization**: Benchmark selective vs full model loading
4. **Economic System**: Test dynamic reward calculations
5. **Governance**: Implement expert quality voting mechanisms

## 📊 Performance Benchmarks

Expected performance on a typical system:

| Operation | Time | Notes |
|-----------|------|-------|
| Expert Extraction | 10-30s | Depends on model size |
| Expert Upload | 5-15s | Per expert block |
| Selective Inference | 0.1-2s | Much faster than full model |
| Expert Analytics | <0.5s | Real-time queries |

If your system significantly exceeds these times, consider:
- Using smaller models for testing
- Enabling GPU acceleration
- Reducing the number of experts per layer
- Using quantized models (int8/fp16)

---

🎉 **When all tests pass, you have successfully created the world's first self-learning AI blockchain organism!** 🌱✨