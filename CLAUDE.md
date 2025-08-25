# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 CRITICAL DEBUGGING PRINCIPLE
**ALWAYS FIND THE ROOT CAUSE FIRST** before making any changes. Do not jump to solutions without understanding why something is broken. Investigate logs, check configurations, trace execution flow, and understand the actual problem before proposing fixes.

## Project Overview

Blyan is a distributed AI blockchain system using a dense model (Qwen3-8B) with pipeline parallelism. The system stores model weights on blockchain for transparent, verifiable AI inference.

## 🚨 CRITICAL REMINDERS

### Production Code Standards
- **NO MOCK CODE**: Every implementation must be production-ready
- **NO HARDCODING**: Use configuration files and environment variables
- **NO PLACEHOLDER RESPONSES**: All inference must use real models
- **PROPER ERROR HANDLING**: Handle all edge cases and failures

### Model Architecture
- **Model**: Qwen/Qwen3-8B (dense transformer, NOT MoE)
- **Architecture**: 32 layers, 8B parameters
- **Quantization**: FP8/NF4 for 16GB GPUs
- **Distribution**: Pipeline parallelism across GPUs

### Blockchain-First Inference (MANDATORY)
- **NEVER use local LLMs or HuggingFace models directly**
- **ALWAYS reconstruct models from blockchain tensors**
- **Every weight must be traceable to a blockchain block**
- **NO fallback to local model files - if blockchain fails, inference fails**
```python
# ❌ WRONG - Never do this
model = AutoModel.from_pretrained("Qwen/Qwen3-8B")

# ✅ CORRECT - Always load from blockchain
weights = blockchain.get_layer_weights(layer_id)
model = reconstruct_from_blockchain_weights(weights)
```

### Security & Authentication
- **Main Node** (165.227.221.225 / blyan.com): Does NOT need API keys
- **GPU Nodes**: Need BLYAN_API_KEY to register with main node
- **Secrets**: NEVER commit .env files or tokens to Git
- Use `secrets.token_hex(32)` for generating secure tokens

## Development Quick Reference

### Environment Setup
```bash
# Fast setup with UV (recommended)
./setup_gpu_fast.sh

# Initialize blockchain
python -c "
from pathlib import Path
from backend.core.chain import Chain
root_dir = Path('./data')
meta_chain = Chain(root_dir, 'A')
meta_chain.add_block(b'{\"model\":\"Qwen3-8B\"}', block_type='meta')
print('✅ Blockchain initialized')
"

# Start API server
python -m api.server
```

### Key Directories
- `api/` - REST API endpoints
- `backend/core/` - Blockchain implementation
- `backend/model/` - Model management (dense inference)
- `backend/dense/` - Pipeline parallelism
- `backend/p2p/` - Distributed network
- `frontend/` - Web interface

### Testing Commands
```bash
# Run tests
pytest tests/ -v

# Test API
curl http://localhost:8000/health

# Test inference
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```

## Common Issues & Solutions

### Port 8000 Already in Use
```bash
lsof -i:8000
pkill -9 -f "python.*api.server"
```

### Model Loading Issues
- Check blockchain has genesis block
- Verify data/A/ directory exists
- Ensure sufficient GPU memory

### Authentication Errors
- Main node: Should NOT use API keys
- GPU nodes: Must have valid BLYAN_API_KEY
- Check /health endpoint (no auth required)

## Important Files
- `requirements.txt` - Python dependencies
- `config/tokenomics.yaml` - Economic parameters
- `.env` - Environment variables (NEVER commit)
- `data/` - Blockchain storage

## Documentation References
- [API Documentation](API_DOCS.md)
- [Technical Specification](TECHNICAL_SPEC.md)
- [Architecture](ARCHITECTURE.md)
- [User Guide](USER_GUIDE.md)

---
*For folder-specific guidance, check CLAUDE.md in each directory.*