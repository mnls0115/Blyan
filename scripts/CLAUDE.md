# Scripts Module Guidelines

## Overview
Utility scripts for testing, deployment, and demonstrations.

## Key Scripts

### Testing & Demos
- `demo_distributed_moe.py` - Test distributed inference
- `test_inference_only.py` - Simple inference test
- `test_streaming.py` - Test streaming responses

### Deployment
- `setup_gpu_fast.sh` - Fast environment setup with UV
- `server.sh` - Start/stop/restart services
- `monitor_memory_vram.py` - GPU memory monitoring

### Model Management
- `extract_individual_experts.py` - Extract layers (outdated for MoE)
- `build_and_ship_chain.py` - Build blockchain with model

## Usage Examples

### Test Inference
```bash
python scripts/test_inference_only.py
```

### Monitor GPU Memory
```bash
python scripts/monitor_memory_vram.py --interval 5
```

### Start Services
```bash
./server.sh start       # Start all
./server.sh start api   # API only
./server.sh status      # Check status
./server.sh restart     # Restart all
```

## Critical Reminders

### Production Code Standards
- **NO MOCK SCRIPTS**: Every script must work with real blockchain data
- **NO TEST RESPONSES**: Never generate fake outputs
- **USE REAL MODELS**: Always load from blockchain, never from files

### Model Type
- Scripts may reference old MoE architecture
- Current model is Qwen3-8B (dense, not MoE)
- Use pipeline parallelism, not expert extraction

### Blockchain-Only Inference
```python
# ❌ FORBIDDEN in any script
def test_inference():
    return "Mock response for testing"  # NEVER!

# ✅ REQUIRED - Real blockchain inference
def test_inference():
    from backend.core.chain import Chain
    from backend.model.manager import UnifiedModelManager
    
    # Load from blockchain
    chain = Chain(Path("./data"), "B")
    manager = UnifiedModelManager()
    manager.load_from_blockchain(chain)
    
    # Generate real response
    response = manager.generate("test prompt", max_tokens=50)
    return response
```

### Environment
- Always activate venv before running
- Check GPU availability with nvidia-smi
- Ensure blockchain is initialized

### Security
- Don't hardcode API keys in scripts
- Use environment variables for secrets
- Clean up test data after demos

## Common Patterns

### Load Model from Blockchain
```python
from backend.core.chain import Chain
from backend.model.manager import UnifiedModelManager

chain = Chain(Path("./data"), "A")
manager = UnifiedModelManager()
manager.load_from_blockchain(chain)
```

### Test API Endpoint
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"prompt": "test", "max_tokens": 50}
)
print(response.json())