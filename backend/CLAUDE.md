# Backend Module Guidelines

## Overview
Core backend implementation for blockchain, model management, and distributed systems.

## Subdirectories

### backend/core/
- `chain.py` - DAG blockchain implementation
- `block.py` - Block structure and validation
- `storage.py` - Persistent storage
- `reward_engine.py` - BLY token rewards

### backend/model/
- `manager.py` - Unified model management
- `dense_inference.py` - Dense model inference (NOT MoE!)
- `chunked_blockchain_loader.py` - Load model layers from blockchain

### backend/dense/
- `partition_planner.py` - Pipeline parallelism partitioning
- `delta_format.py` - Delta compression for updates
- `speculative_decoder.py` - Speculative decoding optimization

### backend/p2p/
- `distributed_inference.py` - P2P coordination
- `node_registry.py` - GPU node management
- `inference_queue.py` - Priority queue for requests

### backend/learning/
- `consensus_learning.py` - Distributed training
- `micro_step_trainer.py` - Non-blocking training
- `pipeline_parallel.py` - Pipeline parallel training

## Critical Reminders

### Production Code Standards
- **NO MOCK IMPLEMENTATIONS**: Every function must work with real data
- **NO HARDCODED VALUES**: Use config files or environment variables
- **NO FALLBACK TO LOCAL MODELS**: Blockchain is the only source of truth

### Blockchain-Only Model Loading (MANDATORY)
```python
# ❌ ABSOLUTELY FORBIDDEN - Never use local models
model = AutoModel.from_pretrained("Qwen3-8B")
response = "This is a mock response"  # NEVER DO THIS

# ✅ REQUIRED - Always load from blockchain
from backend.core.chain import Chain
from backend.model.manager import UnifiedModelManager

chain = Chain(data_dir, "B")  # Parameter chain
layer_weights = chain.get_block_by_type("layer", layer_id).data
tensor = deserialize_tensor(layer_weights)

# If blockchain fails, propagate the error
if not tensor:
    raise RuntimeError("Cannot proceed without blockchain weights")
```

### Dense Model (NOT MoE)
- Qwen3-8B is a dense 32-layer model
- Use pipeline parallelism, not expert routing
- Partition layers across GPUs

### Security
- Never expose node secrets
- Validate all blockchain blocks
- Use cryptographic hashes for integrity

## Common Patterns

### Add Block to Chain
```python
chain = Chain(data_dir, chain_id)
block_hash = chain.add_block(
    data=weight_bytes,
    block_type="delta",
    metadata={"layer": 0}
)
```

### Register GPU Node
```python
coordinator = DistributedInferenceCoordinator()
await coordinator.register_node({
    "node_id": "gpu-001",
    "host": "10.0.0.1",
    "port": 8001,
    "layers": [0, 1, 2, 3]  # Layers this node handles
})
```