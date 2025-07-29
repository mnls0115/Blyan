# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-Block is a revolutionary distributed MoE (Mixture-of-Experts) blockchain system that hosts evolving AI models using DAG (Directed Acyclic Graph) structure. Each expert is stored as an independent block, enabling selective inference, partial mining, and distributed computing. The system has evolved into a **self-learning, evolving AI organism** rather than static data storage.

## Development Setup

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Initialize Genesis Block
Before first run, create the meta-chain genesis block with MoE specification:
```bash
python - <<'PY'
import json
from pathlib import Path
from backend.core.chain import Chain

root_dir = Path("./data")
meta_chain = Chain(root_dir, "A")
spec = {
    "model_name": "distilbert-base-uncased",
    "architecture": "mixture-of-experts", 
    "num_layers": 4,
    "num_experts": 8,
    "routing_strategy": "top2"
}
meta_chain.add_block(json.dumps(spec).encode(), block_type='meta')
print("✅ Meta chain initialized with MoE architecture.")
PY
```

**Note**: DAG validation is temporarily optimized for performance with large expert uploads. Cross-chain dependencies between meta-chain (A) and parameter-chain (B) have been resolved.

### Running the Application
- **API Server**: `uvicorn api.server:app --reload` (runs on http://127.0.0.1:8000)
- **Frontend**: Open `frontend/index.html` directly in browser (no build step required)

## Architecture

### Core Components
- **Chain (`backend/core/chain.py`)**: DAG blockchain with cycle detection and topological sorting
- **Block (`backend/core/block.py`)**: DAG block structure with expert metadata and dependencies
- **MoEModelManager (`backend/model/moe_infer.py`)**: Selective expert loading and MoE inference
- **DistributedInferenceCoordinator (`backend/p2p/distributed_inference.py`)**: P2P expert coordination
- **ExpertUsageTracker**: Real-time expert performance and reward tracking
- **API Server (`api/server.py`)**: Comprehensive REST API with MoE and distributed inference

### DAG Blockchain Structure
- **Meta-chain (A)**: Stores model architecture and routing rules (`block_type: 'meta'`)
- **Parameter-chain (B)**: Stores individual expert weights as DAG blocks (`block_type: 'expert'`, `'router'`)
- **DAG Dependencies**: Blocks have `depends_on` field enabling parallel expert evolution
- **Block Types**: `meta`, `expert`, `router` for different AI components
- **Cycle Detection**: `has_cycle()` and `topological_sort()` ensure DAG validity

### MoE Data Flow
1. Meta-chain defines MoE model architecture and expert configuration
2. Each expert stored as independent block with `expert_name` and `layer_id`
3. **Selective Loading**: MoEModelManager loads only required experts for inference
4. **Router Logic**: Router blocks define expert selection strategies
5. **Usage Tracking**: ExpertUsageTracker records performance for dynamic rewards
6. **Distributed Inference**: P2P coordinator distributes experts across nodes

### Enhanced File Structure
- `backend/core/`: DAG blockchain with cycle detection and expert filtering
- `backend/model/`: MoE inference with selective expert loading and usage tracking
- `backend/p2p/`: Distributed inference coordination and expert node management
- `miner/`: MoE model extraction and expert-specific upload tools
- `scripts/`: Testing and demonstration tools for distributed MoE

### Advanced Mining & Inference
- **MoE Expert Upload**: `upload_moe_parameters.py` extracts and uploads individual experts (fixed cross-chain dependencies)
- **Selective Inference**: Load only required experts based on routing decisions
- **Distributed Computing**: Experts run on specialized nodes with load balancing and heartbeat monitoring
- **Quality-Based Rewards**: Dynamic reward calculation based on usage, speed, and quality with ExpertUsageTracker
- **Expert Evolution**: Independent expert improvement through DAG versioning
- **Performance Optimization**: DAG validation optimized for large tensor uploads
- **P2P Node Registry**: Complete distributed inference coordination with node registration/discovery

## Key Commands for Development

### MoE Model Management
```bash
# Extract and upload MoE model (requires candidate-loss parameter)
python miner/upload_moe_parameters.py --address alice --model-file model.pt --meta-hash <full-meta-hash> --candidate-loss 0.8

# Get correct meta hash
curl -s http://127.0.0.1:8000/chain/A/blocks | grep -o '"hash":"[^"]*"' | head -1

# Test MoE extraction (dry-run)
python miner/upload_moe_parameters.py --address alice --model-file model.pt --meta-hash <hash> --candidate-loss 0.8 --dry-run

# Run distributed demo
python scripts/demo_distributed_moe.py
```

### API Testing
```bash
# MoE inference
curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" \
  -d '{"prompt": "test", "use_moe": true, "top_k_experts": 2}'

# Distributed inference  
curl -X POST "http://127.0.0.1:8000/chat/distributed" -H "Content-Type: application/json" \
  -d '{"prompt": "test", "top_k_experts": 3}'

# Expert analytics
curl -X GET "http://127.0.0.1:8000/experts/top?limit=5"
curl -X GET "http://127.0.0.1:8000/experts/stats/layer0.expert1"
```

### P2P Network
```bash
# Start expert nodes
python -m backend.p2p.distributed_inference server node1 8001
python -m backend.p2p.distributed_inference server node2 8002

# Register expert node
curl -X POST "http://127.0.0.1:8000/p2p/register" -H "Content-Type: application/json" \
  -d '{"node_id": "node1", "host": "localhost", "port": 8001, "available_experts": ["layer0.expert0"]}'

# List registered nodes
curl -X GET "http://127.0.0.1:8000/p2p/nodes"

# Unregister node
curl -X DELETE "http://127.0.0.1:8000/p2p/nodes/node1"
```

## System Capabilities

### Achieved Milestones (All 5 Core Features Complete + Performance Optimizations)
1. ✅ **Selective Inference**: Load only required experts for specific queries
2. ✅ **Partial Mining**: Contributors can improve individual experts independently  
3. ✅ **Expert Evolution**: Independent expert improvement and specialization via DAG
4. ✅ **Distributed Computing**: Experts run on different nodes with P2P coordination and heartbeat monitoring
5. ✅ **Quality-Based Rewards**: Dynamic rewards based on expert usage and performance with persistent tracking
6. ✅ **Upload Stability**: Fixed cross-chain dependency issues and DAG validation performance
7. ✅ **P2P Infrastructure**: Complete node registry with registration, discovery, and load balancing

### AI Life Form Characteristics
- **🔄 Autonomous Evolution**: Expert-level independent performance improvement
- **🤝 Distributed Cooperation**: P2P expert sharing and load balancing
- **📈 Continuous Learning**: Real-time performance monitoring and adaptive routing
- **🧬 Organic Growth**: DAG structure enables parallel expert development
- **💰 Economic Incentives**: Usage-based automatic reward distribution