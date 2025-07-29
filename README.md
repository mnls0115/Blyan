# AI-Block: Distributed MoE Blockchain

A revolutionary blockchain system that hosts evolving AI models using DAG (Directed Acyclic Graph) structure and Mixture-of-Experts (MoE) architecture. Each expert is stored as an independent block, enabling selective inference, partial mining, and distributed computing.

## How to Run

### 1. Setup Backend

First, set up a Python virtual environment and install the required packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Initialize Chains

Before running the server, you must create the first "genesis" block for the meta-chain. This block defines the MoE model architecture.

```bash
python - <<'PY'
import json
from pathlib import Path
from backend.core.chain import Chain

# This will create a ./data/A directory for the meta-chain
root_dir = Path("./data")
meta_chain = Chain(root_dir, "A")

# Initialize with MoE model specification
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

### 3. Run API Server

Now, you can start the FastAPI server.

```bash
uvicorn api.server:app --reload
```

The server will be available at `http://127.0.0.1:8000`.

### 4. Run Frontend

Open a new terminal. You don't need a build step for the frontend. Simply open the `frontend/index.html` file in your web browser.

- On macOS: `open frontend/index.html`
- On Linux: `xdg-open frontend/index.html`
- On Windows: `start frontend/index.html`

You can now chat with the AI through the web interface.

## 🧠 MoE Expert Management

### Upload MoE Experts

Upload individual experts to the DAG blockchain:

```bash
# Get the correct meta block hash first
curl -s http://127.0.0.1:8000/chain/A/blocks | grep -o '"hash":"[^"]*"' | head -1

# Upload MoE model with expert extraction (requires full hash)
python miner/upload_moe_parameters.py \
  --address your_wallet_address \
  --model-file path/to/moe_model.pt \
  --meta-hash <full-64-char-hash> \
  --candidate-loss 0.85

# Test extraction first (dry-run)
python miner/upload_moe_parameters.py \
  --address alice \
  --model-file model.pt \
  --meta-hash <hash> \
  --candidate-loss 0.8 \
  --dry-run

# Upload individual expert (after extraction)
# Note: Cross-chain dependencies removed for performance
curl -X POST "http://127.0.0.1:8000/upload_moe_experts" \
  -H "Content-Type: application/json" \
  -d '{
    "expert_name": "layer0.expert1",
    "layer_id": "layer0", 
    "block_type": "expert",
    "depends_on": [],
    "tensor_data_b64": "...",
    "candidate_loss": 0.85,
    "miner_address": "alice",
    "miner_pub": "...",
    "payload_sig": "..."
  }'
```

### Inference Modes

```bash
# Standard inference
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "use_moe": false}'

# MoE inference (selective expert loading)
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "use_moe": true, "top_k_experts": 2}'

# Distributed inference across nodes
curl -X POST "http://127.0.0.1:8000/chat/distributed" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "top_k_experts": 3}'
```

## 🌐 Distributed Expert Nodes

### Start Expert Nodes

Run expert nodes on different machines:

```bash
# Node 1: Serves layer 0-1 experts
python -m backend.p2p.distributed_inference server node1 8001

# Node 2: Serves layer 2-3 experts  
python -m backend.p2p.distributed_inference server node2 8002

# Node 3: Serves specialized experts
python -m backend.p2p.distributed_inference server node3 8003
```

### Register Expert Nodes

```bash
# Register node with coordinator
curl -X POST "http://127.0.0.1:8000/p2p/register" \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "node1",
    "host": "localhost", 
    "port": 8001,
    "available_experts": ["layer0.expert0", "layer0.expert1", "layer1.expert0"]
  }'

# Check registered nodes
curl -X GET "http://127.0.0.1:8000/p2p/nodes"

# Unregister node when shutting down
curl -X DELETE "http://127.0.0.1:8000/p2p/nodes/node1"
```

## 📊 Expert Analytics & Rewards

### Monitor Expert Performance

```bash
# Get expert usage statistics
curl -X GET "http://127.0.0.1:8000/experts/stats/layer0.expert1"

# Get top performing experts
curl -X GET "http://127.0.0.1:8000/experts/top?limit=10"

# Trigger expert reward calculation
curl -X POST "http://127.0.0.1:8000/experts/reward/layer0.expert1"
```

### Expert Evolution

- **Selective Inference**: Only required experts are loaded for each query
- **Partial Mining**: Contributors can improve individual experts independently  
- **Quality Rewards**: Dynamic rewards based on usage frequency, speed, and quality
- **Distributed Computing**: Experts run on specialized nodes across the network
- **DAG Evolution**: Expert blocks form dependency graphs allowing parallel development

## 🚀 Quick Demo

Run the complete demonstration:

```bash
# Run full distributed MoE demo
python scripts/demo_distributed_moe.py

# Test MoE extraction and DAG validation
python scripts/test_moe_upload.py
```

## 🏗️ Architecture

- **Meta Chain (A)**: Stores model architecture and routing rules
- **Parameter Chain (B)**: Stores individual expert weights as DAG blocks (optimized for large uploads)
- **DAG Structure**: Blocks have dependencies enabling parallel expert evolution (validation optimized)
- **Block Types**: `meta`, `expert`, `router` for different AI components
- **P2P Network**: Distributed expert nodes with load balancing and heartbeat monitoring
- **Usage Tracking**: Real-time expert performance and reward calculation with persistent storage
- **Performance Optimizations**: Cross-chain dependency resolution, DAG validation improvements

## 📁 Key Files

- `backend/core/block.py` - DAG block structure with expert metadata
- `backend/core/chain.py` - DAG blockchain with cycle detection (performance optimized)
- `backend/model/moe_infer.py` - MoE inference with selective loading and usage tracking
- `backend/p2p/distributed_inference.py` - P2P expert coordination with node registry
- `miner/upload_moe_parameters.py` - MoE model extraction and upload (dependency-fixed)
- `api/server.py` - REST API with all endpoints including P2P management

## ⚠️ Known Optimizations

- **Large Tensor Uploads**: DAG validation temporarily optimized for performance with large expert blocks
- **Cross-Chain Dependencies**: Removed to prevent validation cycles between meta-chain (A) and parameter-chain (B)
- **Memory Management**: Use smaller expert sizes for optimal performance in current implementation 