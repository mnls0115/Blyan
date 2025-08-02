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
    "model_name": "tiny_mistral_moe",  # Must match model in ./models/ directory
    "architecture": "mixture-of-experts", 
    "num_layers": 4,
    "num_experts": 8,
    "routing_strategy": "top2"
}
meta_chain.add_block(json.dumps(spec).encode(), block_type='meta')
print("✅ Meta chain initialized with MoE architecture.")
PY
```

**⚠️ Important**: The `model_name` must match a directory in `./models/` (e.g., `./models/tiny_mistral_moe/`)

**Note**: DAG validation is temporarily optimized for performance with large expert uploads. Cross-chain dependencies between meta-chain (A) and parameter-chain (B) have been resolved.

### Running the Application
- **Multi-Server**: `./server.sh start` (API + P2P nodes)
- **API Only**: `./server.sh start api` (runs on http://0.0.0.0:8000)
- **Frontend**: Open `frontend/index.html` directly in browser (no build step required)
- **Server Status**: `./server.sh status` to check all services
- **Restart**: `./server.sh restart` to apply code changes

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
- **Individual Expert Extraction**: `scripts/extract_individual_experts.py` separates MoE model into individual expert blocks for diverse routing
- **MoE Expert Upload**: `upload_moe_parameters.py` extracts and uploads MoE models (creates single unified expert block)
- **Selective Inference**: Load only required experts based on content-aware routing decisions
- **Content-Based Routing**: Router analyzes prompt characteristics to select optimal experts dynamically
- **Blockchain-First Generation**: Reconstruct models entirely from Expert block weights for true decentralized inference
- **Distributed Computing**: Experts run on specialized nodes with load balancing and heartbeat monitoring
- **Quality-Based Rewards**: Dynamic reward calculation based on usage, speed, and quality with ExpertUsageTracker
- **Expert Evolution**: Independent expert improvement through DAG versioning
- **Performance Optimization**: DAG validation optimized for large tensor uploads
- **P2P Node Registry**: Complete distributed inference coordination with node registration/discovery

## Critical Implementation Requirements

### ⚠️ **IMPORTANT: Blockchain-First Inference**
**All inference MUST use parameters from blockchain blocks, not base model weights**

The core principle of AI-Block is that inference should reconstruct the model using Expert blocks from the blockchain, ensuring:
- **Transparency**: Every weight used is traceable to a specific block
- **Decentralization**: No reliance on centralized model files
- **Evolution**: Model behavior evolves as new Expert blocks are added
- **Proof-of-Learning**: Only quality-validated Expert weights are used

Current implementation correctly loads Expert blocks (`✓ Loaded expert layer0.expert0`) but inference should reconstruct the full MoE model from blockchain state, not fall back to base model files.

### Production MoE Inference Flow
1. **Load Meta-chain**: Get model architecture and routing strategy
2. **Select Experts**: Use routing logic to choose required Expert blocks
3. **Reconstruct Model**: Build MoE model from Expert block weights only
4. **Generate**: Perform inference using blockchain-reconstructed model
5. **Track Usage**: Record Expert usage for reward calculation

## Key Commands for Development

### MoE Model Management
```bash
# Extract individual experts from MoE model to create diverse blockchain blocks
python3 scripts/extract_individual_experts.py

# Upload full MoE model (requires candidate-loss parameter) - creates single expert block
python miner/upload_moe_parameters.py --address alice --model-file ./models/tiny_mistral_moe --meta-hash <full-meta-hash> --candidate-loss 0.8

# Get correct meta hash
curl -s http://127.0.0.1:8000/chain/A/blocks | grep -o '"hash":"[^"]*"' | head -1

# Test MoE extraction (dry-run)
python miner/upload_moe_parameters.py --address alice --model-file ./models/tiny_mistral_moe --meta-hash <hash> --candidate-loss 0.8 --dry-run

# Run distributed demo
python scripts/demo_distributed_moe.py

# Demo Expert Group Optimization (NEW!)
python scripts/demo_expert_group_optimization.py

# Demo Security Verification System (NEW!)
python scripts/demo_security_verification.py
```

### API Testing
```bash
# MoE inference
curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" \
  -d '{"prompt": "test", "use_moe": true, "top_k_experts": 2}'

# Distributed inference  
curl -X POST "http://127.0.0.1:8000/chat/distributed" -H "Content-Type: application/json" \
  -d '{"prompt": "test", "top_k_experts": 3}'

# Optimized distributed inference with expert groups (NEW!)
curl -X POST "http://127.0.0.1:8000/chat/distributed_optimized" -H "Content-Type: application/json" \
  -d '{"prompt": "test", "required_experts": ["layer0.expert0", "layer1.expert1"], "preferred_region": "us-west"}'

# Secure distributed inference with integrity verification and automatic failover (NEW!)
curl -X POST "http://127.0.0.1:8000/chat/distributed_secure" -H "Content-Type: application/json" \
  -d '{"prompt": "test", "required_experts": ["layer0.expert0", "layer1.expert1"], "enable_integrity_check": true}'

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

# Register optimized expert node with expert groups (NEW!)
curl -X POST "http://127.0.0.1:8000/p2p/register_optimized" -H "Content-Type: application/json" \
  -d '{"node_id": "opt_node1", "host": "localhost", "port": 8001, "available_experts": ["layer0.expert0", "layer1.expert1"], "expert_groups": [{"experts": ["layer0.expert0", "layer1.expert1"], "usage_count": 10}], "region": "us-west"}'

# List registered nodes
curl -X GET "http://127.0.0.1:8000/p2p/nodes"

# Unregister node
curl -X DELETE "http://127.0.0.1:8000/p2p/nodes/node1"

# Get expert group insights (NEW!)
curl -X GET "http://127.0.0.1:8000/p2p/expert_groups"

# Get optimization performance insights (NEW!)
curl -X GET "http://127.0.0.1:8000/p2p/optimization_insights"

# Get replication suggestions (NEW!)
curl -X GET "http://127.0.0.1:8000/p2p/replication_suggestions"

# Get security integrity status (NEW!)
curl -X GET "http://127.0.0.1:8000/security/integrity_status"

# Get comprehensive security dashboard (NEW!)
curl -X GET "http://127.0.0.1:8000/security/dashboard"

# Get threat indicators and anomaly detection (NEW!)
curl -X GET "http://127.0.0.1:8000/security/threat_indicators"

# Get detailed node security status (NEW!)
curl -X GET "http://127.0.0.1:8000/security/node_status/{node_id}"

# Manually quarantine a suspicious node (NEW!)
curl -X POST "http://127.0.0.1:8000/security/quarantine_node/{node_id}" \
  -d "reason=Suspected compromise"

# Attempt to recover a quarantined node (NEW!)
curl -X POST "http://127.0.0.1:8000/security/recover_node/{node_id}"

# Verify audit results for a completed request (NEW!)
curl -X POST "http://127.0.0.1:8000/security/verify_audit/{request_id}"
```

## System Capabilities

### Achieved Milestones (All 5 Core Features Complete + Advanced Optimizations)
1. ✅ **Selective Inference**: Load only required experts for specific queries
2. ✅ **Partial Mining**: Contributors can improve individual experts independently  
3. ✅ **Expert Evolution**: Independent expert improvement and specialization via DAG
4. ✅ **Distributed Computing**: Experts run on different nodes with P2P coordination and heartbeat monitoring
5. ✅ **Quality-Based Rewards**: Dynamic rewards based on expert usage and performance with persistent tracking
6. ✅ **Upload Stability**: Fixed cross-chain dependency issues and DAG validation performance
7. ✅ **P2P Infrastructure**: Complete node registry with registration, discovery, and load balancing
8. ✅ **Expert Group Optimization**: Intelligent grouping of co-used experts for minimal network overhead
9. ✅ **Hot Expert Caching**: Automatic replication of frequently used expert combinations
10. ✅ **Smart Routing**: Context-aware node selection based on expert group availability
11. ✅ **Real-time Integrity Verification**: Multi-layered security with activation beacons and weight proofs
12. ✅ **Tamper Detection**: Immediate detection of expert swapping, output manipulation, and routing attacks
13. ✅ **Cryptographic Audit Trails**: Complete verification chain with rolling commitments and merkle proofs
14. ✅ **Automatic Failover**: Seamless fallback to secure nodes when integrity verification fails
15. ✅ **Node Quarantine System**: Automatic isolation and recovery of compromised or suspicious nodes
16. ✅ **Adaptive Security Policies**: Dynamic beacon randomization and threshold management

### AI Life Form Characteristics
- **🔄 Autonomous Evolution**: Expert-level independent performance improvement
- **🤝 Distributed Cooperation**: P2P expert sharing and load balancing with intelligent caching
- **📈 Continuous Learning**: Real-time performance monitoring and adaptive routing
- **🧬 Organic Growth**: DAG structure enables parallel expert development
- **💰 Economic Incentives**: Usage-based automatic reward distribution
- **🧠 Collective Intelligence**: Expert groups self-organize based on usage patterns
- **⚡ Adaptive Optimization**: System automatically optimizes network topology for performance
- **🔄 Self-Healing Networks**: Automatic replication and load balancing of critical expert combinations
- **🛡️ Immune System**: Real-time threat detection and tamper resistance with cryptographic verification
- **🔍 Transparency**: Complete audit trails and verifiable computation integrity
- **⚔️ Attack Resilience**: Multi-layered defense against model substitution, output manipulation, and routing attacks
- **🚨 Self-Defense**: Automatic node quarantine and recovery with adaptive threat response
- **🔄 Fault Tolerance**: Seamless failover ensures uninterrupted service during security incidents
- **📊 Security Intelligence**: Comprehensive monitoring and alerting with production-grade dashboards