# CODEMAP.md - Blyan Network Codebase Map

## Directory Overview

| Directory | Purpose | Key Entry Files |
|-----------|---------|-----------------|
| `api/` | Top-level REST API server | `server.py` (main), `chat_atomic.py`, `streaming.py` |
| `backend/` | Core subsystems | See subsystem index below |
| `frontend/` | Web UI (vanilla JS) | `index.html`, `chat.html`, `explorer.html` |
| `miner/` | Model upload tools | `upload_moe_parameters.py`, `submit_payload.py` |
| `scripts/` | Utilities & demos | `demo_*.py`, setup scripts, monitoring |
| `migrations/` | Database schemas | `001_ledger_schema.sql` |
| `proto/` | gRPC definitions | `pipeline_rpc.proto`, `edge_aggregator.proto` |
| `config/` | System configuration | `tokenomics.yaml`, `economics.yaml` |
| `client/` | Node client SDK | `blyan_client.py` |
| `data/` | Runtime storage | Chain data (A/B/D), checkpoints, keys |

## Subsystem Index

| Module | Path | Responsibilities |
|--------|------|------------------|
| **Core Blockchain** | `backend/core/` | DAG chain, blocks, tensorblocks, storage |
| **Consensus** | `backend/consensus/` | State sync, validator rewards |
| **P2P Network** | `backend/p2p/` | Distributed inference, node registry, expert groups |
| **PoL System** | `backend/pol/` | Proof-of-Learning validation, anti-gaming |
| **Rewards** | `backend/rewards/` | Automatic BLY token distribution |
| **Inference** | `backend/inference/` | Batch management, caching |
| **Learning** | `backend/learning/` | Consensus learning, micro-step training, pipeline parallel |
| **Model** | `backend/model/` | MoE inference, expert cache, architecture |
| **Security** | `backend/security/` | Integrity verification, key management, SBOM |
| **Economics** | `backend/economics/` | Cost calculator, billing gateway |
| **Data Pipeline** | `backend/data/` | Quality validation, PoL dataset scoring |
| **Optimization** | `backend/optimization/` | KV cache, multi-GPU, quantization |
| **Monitoring** | `backend/monitoring/` | Metrics export, dashboards |
| **API Routes** | `backend/api/` | Auth, voting, wallet integration |

## Main Entry Points

| Component | Command | Description |
|-----------|---------|-------------|
| **Full Stack** | `./server.sh start` | API server + P2P nodes |
| **API Only** | `./server.sh start api` | REST API on :8000 |
| **API Direct** | `python -m api.server` | API server (systemd) |
| **P2P Node** | `python -m backend.p2p.distributed_inference server <id> <port>` | Expert node |
| **Frontend** | Open `frontend/index.html` | No build required |
| **Status** | `./server.sh status` | Check all services |

## Data & Schema Locations

| Type | Location | Used By |
|------|----------|---------|
| **Ledger Schema** | `migrations/001_ledger_schema.sql` | `backend/accounting/ledger_postgres.py` |
| **Chain Data** | `data/A/` (meta), `data/B/` (params), `data/D/` (datasets) | `backend/core/chain.py` |
| **Tokenomics** | `config/tokenomics.yaml` | `backend/core/reward_engine_secure.py` |
| **Node Auth** | `config/node_auth.json` | `backend/api/node_auth.py` |
| **gRPC Protos** | `proto/*.proto` | Pipeline parallel, edge aggregation |
| **Genesis Block** | See `CLAUDE.md` init instructions | Meta-chain setup |

## Key Workflows

### Mining/Upload
1. Extract experts: `scripts/extract_individual_experts.py`
2. Upload MoE: `miner/upload_moe_parameters.py --meta-hash <hash> --candidate-loss 0.8`
3. Submit via: `miner/submit_payload.py`

### Consensus & Validation
1. State sync: `backend/consensus/state_sync.py`
2. PoL validation: `backend/pol/proof_of_learning.py`
3. Validator rewards: `backend/consensus/validator_rewards.py`

### Inference Flow
1. Request → `api/server.py` → `/chat` endpoint
2. Route to experts via `backend/p2p/distributed_inference.py`
3. MoE inference: `backend/model/moe_infer.py`
4. Batch optimization: `backend/inference/batch_manager.py`

### Training/Learning
1. Consensus epochs: `backend/learning/consensus_learning.py`
2. Micro-step training: `backend/learning/micro_step_trainer.py`
3. Pipeline parallel: `backend/learning/pipeline_parallel.py`

### Rewards Distribution
1. Usage tracking in `backend/p2p/distributed_inference.py`
2. Calculation via `backend/core/reward_engine_secure.py`
3. Auto-distribution: `backend/rewards/automatic_distribution.py`

### P2P Operations
1. Register node: POST `/p2p/register`
2. Expert groups: `backend/p2p/expert_group_optimizer.py`
3. Hot cache: `backend/p2p/hot_expert_cache.py`

## Cross-Links to Deeper Docs

- **Project Overview**: `README.md` - User-facing intro
- **Development Guide**: `CLAUDE.md` - Setup & commands
- **Architecture Whitepaper**: `moe_dag_whitepaper.md` - Technical design
- **Roadmaps**: `DEVELOPMENT_ROADMAP.md`, `PRODUCTION_ROADMAP_2025.md`
- **Security Setup**: `docs/SECURITY_SETUP.md`
- **Operations Runbook**: `docs/RUNBOOK.md`

## How to Update This Map

1. **When adding subsystems**: Add to Subsystem Index with path and ≤15 word description
2. **New entry points**: Update Main Entry Points table with run command
3. **Schema changes**: Update Data & Schema Locations
4. **Workflow changes**: Update relevant workflow section
5. **Keep concise**: Tables for overviews, bullets for workflows
6. **Verify paths**: Ensure all paths are relative and valid

Last updated: 2025-01-15