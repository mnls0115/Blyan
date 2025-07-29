# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-Block is an experimental blockchain that runs AI models, combining proof-of-work consensus with proof-of-learning for parameter updates. The system consists of two blockchains: a meta-chain (chain A) for model specifications and a parameter-chain (chain B) for model weights.

## Development Setup

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Initialize Genesis Block
Before first run, create the meta-chain genesis block:
```bash
python - <<'PY'
import json
from pathlib import Path
from backend.core.chain import Chain

root_dir = Path("./data")
meta_chain = Chain(root_dir, "A")
spec = {"model_name": "distilbert-base-uncased"}
meta_chain.add_block(json.dumps(spec).encode())
print("✅ Meta chain initialized.")
PY
```

### Running the Application
- **API Server**: `uvicorn api.server:app --reload` (runs on http://127.0.0.1:8000)
- **Frontend**: Open `frontend/index.html` directly in browser (no build step required)

## Architecture

### Core Components
- **Chain (`backend/core/chain.py`)**: Blockchain implementation with proof-of-work mining
- **Block (`backend/core/block.py`)**: Block structure with headers and payloads
- **ModelManager (`backend/model/infer.py`)**: Manages AI model loading from blockchain state
- **Ledger (`backend/core/ledger.py`)**: Simple token balance tracking
- **API Server (`api/server.py`)**: FastAPI endpoints for chat, mining, and chain operations

### Blockchain Structure
- **Meta-chain (A)**: Stores model specifications (model name, architecture configs)
- **Parameter-chain (B)**: Stores model weights with proof-of-learning validation
- **Proof-of-Learning**: Requires submitted parameters to improve model loss over previous version

### Key Data Flow
1. Meta-chain defines which AI model to use
2. Parameter-chain stores incremental model weight updates
3. ModelManager reconstructs complete model from latest meta + parameter blocks
4. Chat endpoint uses reconstructed model for inference
5. Mining endpoints allow submitting new parameters with PoL validation

### File Structure
- `backend/core/`: Blockchain primitives (chain, block, storage, consensus)
- `backend/model/`: AI model architecture and inference management
- `api/`: FastAPI server with REST endpoints
- `frontend/`: Simple HTML/JS web interface
- `miner/`: Scripts for parameter submission and mining

### Mining Process
- Single parameters: `/mine` endpoint with base64-encoded weights
- Bulk parameters: `/upload_parameters` endpoint chunks large state_dict files
- All submissions require ECDSA signature verification and proof-of-learning validation