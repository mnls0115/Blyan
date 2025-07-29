# AI-Block: Blockchain-Native AI Model Platform

---

## 1. Concept & Motivation

* **Trustworthy AI** – Embeds model behaviour rules, code, and weights immutably on a blockchain so anyone can audit what the model will do.
* **Dual-Chain DNA** – Treats the combined Meta-chain (rules/code) and Parameter-chain (weights) as the AI’s *DNA*: reproducible, verifiable, and upgradable through consensus.
* **Proof-of-Learning Mining** – New parameter blocks are accepted only if they demonstrably improve model quality on a public validation set, blending *quality-gated PoL* with a light PoW for spam resistance.
* **Economic Incentives** – A token ledger rewards miners that contribute compute or data, and users pay small fees to query the AI, creating a closed economy.

---

## 2. Architecture Overview

### 2.1 Block Structure (common header)
| Field | Purpose |
|-------|---------|
| `index` | Sequential height in its chain |
| `timestamp` | Unix epoch (float) |
| `prev_hash` | Hash of previous block in same chain |
| `chain_id` | "A" = Meta, "B" = Parameters |
| `points_to` | Hash in sister chain this block depends on |
| `payload_hash` | SHA-256 of payload bytes |
| `payload_size` | Bytes |
| `nonce` | PoW nonce |
| `miner_pub` | Compressed ECDSA public key |
| `payload_sig` | ECDSA signature of payload |

### 2.2 Dual-Chain Model
* **Meta-Chain (A)** – Stores immutable behaviour guidelines and occasionally updated execution code or architecture specs.
* **Parameter-Chain (B)** – Stores model weights. *One parameter tensor → one block* for fast selective loading. Each block links to the Meta block hash it complies with.

### 2.3 Inference Flow
1. Web/UI sends a prompt to `/chat`.
2. `ModelManager` reads latest Meta block to know which HF model arch to instate.
3. Parameter Index maps tensor names to block indices → only necessary blocks are read & merged into `state_dict`.
4. Model weights loaded into memory → HuggingFace generates response → result returned to user.

### 2.4 Mining Flow
1. Miner trains or fine-tunes, producing new weights file (`state_dict`).
2. Miner signs raw bytes with ECDSA key, submits via `/mine` (single chunk) or `/upload_parameters` (automatic tensor-splitting).
3. Server verifies:
   * PoL – candidate loss improves baseline loss ≥ δ.
   * Signature – secp256k1 + SHA-256.
   * PoW – finds nonce to satisfy difficulty.
4. For every accepted block: token reward → ledger credit, index updated.

---

## 3. Technology Stack
* **Language / Runtime** – Python 3.9+
* **AI / ML** – PyTorch, HuggingFace Transformers
* **Web API** – FastAPI, Uvicorn
* **Blockchain Logic** – Custom Python modules (`backend/core/*`)
* **Crypto** – `hashlib` & `ecdsa` (secp256k1)
* **Frontend** – Vanilla HTML + CSS + JavaScript (no build step)
* **Data Storage** – JSON files (blocks, ledgers, param index) → *planned LevelDB upgrade*

---

## 4. Current Usage

### 4.1 Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # includes fastapi, uvicorn, transformers, torch, ecdsa
```

### 4.2 Initialize Genesis Meta Block
```bash
python scripts/init_meta.py   # see README for snippet
```

### 4.3 Run Server
```bash
uvicorn api.server:app --reload
```

### 4.4 Interact
* **Chat** – POST `/chat` `{ "prompt": "Hello" }`
* **Mine (single chunk)** – `python miner/submit_payload.py ...`
* **Mine (full weights)** – `python miner/upload_parameters.py ...`
* **Balance** – GET `/balance/{address}`
* **Transfer** – POST `/transfer`
* **Chain Browse** – GET `/chain/B/blocks`, `/chain/B/block/{index}`

---

## 5. Milestones Completed
- ✅ Core blockchain (Block, Chain, Storage) with PoW
- ✅ Dual-chain (Meta / Parameter)
- ✅ Proof-of-Learning stub integration
- ✅ Parameter splitting & **index-based** fast loading
- ✅ Token ledger (credit / transfer)
- ✅ ECDSA signing & verification for every block
- ✅ Miner CLI scripts (auto-key generation)
- ✅ REST API: chat, mine, upload, balance, transfer, chain browse
- ✅ Web frontend: chat UI + wallet balance + recent blocks list

---

## 6. Roadmap / Next Steps
- [ ] Replace JSON `ParameterIndex` with **LevelDB** or RocksDB for O(1) random reads & concurrent writes
- [ ] Streamed / chunked inference to reduce peak GPU memory
- [ ] Full Proof-of-Learning pipeline with public validation dataset & multi-node consensus
- [ ] P2P network & block gossip (libp2p, websockets)
- [ ] Advanced frontend (React or Svelte) with mining dashboard & block explorer
- [ ] Hardware wallet or MetaMask-like signing for users
- [ ] Docker / k8s deployment scripts

---

## 7. Developer On-Boarding

### 7.1 Repository Layout
```
backend/
  core/          # blockchain, ledger, PoW, PoL, index
  model/         # HF wrapper, ModelManager
api/             # FastAPI server & endpoints
miner/           # CLI mining utilities
frontend/        # index.html, style.css, main.js
scripts/         # one-off helpers (e.g., init_meta.py)
```

### 7.2 Conventions & Tips
* **Typing** – Python ≥3.9, use type hints; `# type: ignore` only for optional deps.
* **Commit Style** – Conventional Commits (`feat:`, `fix:`, `docs:` …).
* **Testing** – `pytest` (future work) for unit tests of chain validation.
* **Security** – Never commit private keys; miner scripts will generate keys when omitted.
* **Performance** – Keep block payloads small (tensor-level) to avoid large read overhead.

### 7.3 Getting Help
* Run `uvicorn` with `--reload` during dev for hot-reloading.
* Search codebase via `ripgrep` (`rg`), symbols are organised per folder above.

---

Happy hacking! Contributions, ideas, and PRs are welcome to push this vision of transparent, decentralised AI forward. 