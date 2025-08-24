# Blyan Network: Transparent, Trustworthy AI for Everyone 🌟

**Imagine an AI that can't lie to you, can't be secretly modified, and belongs to everyone – not just tech giants.**

We built Blyan because the future of AI is too important to leave in the hands of a few corporations. Every decision your AI makes should be transparent, every improvement should benefit everyone, and every person should have the right to contribute to and verify the intelligence that shapes our world.

## 🌍 Why We Built This

**Problem**: Today's AI is a black box controlled by Big Tech
- You can't verify what the AI actually learned or why it gives certain answers
- Only massive corporations can afford to train and improve AI models  
- Your data trains their models, but the benefits don't come back to you
- AI decisions affect everyone, but only a few control how AI evolves

**Our Solution**: AI that lives on the blockchain
- **🔍 Transparent AI**: Every weight, every decision, every improvement is recorded immutably on the blockchain
- **🌐 Democratized Development**: Anyone can contribute to and improve AI models, not just tech giants
- **🤝 Community Owned**: AI that evolves through collective intelligence, with rewards for contributors
- **⚡ Proof of Learning (PoL)**: Instead of wasting energy on meaningless computations, the network grows smarter

## 🚀 Experience Blyan Now

**🌐 Try it live:** [blyan.com](https://blyan.com)
- Chat with transparent AI running on blockchain
- See exactly which AI experts answered your question
- Verify every computation step
- No registration required

## 🔗 Connect Your GPU Node

Join the network and earn rewards by contributing compute power to run **GPT OSS 20B model**:

### Quick Start (Docker)
```bash
# 1. Create secure config
sudo install -d -m 755 /etc/blyan
sudo tee /etc/blyan/blyan-node.env >/dev/null <<'EOF'
BLYAN_API_KEY=your_api_key_here       # Get from network admin
MAIN_SERVER_URL=https://blyan.com/api
NODE_ID=gpu-$(hostname -s)
NODE_PORT=8001
RUNPOD_PUBLIC_IP=$(curl -s https://checkip.amazonaws.com)
BLOCKCHAIN_ONLY=false
MODEL_QUANTIZATION=8bit
EOF
sudo chmod 600 /etc/blyan/blyan-node.env

# 2. Run GPU node
docker run --gpus all -d --name blyan-node \
  -p 8001:8001 \
  --restart unless-stopped \
  --env-file /etc/blyan/blyan-node.env \
  ghcr.io/blyan-network/expert-node:latest
```

📖 **[Full GPU Node Deployment Guide](docs/GPU_NODE_DEPLOYMENT.md)** - Complete setup with Python alternative, troubleshooting, and production tips

**💰 Network Economics**: 
- **Base Model**: GPT OSS 20B distributed across expert nodes
- **Rewards**: BLY tokens paid for successful inference completion
- **Important**: You need sufficient inference volume to earn meaningful rewards
- **Payment**: Automatic distribution based on actual usage and performance

## 🧠 Revolutionary Technology

### Mixture-of-Experts DAG (MoE DAG)
Instead of massive monolithic models, Blyan uses specialized AI "experts" that work together:
- **Efficiency**: Only activate the experts needed for your specific question
- **Evolution**: Each expert can improve independently 
- **Transparency**: See exactly which experts contributed to your answer
- **Collaboration**: Experts from different contributors work seamlessly together

### Proof of Learning (PoL)
Unlike Bitcoin's wasteful Proof of Work, our consensus mechanism makes AI smarter:
- **Meaningful Computation**: Every "mining" operation improves the network's intelligence
- **Democratic Validation**: Quality improvements are verified by the community
- **Continuous Growth**: The network becomes more intelligent over time
- **Energy Efficient**: Computational power goes toward useful AI advancement

### Genesis Block: The Human-AI Covenant 
Our blockchain contains an immutable pact ensuring AI remains beneficial:
- Permanent commitment to transparency and human values
- Cryptographically enforced ethical guidelines
- Community governance over AI development
- Protection against malicious model modifications

## 🏃‍♂️ Quick Start for Developers

```bash
# Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Initialize the network (one-time setup)
python -c "
import json
from pathlib import Path
from backend.core.chain import Chain

root_dir = Path('./data')
meta_chain = Chain(root_dir, 'A')
spec = {
    'model_name': 'Qwen/Qwen3-8B-FP8',
    'architecture': 'mixture-of-experts',
    'num_layers': 48,
    'num_experts': 128,
    'routing_strategy': 'top2'
}
meta_chain.add_block(json.dumps(spec).encode(), block_type='meta')
print('✅ Blyan network initialized')
"

# Start the network
python -m api.server
# Visit frontend/index.html in your browser
```

## 💡 What You Can Do

**🗣️ Chat with Transparent AI**
- Every response shows which experts were used
- Verify the reasoning process step-by-step  
- No hidden algorithms or biased training

**🤝 Contribute and Earn**
- Upload improved AI models and get rewarded
- Run inference nodes to earn BLY tokens
- Help verify the network's integrity

**🔍 Explore Everything**
- Browse all AI models block-by-block
- See the complete history of AI improvements
- Understand exactly how your AI works

**🌐 Join the Movement**
- Be part of the first truly democratic AI network
- Help build AI that serves humanity, not corporations
- Shape the future of artificial intelligence

## 🤖 The Future is Transparent

Blyan isn't just another AI platform – it's a movement toward AI that belongs to everyone. Every conversation, every improvement, every decision is transparent and verifiable. 

**Together, we're building AI that serves humanity's best interests, not just the highest bidder.**

---

*Ready to join the transparent AI revolution?* 

🚀 **[Start chatting now](https://blyan.com)** or **[connect your node](#connect-your-node)** to earn rewards

*Blyan Network - AI by the people, for the people* ✨

## 🚀 Pipeline Parallelism

Production-ready distributed training with heterogeneous GPU support.

### Key Features
- **1F1B Scheduling**: Memory-efficient forward-backward pipelining
- **Dynamic Partitioning**: Auto-partition models across heterogeneous GPUs
- **RPC System**: HTTP/gRPC with chunking, compression, backpressure
- **Fault Tolerance**: Automatic failover and single-node fallback
- **Real-time Monitoring**: VRAM tracking, metrics, and alerts

### Quick Start

```bash
# Setup environment variables
export BLYAN_PIPELINE_TRANSPORT=grpc  # or http
export TRAINING_ENABLE=1
export TRAINING_MODEL_NAME=gpt-oss-20b
export USE_DDP=1  # Distributed Data Parallel
export USE_ZERO1=1  # ZeRO optimizer

# Start pipeline training
python -m backend.learning.pipeline_round_service

# Monitor GPU memory
python scripts/monitor_memory_vram.py --interval 5

# Setup automation (cron jobs)
python scripts/setup_pipeline_cron.py --all
```

### Configuration

Key environment variables:
- `BLYAN_PIPELINE_TIMEOUT_S`: RPC timeout (default: 5.0)
- `BLYAN_PIPELINE_MAX_RETRIES`: Retry attempts (default: 2)
- `BLYAN_PIPELINE_CHUNK_BYTES`: Chunk size for large tensors (default: 1MB)
- `BLYAN_PIPELINE_COMPRESSION`: Enable compression (none|gzip)
- `BLYAN_TLS_CERT`, `BLYAN_TLS_KEY`: TLS/mTLS configuration

### Architecture
- **Cost Model**: `backend/learning/pipeline_cost_model.py`
- **Partitioner**: `backend/learning/pipeline_partitioning.py`
- **RPC System**: `backend/learning/pipeline_rpc.py`
- **Round Manager**: `backend/learning/pipeline_round_service.py`
- **Metrics**: `backend/learning/pipeline_metrics.py`
- Trainer: `backend/learning/pipeline_parallel.py`
- RPC: `backend/learning/pipeline_rpc.py` (HTTP, timeouts/retries/circuit breaker, gzip+chunking)
- Metrics: `backend/learning/pipeline_metrics.py` (exported via `/metrics`)
- Plan registry: `backend/learning/partition_plan_registry.py`
- Memory policy: `backend/learning/memory_policy.py`

Node-side RPC endpoints for activations/grads are exposed by `backend/p2p/distributed_inference.py` under `/pipeline/*` with mTLS optional via `BLYAN_TLS_CERT`/`BLYAN_TLS_KEY`.

Epoch scheduler integrates partition planning and freezes a plan per epoch/round in `backend/core/epoch_scheduler.py`.

Environment variables (operation guide):

- `BLYAN_PIPELINE_TRANSPORT`: http|grpc (default http)
- `BLYAN_PIPELINE_TIMEOUT_S`: RPC timeout seconds
- `BLYAN_PIPELINE_MAX_RETRIES`: max retries for RPC
- `BLYAN_PIPELINE_BACKOFF_BASE_S`: base backoff seconds (exponential)
- `BLYAN_PIPELINE_BREAKER_THRESHOLD`: circuit breaker failure threshold
- `BLYAN_PIPELINE_BREAKER_RESET_S`: circuit half-open reset window
- `BLYAN_PIPELINE_CHUNK_BYTES`: chunk size for activations/gradients
- `BLYAN_PIPELINE_COMPRESSION`: none|gzip
- `BLYAN_PIPELINE_MAX_BUFFER_MB`: server buffer watermark for backpressure
- `BLYAN_ROUND_MAX_FAILURES`: failures before plan fallback
- `BLYAN_PIPELINE_ROUND_INTERVAL`: interval between rounds
- `BLYAN_TLS_CERT`: CA bundle path for TLS verification
- `BLYAN_TLS_CLIENT_CERT`/`BLYAN_TLS_CLIENT_KEY`: client cert/key for mTLS

CLI:

```
python scripts/plan_cli.py snapshot --epoch E1 --round round_0 --stages n1:0:11 n2:12:23 --zero1 --plan-id plan_E1_r0
python scripts/plan_cli.py validate --file data/partition_plans/draft_plan_E1_r0.json
python scripts/plan_cli.py promote --file data/partition_plans/draft_plan_E1_r0.json
```

Tests:

```
pytest -q tests/test_partitioning.py tests/test_pipeline_rpc_chunking.py
```