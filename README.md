# Blyan Network

**Transparent, community-owned AI on the blockchain.**

We're building AI that can't lie to you, can't be secretly modified, and belongs to everyone—not just tech giants.

## 🚀 Try It Now

**Web**: [blyan.com](https://blyan.com) - Start chatting immediately, no signup required

**API**: 
```bash
curl -X POST https://blyan.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, Blyan!", "max_tokens": 50}'
```

## 💰 Run a GPU Node (Earn BLY Tokens)

```bash
# Quick start with Docker
docker run --gpus all -d --name blyan-node \
  -p 8001:8001 \
  -e MAIN_NODE_URL=https://blyan.com/api \
  ghcr.io/blyan-network/expert-node:latest
```

Requirements: 16GB+ VRAM GPU (RTX 3090/4090), Linux, Public IP

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [User Guide](USER_GUIDE.md) | Getting started, setup, earning rewards |
| [API Documentation](API_DOCS.md) | REST API reference, SDKs, examples |
| [Technical Specification](TECHNICAL_SPEC.md) | Architecture, protocols, data structures |
| [Architecture](ARCHITECTURE.md) | System design, components, data flow |
| [Release Notes](RELEASE_NOTES.md) | Version history, migration guides |

### For Developers
- [Test Plan](TEST_PLAN.md) - Testing strategy and commands
- [PRD](PRD.md) - Product requirements and vision
- [CLAUDE.md](CLAUDE.md) - AI assistant guidelines

## 🏗️ Quick Development Setup

```bash
# Clone repository
git clone https://github.com/blyan-network/blyan.git
cd blyan

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

## 🌟 Why Blyan?

- **🔍 Transparent**: Every model weight on-chain, fully verifiable
- **🌐 Decentralized**: No single entity controls the AI
- **💎 Community-Owned**: Earn tokens by contributing compute
- **⚡ Efficient**: Pipeline parallelism for fast inference
- **🔒 Secure**: Cryptographic proofs for all operations

## 🤝 Community

- **Discord**: [discord.gg/blyan](https://discord.gg/blyan)
- **Twitter**: [@BlyanNetwork](https://twitter.com/BlyanNetwork)
- **GitHub**: [github.com/blyan-network](https://github.com/blyan-network)

## 📄 License

MIT License - See [LICENSE](LICENSE) file

---

**Together, we're building AI for everyone.** 🚀