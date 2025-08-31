# Blyan Network

**Transparent, community-owned AI on the blockchain.**

We're building AI that is for everyone, built by everyone, and belongs to everyone—not just tech giants.

## 🚀 Try It Now

**Web**: [blyan.com](https://blyan.com) - Start chatting immediately, no signup required

**API**:
```bash
curl -X POST https://api.blyan.com/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, Blyan!", "max_tokens": 50}'
```

## 💰 Run a GPU Node (Docker)

```bash
# Standardized Docker command (first run requires JOIN_CODE)
docker run -d --name blyan-node \
  --gpus all \
  -p 8001:8001 \
  -v /var/lib/blyan/data:/data \
  -e JOIN_CODE=YOUR_CODE_HERE \
  -e MAIN_NODE_URL=https://api.blyan.com \
  -e PUBLIC_HOST=$(curl -s https://checkip.amazonaws.com) \
  -e PUBLIC_PORT=8001 \
  -e JOB_CAPACITY=1 \
  -e NODE_ENV=production \
  --restart unless-stopped \
  mnls0115/blyan-node:latest
```

- Requirements: NVIDIA GPU (12+ GB VRAM), Linux/WSL2, CUDA 12.x, Docker + NVIDIA Container Toolkit
- Security: Container runs as non-root. Credentials are stored in `/data/credentials.json` with 0600 permissions.

Or use Compose:
```bash
export JOIN_CODE=YOUR_CODE_HERE
docker-compose up -d
```

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

Apache 2.0 License - See [LICENSE](LICENSE) file

---

**Together, we're building AI for everyone.** 🚀
## 🔑 API Keys (JWT)

API keys are required for authenticated access to the main API; not needed for running a node.

Register a key:
```bash
curl -X POST https://api.blyan.com/auth/v2/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-api-key",
    "key_type": "basic",
    "metadata": {"purpose": "testing"}
  }'
```

Validate your key:
```bash
curl -X GET https://api.blyan.com/auth/v2/validate \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Refresh your key:
```bash
curl -X POST https://api.blyan.com/auth/v2/refresh \
  -H "Content-Type: application/json" \
  -d '{"current_key": "YOUR_CURRENT_API_KEY"}'
```

See full details in `frontend/contribute.html#api-key-section`.
