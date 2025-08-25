# Release Notes
# Blyan Network Version History

## Version 2.0.0 (January 2025) - Current
**Major Architecture Overhaul: Dense Model with Pipeline Parallelism**

### 🔄 Breaking Changes
- **Model Migration**: Switched from MoE (GPT OSS 20B) to dense model (Qwen3-8B)
- **API Changes**: Removed `/upload_moe_experts`, use `/upload_parameters` instead
- **Config Changes**: Updated model configuration format in genesis block

### ✨ New Features
- **Pipeline Parallelism**: Layer-wise model distribution across GPUs
- **Delta Compression**: INT8+Sparse+LoRA for efficient updates
- **Speculative Decoding**: 4x faster inference with draft models
- **Thinking Mode**: Chain-of-thought reasoning support
- **Hybrid Scheduler**: Intelligent workload distribution

### 🚀 Improvements
- **Memory Optimization**: FP8 quantization reduces memory by 50%
- **Inference Speed**: 2x faster with pipeline parallelism
- **Network Efficiency**: 70% reduction in inter-node communication
- **Training Efficiency**: LoRA/QLoRA reduces training time by 80%

### 🐛 Bug Fixes
- Fixed indentation errors in `backend/model/manager.py`
- Resolved cross-chain dependency issues in DAG validation
- Fixed memory leaks in long-running inference sessions
- Corrected rate limiting bypass in health endpoints

### 📝 Migration Guide
```python
# Old (MoE)
response = client.upload_moe_experts(
    model_file="model.bin",
    num_experts=16
)

# New (Dense)
response = client.upload_parameters(
    model_file="model.bin",
    layer_id=0
)
```

---

## Version 1.5.0 (December 2024)
**Production Hardening & Security Enhancements**

### ✨ New Features
- **Two-Chain Architecture**: Separated transaction and PoL chains
- **SIWE Authentication**: Ethereum wallet-based authentication
- **Hardware Binding**: GPU UUID-based node verification
- **Content Safety**: Automated PII/toxicity detection
- **Enterprise Key Management**: AWS KMS/Vault integration

### 🚀 Improvements
- **Consensus**: Tendermint BFT with <2s finality
- **Storage**: PostgreSQL backend with replication
- **Monitoring**: Prometheus/Grafana dashboards
- **Security**: TLS 1.3 for all communications

### 🐛 Bug Fixes
- Resolved consensus fork under high load
- Fixed reward distribution race conditions
- Corrected CORS issues in frontend
- Patched SQL injection vulnerabilities

---

## Version 1.0.0 (October 2024)
**Initial Public Release**

### ✨ Core Features
- **Blockchain Infrastructure**: DAG-based blockchain with cycle detection
- **MoE Architecture**: 384 expert blocks with top-2 routing
- **P2P Network**: Distributed inference coordination
- **Token Economy**: BLY token with dynamic rewards
- **Web Interface**: Chat interface with transparency features

### 📊 Initial Specifications
- Model: GPT OSS 20B (MoE)
- Consensus: Proof-of-Learning
- Network: 100+ GPU nodes
- Performance: <1s inference latency

---

## Version 0.9.0-beta (August 2024)
**Beta Release**

### ✨ Features
- Basic blockchain implementation
- Simple inference routing
- Test token distribution
- Developer API

### ⚠️ Known Issues
- Limited to 10 concurrent nodes
- No persistence between restarts
- Mock inference responses
- No production security

---

## Upcoming Releases

### Version 2.1.0 (Q1 2025 - Planned)
**Scalability & Performance**

#### Planned Features
- [ ] NAT traversal for residential nodes
- [ ] Persistent storage backend (RocksDB)
- [ ] Multi-region deployment
- [ ] Mobile SDK (iOS/Android)
- [ ] Cross-chain bridges to Ethereum

#### Performance Targets
- 10,000+ concurrent nodes
- 1,000+ TPS throughput
- <500ms inference latency
- 99.99% uptime SLA

### Version 2.2.0 (Q2 2025 - Planned)
**Autonomous Evolution**

#### Planned Features
- [ ] Automated model improvement
- [ ] Self-organizing node clusters
- [ ] Multi-modal support (vision/audio)
- [ ] DAO governance implementation
- [ ] Zero-knowledge proofs

### Version 3.0.0 (Q3 2025 - Vision)
**Full Decentralization**

#### Goals
- Complete removal of central dependencies
- Autonomous network operation
- Community-driven development
- Cross-chain interoperability
- 100B+ parameter model support

---

## Versioning Policy

### Semantic Versioning
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes requiring migration
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes and minor improvements

### Support Policy
- **Current Version**: Full support
- **Previous Major**: Security updates only (6 months)
- **Older Versions**: No support

### Deprecation Policy
1. Features marked deprecated in MINOR release
2. Warning period of 3 months
3. Removal in next MAJOR release
4. Migration guide provided

---

## Changelog Format

### Version Template
```markdown
## Version X.Y.Z (Month Year)
**Release Title**

### 🔄 Breaking Changes
- List breaking changes requiring action

### ✨ New Features
- List new capabilities added

### 🚀 Improvements
- List performance/UX improvements

### 🐛 Bug Fixes
- List resolved issues

### 📝 Documentation
- List documentation updates

### 🔒 Security
- List security fixes (with CVE if applicable)

### 📦 Dependencies
- List major dependency updates

### 💔 Deprecated
- List deprecated features

### 🗑️ Removed
- List removed features

### Migration Guide
- Step-by-step migration instructions
```

---

## Release Channels

### Stable
- Production-ready releases
- Extensive testing completed
- Full documentation available
- `blyan.com/api`

### Beta
- Feature-complete releases
- Testing in progress
- May contain bugs
- `beta.blyan.com/api`

### Nightly
- Latest development builds
- Highly experimental
- Frequent breaking changes
- `nightly.blyan.com/api`

---

## Installation & Upgrade

### Docker (Recommended)
```bash
# Pull latest stable
docker pull ghcr.io/blyan-network/blyan:latest

# Pull specific version
docker pull ghcr.io/blyan-network/blyan:2.0.0

# Upgrade
docker-compose pull
docker-compose up -d
```

### Python Package
```bash
# Install latest
pip install blyan-network

# Install specific version
pip install blyan-network==2.0.0

# Upgrade
pip install --upgrade blyan-network
```

### From Source
```bash
# Clone repository
git clone https://github.com/blyan-network/blyan.git
cd blyan

# Checkout version
git checkout v2.0.0

# Install
pip install -r requirements.txt
python setup.py install
```

---

## Security Advisories

### Reporting Vulnerabilities
- Email: security@blyan.com
- PGP Key: [Download](https://blyan.com/pgp.asc)
- Bug Bounty: https://blyan.com/security/bounty

### Recent Advisories
- **CVE-2024-xxxxx**: Rate limiting bypass (Fixed in 1.5.0)
- **CVE-2024-xxxxx**: XSS in chat interface (Fixed in 1.5.0)

---

## Community

### Get Involved
- GitHub: https://github.com/blyan-network
- Discord: https://discord.gg/blyan
- Twitter: @BlyanNetwork
- Blog: https://blog.blyan.com

### Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---
*Last Updated: January 2025*
*Current Version: 2.0.0*