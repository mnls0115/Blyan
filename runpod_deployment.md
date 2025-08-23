# RunPod GPU Node Deployment Guide

## Quick Start (One-Command Setup)

```bash
# SSH into your RunPod instance and run:
curl -sSL https://raw.githubusercontent.com/mnls0115/blyan/main/setup_runpod.sh | bash
```

## Manual Setup Instructions

### 1. Create RunPod Instance

1. Go to [RunPod.io](https://runpod.io)
2. Choose a GPU instance:
   - **REQUIRED**: A100 80GB or H100 80GB (Qwen3-30B needs 80GB+ VRAM)
   - **Multi-GPU**: 2x A100 40GB as alternative
3. Select Ubuntu 22.04 template
4. Configure networking:
   - Enable public IP
   - Open ports: 8002 (or your chosen port)

### 2. SSH into RunPod Instance

```bash
ssh root@your-runpod-ip
```

### 3. Clone and Setup Repository

```bash
# Update system
apt update && apt upgrade -y

# Install required system packages
apt install -y python3.10 python3.10-venv python3-pip git wget curl htop nvtop

# Clone repository
cd /workspace
git clone https://github.com/mnls0115/blyan.git
cd blyan

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install UV for fast package installation (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Install dependencies with UV (10x faster)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements-gpu.txt

# OR standard pip install (slower)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install -r requirements-gpu.txt
```

### 4. Configure Environment Variables

```bash
# Create environment file
cat > .env << 'EOF'
# Node Configuration
NODE_PORT=8002
MAIN_NODE_URL=https://blyan.com/api

# Model Configuration (auto-downloads on first run)
MODEL_NAME=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
AUTO_UPLOAD=true
SKIP_POL=true

# Data Directory
BLYAN_DATA_DIR=/workspace/blyan/data

# Enable Block Runtime (new inference layer)
BLOCK_RUNTIME_ENABLED=true
BLOCK_RUNTIME_MEMORY_CACHE_MB=8192
BLOCK_RUNTIME_DISK_CACHE_MB=20480

# HuggingFace Cache
HF_HOME=/workspace/blyan/data/.hf
HF_HUB_ENABLE_HF_TRANSFER=1

# CUDA Settings
CUDA_VISIBLE_DEVICES=0,1  # Use all available GPUs
EOF

source .env
```

### 5. Test GPU Detection

```bash
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); print(f'GPU 0: {torch.cuda.get_device_name(0)}')"
```

### 6. Run GPU Node

```bash
# Activate environment
source .venv/bin/activate
source .env

# Run the GPU node
python run_gpu_node.py
```

### 7. Create SystemD Service (for Persistent Running)

```bash
cat > /etc/systemd/system/blyan-gpu.service << 'EOF'
[Unit]
Description=BLYAN GPU Node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace/blyan
Environment="PATH=/workspace/blyan/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONPATH=/workspace/blyan"
EnvironmentFile=/workspace/blyan/.env
ExecStart=/workspace/blyan/.venv/bin/python run_gpu_node.py
Restart=always
RestartSec=10
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable blyan-gpu
systemctl start blyan-gpu

# Check status
systemctl status blyan-gpu

# View logs
journalctl -u blyan-gpu -f
```

## Model Auto-Download and Upload

The GPU node will automatically:
1. **Download** the model from HuggingFace on first run
2. **Extract** experts from the MoE model
3. **Upload** to blockchain as individual expert blocks
4. **Cache** for future inference

This process happens once and takes 60-90 minutes depending on:
- Model size (Qwen3-30B: ~60GB)
- Network speed
- GPU memory (80GB required)

## Model Configuration

```bash
# ONLY USE THIS MODEL - Qwen3 30B
MODEL_NAME=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8  # 30B params, requires 80GB+ GPU
```

## Monitoring

### Check Node Status
```bash
# Node health
curl http://localhost:8002/health

# Blockchain status
curl http://localhost:8002/chain/B | jq '.total_blocks'

# Test inference
curl -X POST http://localhost:8002/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_new_tokens": 50}'
```

### GPU Monitoring
```bash
# Real-time GPU usage
nvtop

# Or nvidia-smi
watch -n 1 nvidia-smi
```

### Logs
```bash
# Service logs
journalctl -u blyan-gpu -f

# Or direct output if running manually
python run_gpu_node.py 2>&1 | tee gpu_node.log
```

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size or use quantization
export LOAD_IN_8BIT=true  # For 8-bit quantization
export LOAD_IN_4BIT=true  # For 4-bit quantization (requires bitsandbytes)
```

### Connection Issues
```bash
# Check if main node is reachable
curl https://blyan.com/api/health

# Check firewall
ufw status
ufw allow 8002/tcp  # If needed
```

### Model Download Issues
```bash
# Clear HF cache and retry
rm -rf /workspace/blyan/data/.hf
python run_gpu_node.py
```

### Disk Space Issues
```bash
# Check disk usage
df -h

# Clean up old blocks if needed
rm -rf /workspace/blyan/data/chain_B/*.block
```

## Performance Optimization

### Multi-GPU Setup
The node automatically detects and uses all available GPUs:
```python
# Automatic device_map="auto" distributes model across GPUs
```

### Memory Optimization
- **FP16**: Default for most models
- **FP8**: Automatic for supported models (Qwen3-30B)
- **Quantization**: Available via LOAD_IN_8BIT/4BIT flags

### Caching
With block runtime enabled:
- Memory cache: 8GB default
- Disk cache: 20GB default
- Expert prefetching for faster inference

## Cost Optimization

### RunPod Spot Instances
- Use spot instances for 50-80% cost savings
- Enable auto-restart in case of preemption
- Use persistent storage for blockchain data

### Instance Requirements for Qwen3-30B
- **Minimum**: A100 80GB (~$2.20/hr)
- **Recommended**: H100 80GB (~$3.50/hr)
- **Alternative**: 2x A100 40GB (~$2.20/hr total)

## Security

### API Key (Optional)
```bash
# Generate API key for this node
export BLYAN_API_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
echo "BLYAN_API_KEY=$BLYAN_API_KEY" >> .env
```

### Firewall
```bash
# Only allow specific IPs (optional)
ufw allow from 165.227.221.225 to any port 8002
ufw enable
```

## Advanced Configuration

### Custom Model Upload
```bash
# Upload specific model manually
python miner/upload_moe_parameters.py \
  --address gpu_node \
  --model-file /path/to/model \
  --meta-hash $(curl -s http://localhost:8002/chain/A | jq -r '.latest_hash')
```

### Distributed Training (PoL)
```bash
# Enable Proof-of-Learning
export SKIP_POL=false
export ENABLE_LEARNING=true
```

### Block Runtime Features
```bash
# Enable advanced features
export BLOCK_RUNTIME_PREFETCH=true
export BLOCK_RUNTIME_HEDGED_FETCH=true
export BLOCK_RUNTIME_ENABLE_VERIFICATION=true
```

## Support

- **Logs**: Check `/workspace/blyan/gpu_node.log`
- **GitHub Issues**: https://github.com/mnls0115/blyan/issues
- **Main Node Status**: https://blyan.com/api/health