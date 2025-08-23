# GPU Node Deployment Guide

## Quick Start (Docker - Recommended)

### 1. Prerequisites
- Linux server with NVIDIA GPU (≥40GB VRAM for 20B model)
- NVIDIA Driver ≥545
- Docker & NVIDIA Container Toolkit installed
- Public IP address or DNS name
- ≥25-40GB free disk space

### 2. System Check
```bash
# Verify prerequisites
docker --version && nvidia-smi && \
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

### 3. Create Secure Configuration
```bash
# Create config directory
sudo install -d -m 755 /etc/blyan

# Create environment file (safer than -e flags)
sudo tee /etc/blyan/blyan-node.env >/dev/null <<'EOF'
# REQUIRED - Get from main node admin
BLYAN_API_KEY=your_api_key_here           # NEVER commit this
MAIN_SERVER_URL=https://blyan.com/api
NODE_ID=gpu-$(hostname -s)
NODE_PORT=8001

# REQUIRED - Your public IP or DNS
RUNPOD_PUBLIC_IP=your.public.ip.here      # Use: curl -s https://checkip.amazonaws.com

# REQUIRED for actual model serving
BLOCKCHAIN_ONLY=false                     # Must be false to load model
# MODEL_QUANTIZATION=8bit                 # WARNING: Only for inference nodes, breaks learning!

# OPTIONAL
DONOR_MODE=false                          # true = help free-tier, no rewards
# AVAILABLE_EXPERT=gpt-oss-20b           # Specify expert to load
# HF_TOKEN=your_huggingface_token        # If using private models
EOF

# Secure the file
sudo chmod 600 /etc/blyan/blyan-node.env
```

### 4. Configure Networking
```bash
# Open firewall (UFW example)
sudo ufw allow 8001/tcp

# For cloud providers (AWS/GCP/Azure)
# Add security group rule: Inbound TCP 8001 from 0.0.0.0/0 or specific IPs
# Ensure outbound HTTPS (443) to blyan.com is allowed
```

### 5. Run GPU Node

#### Production Mode (Background)
```bash
docker run --gpus all -d --name blyan-node \
  -p 8001:8001 \
  --restart unless-stopped \
  --env-file /etc/blyan/blyan-node.env \
  ghcr.io/blyan-network/expert-node:latest
```

#### Debug Mode (Interactive)
```bash
docker run --gpus all --rm -it \
  -p 8001:8001 \
  --env-file /etc/blyan/blyan-node.env \
  ghcr.io/blyan-network/expert-node:latest
```

### 6. Verify Deployment
```bash
# Check logs
docker logs -f blyan-node | grep -E "✅|📦|ERROR"

# Test local health (endpoint is / not /health)
curl -s http://localhost:8001/ | jq .

# Verify registration with main server
export $(grep -v '^#' /etc/blyan/blyan-node.env | xargs)
curl -s -H "X-API-Key: $BLYAN_API_KEY" "$MAIN_SERVER_URL/p2p/nodes" | \
  jq --arg id "$NODE_ID" '.nodes[] | select(.node_id==$id)'

# Test local inference
curl -s -X POST http://localhost:8001/inference \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello, world!","max_new_tokens":16}' | jq .
```

## Alternative: Python Installation

### 1. Clone Repository
```bash
git clone https://github.com/blyan-network/blyan.git
cd blyan
```

### 2. Setup Python Environment
```bash
# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install Dependencies
```bash
# Install GPU requirements (includes PyTorch placeholder)
pip install -r requirements-gpu.txt

# Install CUDA-enabled PyTorch (choose based on your CUDA version)
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# For CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Configure Environment
```bash
# Copy secure config
cp /etc/blyan/blyan-node.env .env && chmod 600 .env

# Ensure these are set in .env:
# BLOCKCHAIN_ONLY=false
# MODEL_QUANTIZATION=8bit
# RUNPOD_PUBLIC_IP=your.public.ip
```

### 5. Run Node
```bash
export $(grep -v '^#' .env | xargs)
python runpod_node.py
```

## Minimum Required Files (Python Path)
- `runpod_node.py` - Main entrypoint
- `backend/model/arch.py` - Model wrapper
- `backend/model/__init__.py`
- `backend/__init__.py`
- `requirements-gpu.txt` - GPU node dependencies
- `.env` - Configuration (optional, can use environment variables directly)

## Common Issues & Solutions

### Registration Fails (400 Error)
- **Cause**: Invalid or unreachable IP
- **Fix**: Set `RUNPOD_PUBLIC_IP` to actual public IP
- **Get IP**: `curl -s https://checkip.amazonaws.com`

### Authentication Error (401)
- **Cause**: Missing or invalid API key
- **Fix**: Set `BLYAN_API_KEY` in env file
- **Contact**: Main node admin for valid key

### GPU Not Available
- **Cause**: Driver/toolkit issues
- **Fix**: 
  ```bash
  # Install NVIDIA Container Toolkit
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update && sudo apt-get install -y nvidia-docker2
  sudo systemctl restart docker
  ```

### Model Out of Memory
- **Cause**: Insufficient VRAM
- **Fix**: Ensure `MODEL_QUANTIZATION=8bit` is set
- **Alternative**: Use smaller expert or increase GPU memory

### Port Already in Use
- **Cause**: Another service on 8001
- **Fix**: Change `NODE_PORT` in env file and Docker `-p` flag

## Donor Mode Operation
```bash
# Enable donor mode (no rewards, helps free-tier users)
sudo sed -i 's/^DONOR_MODE=.*/DONOR_MODE=true/' /etc/blyan/blyan-node.env
docker restart blyan-node
```

## Security Best Practices

### 1. Secrets Management
- **Never** put API keys in command history
- Use env files with `chmod 600`
- Consider Docker Swarm/K8s secrets for orchestration
- Rotate API keys regularly

### 2. Network Security
- Restrict inbound to specific IPs if possible
- Use HTTPS for all communications
- Monitor access logs regularly
- Enable firewall with minimal open ports

### 3. System Requirements
- Keep system time synchronized (NTP)
- Monitor disk space (df -h)
- Check GPU temperature/utilization
- Set up log rotation

## Production Checklist

- [ ] NVIDIA Driver ≥545 installed
- [ ] Docker & NVIDIA Container Toolkit installed
- [ ] Public IP/DNS configured
- [ ] Firewall rules configured (inbound 8001, outbound 443)
- [ ] Environment file created with secure permissions
- [ ] `BLOCKCHAIN_ONLY=false` set
- [ ] `MODEL_QUANTIZATION=8bit` set for 20B model
- [ ] `RUNPOD_PUBLIC_IP` set to reachable address
- [ ] Node registered and visible in `/p2p/nodes`
- [ ] Local health check passing
- [ ] Inference test successful
- [ ] Monitoring/alerting configured
- [ ] Backup/recovery plan in place

## Important Notes

### Environment Variables
- **`.env` file support**: Added via python-dotenv (optional)
- **Direct environment**: Can use `export VAR=value` or Docker `--env-file`
- **BLOCKCHAIN_ONLY**: Defaults to `true` in code, MUST set to `false` for real model serving

### Endpoint Differences
- GPU node health: `GET /` (not `/health`)
- Inference: `POST /inference`
- Expert-specific: `POST /expert/{expert_name}/infer`
- Metrics: `GET /metrics`

## Monitoring Commands

```bash
# Watch logs continuously
docker logs -f blyan-node

# Check resource usage
docker stats blyan-node

# GPU monitoring
nvidia-smi -l 1

# Network connectivity test
nc -zv blyan.com 443

# Disk space check
df -h /var/lib/docker
```

## Support

For issues or questions:
1. Check logs: `docker logs blyan-node`
2. Verify network connectivity
3. Ensure all environment variables are set
4. Contact main node administrator for API key issues