# Blyan Docker Setup Guide

This guide explains how to run a Blyan GPU node using Docker, allowing you to contribute your computer's resources to the distributed AI network.

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- NVIDIA GPU with drivers (optional, for GPU acceleration)
- NVIDIA Container Toolkit (for GPU support)
- At least 20GB free disk space
- Stable internet connection

## Quick Start

### 1. Get Your JOIN_CODE

Visit [https://blyan.com/contribute](https://blyan.com/contribute) and click "Request Join Code" to get your enrollment code.

### 2. Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/mnls0115/Blyan.git
cd Blyan

# Copy environment template
cp .env.docker.example .env

# Edit .env and add your JOIN_CODE
nano .env  # or use your favorite editor

# Start the node
docker-compose up -d

# View logs
docker-compose logs -f
```

### 3. Using Docker Run

```bash
# Create data directory
sudo mkdir -p /var/lib/blyan/data
sudo chown $USER:$USER /var/lib/blyan/data

# Run with GPU support
docker run -d --name blyan-node \
  --gpus all \
  -p 8001:8001 \
  -v /var/lib/blyan/data:/data \
  -e JOIN_CODE=YOUR_CODE_HERE \
  -e PUBLIC_IP=$(curl -s https://checkip.amazonaws.com) \
  -e BLOCKCHAIN_ONLY=false \
  --restart unless-stopped \
  mnls0115/blyan-node:latest

# Run CPU-only (no GPU)
docker run -d --name blyan-node \
  -p 8001:8001 \
  -v /var/lib/blyan/data:/data \
  -e JOIN_CODE=YOUR_CODE_HERE \
  -e PUBLIC_IP=$(curl -s https://checkip.amazonaws.com) \
  -e BLOCKCHAIN_ONLY=false \
  --restart unless-stopped \
  mnls0115/blyan-node:latest
```

## Configuration Options

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `JOIN_CODE` | Enrollment code from blyan.com/contribute | - | Yes (first run) |
| `PUBLIC_IP` | Your public IP address | auto | No |
| `MAIN_NODE_URL` | Main node endpoint | http://165.227.221.225:8000 | No |
| `BLOCKCHAIN_ONLY` | Disable model serving | false | No |
| `NODE_NAME` | Friendly name for your node | gpu-node | No |
| `NODE_PORT` | Port for P2P communication | 8001 | No |
| `MODEL_NAME` | Model to serve | openai/gpt-oss-20b | No |
| `MODEL_QUANTIZATION` | Quantization level | int8 | No |
| `MAX_BATCH_SIZE` | Max inference batch size | 4 | No |
| `MAX_SEQUENCE_LENGTH` | Max token length | 2048 | No |

### Data Persistence

The node stores data in `/data` inside the container. Mount this to persist:
- Blockchain data
- Model weights (10-20GB)
- Node credentials
- Cache files

## GPU Support

### Installing NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Memory Requirements

- **GPU Mode**: 
  - Minimum: 8GB VRAM (RTX 3060 Ti, RTX 4060)
  - Recommended: 16GB+ VRAM (RTX 3090, RTX 4080)
  - Optimal: 24GB+ VRAM (RTX 4090, A100)

- **CPU Mode**:
  - Minimum: 16GB RAM
  - Recommended: 32GB+ RAM

## Managing Your Node

### View Status

```bash
# With docker-compose
docker-compose ps
docker-compose logs blyan-gpu-node

# With docker
docker ps
docker logs blyan-node
```

### Stop/Start Node

```bash
# With docker-compose
docker-compose stop
docker-compose start

# With docker
docker stop blyan-node
docker start blyan-node
```

### Update Node

```bash
# With docker-compose
docker-compose pull
docker-compose up -d

# With docker
docker pull mnls0115/blyan-node:latest
docker stop blyan-node
docker rm blyan-node
# Re-run with same command (credentials persist in volume)
```

### Check Health

```bash
# Check node health
curl http://localhost:8001/health

# Check GPU utilization
docker exec blyan-node nvidia-smi

# Check enrolled status
docker exec blyan-node cat /data/credentials/node_credentials.json
```

## Building from Source

```bash
# Clone repository
git clone https://github.com/mnls0115/Blyan.git
cd Blyan

# Build image
./docker/build.sh

# Build and push to registry
./docker/build.sh --push --tag v1.0.0
```

## Troubleshooting

### JOIN_CODE Issues

**Problem**: "Code has expired" immediately
- **Solution**: Sync your system clock: `sudo ntpdate -s time.nist.gov`

**Problem**: "Invalid or expired code"
- **Solution**: Request a new code, they expire after 30 minutes

### GPU Not Detected

**Problem**: Container doesn't see GPU
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Model Download Fails

**Problem**: Not enough space for model
```bash
# Check disk space
df -h /var/lib/blyan

# Clean up Docker
docker system prune -a

# Use different data directory with more space
DATA_DIR=/mnt/large-disk/blyan docker-compose up -d
```

### Network Issues

**Problem**: Cannot connect to main node
```bash
# Test connectivity
docker exec blyan-node curl http://165.227.221.225:8000/health

# Check firewall
sudo ufw status
sudo ufw allow 8001/tcp
```

## Security Considerations

1. **Credentials Protection**: Node credentials are stored in `/data/credentials/` with 600 permissions
2. **Network Security**: Only expose port 8001 for P2P, use firewall rules
3. **Container Isolation**: Run with `--security-opt=no-new-privileges`
4. **Resource Limits**: Set memory/CPU limits to prevent resource exhaustion

```bash
# Run with security options
docker run -d --name blyan-node \
  --gpus all \
  --security-opt=no-new-privileges \
  --memory=16g \
  --cpus=4 \
  -p 8001:8001 \
  -v /var/lib/blyan/data:/data:Z \
  -e JOIN_CODE=YOUR_CODE \
  mnls0115/blyan-node:latest
```

## Monitoring

### Prometheus Metrics

The node exposes metrics at `http://localhost:8001/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'blyan-node'
    static_configs:
      - targets: ['localhost:8001']
```

### Logging

```bash
# Set log level
-e LOG_LEVEL=DEBUG

# Export logs
docker logs blyan-node > blyan-node.log 2>&1

# Follow logs with filter
docker logs -f blyan-node 2>&1 | grep -E "ERROR|WARNING"
```

## Support

- GitHub Issues: [https://github.com/mnls0115/Blyan/issues](https://github.com/mnls0115/Blyan/issues)
- Documentation: [https://blyan.com/docs](https://blyan.com/docs)
- Community Discord: [https://discord.gg/blyan](https://discord.gg/blyan)

## License

MIT License - See LICENSE file for details