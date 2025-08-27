#!/bin/bash
# GPU Node Startup Script for Vast.ai or similar GPU providers
# Usage: ./start_gpu_node.sh

set -e

# Configuration
export SERVICE_NODE_URL="${SERVICE_NODE_URL:-http://165.227.221.225:8000}"
export BLYAN_API_KEY="${BLYAN_API_KEY:-$(python3 -c 'import secrets; print(secrets.token_hex(32))')}"
export GPU_PORT="${GPU_PORT:-8001}"
export NODE_ID="${NODE_ID:-gpu-$(hostname)-$(date +%s)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Blyan GPU Node Startup ===${NC}"
echo "Service Node: $SERVICE_NODE_URL"
echo "Node ID: $NODE_ID"
echo "GPU Port: $GPU_PORT"

# Check for GPU
if ! nvidia-smi &>/dev/null; then
    echo -e "${RED}ERROR: No GPU detected. This script requires a GPU.${NC}"
    exit 1
fi

echo -e "${GREEN}GPU detected:${NC}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# Check Python version
if ! python3 --version | grep -E "3\.(9|10|11|12|13)" &>/dev/null; then
    echo -e "${YELLOW}WARNING: Python 3.9+ recommended${NC}"
fi

# Install dependencies if needed
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install/upgrade dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers accelerate httpx uvicorn fastapi

# Check if we can connect to service node
echo -e "${YELLOW}Testing connection to service node...${NC}"
if curl -s -f "${SERVICE_NODE_URL}/health" > /dev/null; then
    echo -e "${GREEN}✓ Service node is accessible${NC}"
else
    echo -e "${RED}✗ Cannot reach service node at ${SERVICE_NODE_URL}${NC}"
    echo "Please check the URL and network connectivity."
    exit 1
fi

# Save configuration
cat > .env.gpu <<EOF
# GPU Node Configuration
SERVICE_NODE_URL=${SERVICE_NODE_URL}
BLYAN_API_KEY=${BLYAN_API_KEY}
GPU_PORT=${GPU_PORT}
NODE_ID=${NODE_ID}
IS_GPU_NODE=true
NODE_TYPE=gpu

# Model Configuration
MODEL_NAME=Qwen/Qwen3-8B
USE_BF16=true
MAX_BATCH_SIZE=4

# Performance
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
CUDA_VISIBLE_DEVICES=0
EOF

echo -e "${GREEN}Configuration saved to .env.gpu${NC}"

# Start the GPU node client
echo -e "${GREEN}Starting GPU node client...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}\n"

python3 backend/p2p/gpu_node_client.py