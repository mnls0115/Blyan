#!/bin/bash
# RunPod GPU Node Setup Script
set -e

echo "🚀 Setting up Blyan GPU Node on RunPod..."

# Configuration
WORKSPACE="/workspace/blyan"
RUNPOD_PROXY_URL="${RUNPOD_POD_ID}-8001.proxy.runpod.net"

# Update environment variables
cat > $WORKSPACE/.env << EOF
# Network Configuration
NODE_PORT=8000
PUBLIC_HOST=https://${RUNPOD_PROXY_URL}
PUBLIC_PORT=443
MAIN_NODE_URL=https://blyan.com/api

# Node Configuration  
BLYAN_NODE_ID=runpod_$(hostname)
BLOCKCHAIN_ONLY=true
SKIP_POL=true
AUTO_UPLOAD=false

# Data Paths
BLYAN_DATA_DIR=$WORKSPACE/data
HF_HOME=$WORKSPACE/.cache/huggingface
TRANSFORMERS_CACHE=$WORKSPACE/.cache/huggingface

# Model (for reference, not used in blockchain-only mode)
MODEL_NAME=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
EOF

echo "✅ Environment configured"

# Clean up any stale registrations
echo "🧹 Cleaning up old registrations..."
cd $WORKSPACE
source .venv/bin/activate

# Kill any running instances
pkill -f run_gpu_node.py || true
sleep 2

# Pull latest code if connected to git
if [ -d .git ]; then
    echo "📦 Pulling latest code..."
    git pull || echo "Git pull failed, continuing..."
fi

# Apply the fixes
echo "🔧 Applying fixes..."

# Start the node
echo "🚀 Starting GPU node..."
python run_gpu_node.py