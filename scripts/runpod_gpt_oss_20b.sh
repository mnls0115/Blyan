#!/bin/bash
# Runpod GPT-OSS-20B Processing and Upload Script
# Run this on Runpod GPU instance to process and upload model

set -e  # Exit on error

echo "🚀 Blyan GPT-OSS-20B Runpod Upload Script"
echo "==========================================="

# Configuration
MAIN_NODE_URL="${AIBLOCK_API_URL:-https://blyan.com/api}"
MODEL_NAME="gpt-oss-20b"
WORK_DIR="/workspace/blyan"

# Validate required environment variables
if [[ -z "$BLYAN_NODE_ID" || -z "$BLYAN_MAIN_NODE_TOKEN" || -z "$META_HASH" ]]; then
    echo "❌ Required environment variables missing:"
    echo "   BLYAN_NODE_ID, BLYAN_MAIN_NODE_TOKEN, META_HASH"
    echo ""
    echo "Set them like:"
    echo "   export BLYAN_NODE_ID=main-blyan-0810"
    echo "   export BLYAN_MAIN_NODE_TOKEN=952f35cd..."  
    echo "   export META_HASH=<meta-block-hash>"
    exit 1
fi

# Create work directory
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "📋 Configuration:"
echo "   Main Node: $MAIN_NODE_URL"
echo "   Node ID: $BLYAN_NODE_ID"
echo "   Model: $MODEL_NAME"
echo "   Work Dir: $WORK_DIR"
echo ""

# Step 1: Setup environment
echo "📦 Step 1: Setting up environment..."
pip install torch transformers requests safetensors huggingface_hub
git clone https://github.com/mnls0115/Blyan.git blyan-repo || echo "Repo exists, pulling..."
cd blyan-repo && git pull && cd ..

# Step 2: Download GPT-OSS-20B model
echo "⬇️  Step 2: Downloading GPT-OSS-20B..."
if [[ ! -d "$MODEL_NAME" ]]; then
    echo "Downloading model (this will take a while)..."
    git lfs clone https://huggingface.co/RicardoLee/gpt-oss-20B-base $MODEL_NAME
else
    echo "Model already exists, skipping download"
fi

# Verify model files
if [[ ! -f "$MODEL_NAME/pytorch_model.bin" && ! -f "$MODEL_NAME/model.safetensors" ]]; then
    echo "❌ Model file not found!"
    exit 1
fi

echo "✅ Model ready: $(du -sh $MODEL_NAME)"

# Step 3: Test connection to main node
echo "🔗 Step 3: Testing connection to main node..."
python3 -c "
import requests
import os
import sys

session = requests.Session()
session.headers.update({
    'X-Node-ID': os.environ['BLYAN_NODE_ID'],
    'X-Node-Auth-Token': os.environ['BLYAN_MAIN_NODE_TOKEN']
})

try:
    response = session.get('$MAIN_NODE_URL/health', timeout=10)
    if response.status_code == 200:
        print('✅ Main node connection OK')
    else:
        print(f'❌ Health check failed: {response.status_code}')
        sys.exit(1)
        
    # Test auth
    response = session.get('$MAIN_NODE_URL/pol/status', timeout=10)  
    print(f'✅ Authentication OK: {response.status_code}')
    
except Exception as e:
    print(f'❌ Connection failed: {e}')
    sys.exit(1)
"

# Step 4: Run secure upload
echo "🚀 Step 4: Starting secure upload process..."

export MODEL_PATH="$WORK_DIR/$MODEL_NAME/pytorch_model.bin"
if [[ -f "$WORK_DIR/$MODEL_NAME/model.safetensors" ]]; then
    export MODEL_PATH="$WORK_DIR/$MODEL_NAME/model.safetensors"
fi

echo "Using model file: $MODEL_PATH"

# Copy upload script to workspace
cp blyan-repo/scripts/runpod_secure_upload.py ./

# Run upload
python3 runpod_secure_upload.py

echo ""
echo "✅ Upload process completed!"
echo ""
echo "📊 Next steps:"
echo "1. Check main node logs for upload confirmations"
echo "2. Verify blocks on blockchain explorer"
echo "3. Test distributed inference with new experts"
echo ""
echo "🔗 Explorer: $MAIN_NODE_URL/../explorer.html"

# Cleanup (optional)
read -p "Clean up downloaded model? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧹 Cleaning up model files..."
    rm -rf "$MODEL_NAME"
    echo "✅ Cleanup complete"
fi

echo "🎉 Runpod upload script completed!"