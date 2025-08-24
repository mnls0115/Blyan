#!/bin/bash
#
# Production Docker Entrypoint for Blyan Node
# Handles JOIN_CODE enrollment and credential management
#

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Blyan Node Starting ===${NC}"
echo "Node startup time: $(date)"

# Set defaults
export DATA_DIR="${DATA_DIR:-/data}"
export MAIN_SERVER_URL="${MAIN_SERVER_URL:-https://blyan.com/api}"
export NODE_PORT="${NODE_PORT:-8001}"
# Blockchain is always enabled for GPU nodes

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Check if we have existing credentials
CREDENTIALS_FILE="$DATA_DIR/credentials.json"

if [ -f "$CREDENTIALS_FILE" ]; then
    echo -e "${GREEN}✓ Found existing credentials${NC}"
    
    # Extract node_id and node_key
    export NODE_ID=$(python3 -c "import json; print(json.load(open('$CREDENTIALS_FILE'))['node_id'])")
    export NODE_KEY=$(python3 -c "import json; print(json.load(open('$CREDENTIALS_FILE'))['node_key'])")
    
    echo "Node ID: $NODE_ID"
    
elif [ -n "$JOIN_CODE" ]; then
    echo -e "${YELLOW}No credentials found, enrolling with JOIN_CODE...${NC}"
    
    # Run bootstrap script
    python3 /app/scripts/node_bootstrap.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Enrollment successful${NC}"
        
        # Load the newly created credentials
        export NODE_ID=$(python3 -c "import json; print(json.load(open('$CREDENTIALS_FILE'))['node_id'])")
        export NODE_KEY=$(python3 -c "import json; print(json.load(open('$CREDENTIALS_FILE'))['node_key'])")
        
        # Clear JOIN_CODE for security
        unset JOIN_CODE
    else
        echo -e "${RED}✗ Enrollment failed${NC}"
        echo "Please check your JOIN_CODE and try again"
        exit 1
    fi
    
else
    echo -e "${RED}================================================${NC}"
    echo -e "${RED}ERROR: No credentials and no JOIN_CODE provided${NC}"
    echo ""
    echo "To enroll this node:"
    echo "1. Go to https://blyan.com/contribute"
    echo "2. Click 'Request Join Code'"
    echo "3. Run this container with:"
    echo ""
    echo "   docker run -e JOIN_CODE=YOUR_CODE_HERE \\"
    echo "     -v /var/lib/blyan/data:/data \\"
    echo "     blyan/node:latest"
    echo ""
    echo -e "${RED}================================================${NC}"
    exit 1
fi

# Always check GPU availability (blockchain nodes still benefit from GPU acceleration)
echo -e "${YELLOW}Checking GPU availability...${NC}"

if nvidia-smi > /dev/null 2>&1; then
    echo -e "${GREEN}✓ GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}⚠ No GPU detected - running in CPU mode${NC}"
    echo "For GPU support, ensure NVIDIA drivers and Docker GPU runtime are installed"
fi

# Start the main node process
echo -e "${GREEN}Starting Blyan node service...${NC}"
echo "Main server: $MAIN_SERVER_URL"
echo "Node port: $NODE_PORT"
echo "Blockchain mode: Always enabled"

# Export credentials for the node process
export BLYAN_NODE_ID="$NODE_ID"
export BLYAN_NODE_KEY="$NODE_KEY"

# Check which main script to run
if [ -f "/app/run_gpu_node.py" ]; then
    # GPU node script (primary)
    exec python3 /app/run_gpu_node.py
elif [ -f "/app/run_blyan_node.py" ]; then
    # Production node script
    exec python3 /app/run_blyan_node.py
elif [ -f "/app/backend/p2p/node_runner.py" ]; then
    # P2P node runner
    exec python3 /app/backend/p2p/node_runner.py
else
    # Fallback to API server
    echo -e "${YELLOW}Running in API server mode${NC}"
    exec python3 /app/api/server.py
fi