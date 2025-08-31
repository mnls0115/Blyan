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

# Set defaults with standardized names
export BLYAN_DATA_DIR="${BLYAN_DATA_DIR:-/data}"
export MAIN_NODE_URL="${MAIN_NODE_URL:-https://blyan.com/api}"
export NODE_PORT="${NODE_PORT:-8001}"
export PUBLIC_HOST="${PUBLIC_HOST:-0.0.0.0}"
export PUBLIC_PORT="${PUBLIC_PORT:-8001}"
export JOB_CAPACITY="${JOB_CAPACITY:-1}"
# Blockchain is always enabled for GPU nodes

# Create data directory if it doesn't exist
mkdir -p "$BLYAN_DATA_DIR"

# Check if we have existing credentials
CREDENTIALS_FILE="$BLYAN_DATA_DIR/credentials.json"

if [ -f "$CREDENTIALS_FILE" ]; then
    echo -e "${GREEN}✓ Found existing credentials${NC}"
    
    # Extract node_id and node_key
    export NODE_ID=$(python3 -c "import json; print(json.load(open('$CREDENTIALS_FILE'))['node_id'])")
    export NODE_KEY=$(python3 -c "import json; print(json.load(open('$CREDENTIALS_FILE'))['node_key'])")
    
    echo "Node ID: $NODE_ID"
    # No JOIN_CODE required when credentials exist
    
elif [ -n "$JOIN_CODE" ]; then
    echo -e "${YELLOW}No credentials found, enrolling with JOIN_CODE...${NC}"
    
    # Run bootstrap script
    python3 /app/scripts/node_bootstrap.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Enrollment successful${NC}"
        
        # Load the newly created credentials
        export NODE_ID=$(python3 -c "import json; print(json.load(open('$CREDENTIALS_FILE'))['node_id'])")
        export NODE_KEY=$(python3 -c "import json; print(json.load(open('$CREDENTIALS_FILE'))['node_key'])")
        
        # Ensure credentials file has secure permissions
        chmod 0600 "$CREDENTIALS_FILE"
        
        # Clear JOIN_CODE for security
        unset JOIN_CODE
        echo -e "${GREEN}✓ JOIN_CODE cleared for security${NC}"
    else
        echo -e "${RED}✗ Enrollment failed${NC}"
        echo "Please check your JOIN_CODE and try again"
        exit 1
    fi
    
else
    echo -e "${RED}================================================${NC}"
    echo -e "${RED}ERROR: No credentials found and no JOIN_CODE provided${NC}"
    echo ""
    echo "This node requires either:"
    echo "  1. Pre-existing credentials at $BLYAN_DATA_DIR/credentials.json"
    echo "  2. A JOIN_CODE environment variable for enrollment"
    echo ""
    echo "To enroll this node:"
    echo "1. Go to https://blyan.com/contribute"
    echo "2. Click 'Request Join Code'"
    echo "3. Run this container with:"
    echo ""
    echo "   docker run -e JOIN_CODE=YOUR_CODE_HERE \\"
    echo "     -v /var/lib/blyan/data:/data \\"
    echo "     blyan/node:gpu"
    echo ""
    echo "The JOIN_CODE will be used once for enrollment and then cleared."
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

# Log configuration before starting
echo -e "${GREEN}Starting Blyan node service...${NC}"
echo "Configuration:"
echo "  Main node URL: $MAIN_NODE_URL"
echo "  Public host: $PUBLIC_HOST"
echo "  Public port: $PUBLIC_PORT"
echo "  Node port: $NODE_PORT"
echo "  Job capacity: $JOB_CAPACITY"
echo "  Data directory: $BLYAN_DATA_DIR"
echo "  Blockchain mode: Always enabled"

# Export credentials for the node process
export BLYAN_NODE_ID="$NODE_ID"
export BLYAN_NODE_KEY="$NODE_KEY"

# Prefer run_gpu_node.py as main entrypoint
if [ -f "/app/run_gpu_node.py" ]; then
    echo -e "${GREEN}Executing main GPU node process...${NC}"
    exec python3 /app/run_gpu_node.py
else
    echo -e "${RED}ERROR: /app/run_gpu_node.py not found${NC}"
    echo "Container build may be incomplete"
    exit 1
fi