#!/bin/bash
# Start the main server with P2P enabled

echo "Starting server with P2P/distributed mode enabled..."

# Set environment variables
export P2P_ENABLE=true
export MINIMAL_MODE=false
export SKIP_DB_INIT=true
export PYTHONPATH=/root/dnai

# Kill any existing processes
pkill -9 -f "python.*api.server" || true

# Start the server
cd /root/dnai
source .venv/bin/activate

echo "Environment:"
echo "  P2P_ENABLE=$P2P_ENABLE"
echo "  MINIMAL_MODE=$MINIMAL_MODE"
echo ""

# Start with explicit P2P enable
P2P_ENABLE=true MINIMAL_MODE=false python -m api.server