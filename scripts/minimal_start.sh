#!/bin/bash

# Minimal startup script for recovery mode
# Use this when the server is failing to start due to subsystem initialization errors

echo "🔧 Starting server in MINIMAL/RECOVERY mode..."
echo "   This disables heavy subsystems for quick recovery"
echo ""

# Kill any existing processes
echo "Killing stray uvicorn processes..."
pkill -f uvicorn 2>/dev/null || true
sleep 2

# Set environment variables for minimal mode
export BLYAN_MINIMAL_MODE=true
export BLOCKCHAIN_ONLY=true
export DISABLE_PIPELINE_ROUND=true
export DISABLE_GRPC=true
export P2P_ENABLE=false
export BLYAN_DISABLE_SECURITY_MONITOR=true
export SKIP_POL=true
export ENABLE_POL=false
export SKIP_DB_INIT=true

# Optional: Set logging to see errors
export LOG_LEVEL=WARNING

echo "Environment flags set:"
echo "  BLYAN_MINIMAL_MODE=$BLYAN_MINIMAL_MODE"
echo "  BLOCKCHAIN_ONLY=$BLOCKCHAIN_ONLY"
echo "  DISABLE_PIPELINE_ROUND=$DISABLE_PIPELINE_ROUND"
echo "  DISABLE_GRPC=$DISABLE_GRPC"
echo "  P2P_ENABLE=$P2P_ENABLE"
echo "  BLYAN_DISABLE_SECURITY_MONITOR=$BLYAN_DISABLE_SECURITY_MONITOR"
echo ""

# Start the server
echo "Starting API server on port 8000..."
cd /root/blyan || cd /Users/mnls/projects/blyan || exit 1

# Use Python directly to avoid any wrapper issues
python -m api.server

# Alternative: Use uvicorn directly
# uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1