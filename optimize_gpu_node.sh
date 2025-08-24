#!/bin/bash

# Development mode launcher for BLYAN GPU node
# Production optimizations are now automatic in run_gpu_node.py

echo "🚀 Starting BLYAN GPU Node..."

# For development/testing only
if [ "$1" = "dev" ]; then
    echo "⚠️  Development mode enabled (security disabled)"
    export DEVELOPMENT_MODE=true
    export SKIP_POL=true
else
    echo "🔒 Production mode (default)"
fi

# Optional performance tuning (defaults are good)
# export BLOCK_FETCH_MAX_WORKERS=4      # Parallel fetching (default: 4)
# export SNAPSHOT_MAX_AGE_HOURS=12      # Snapshot validity (default: 12)

echo "✅ Auto-optimizations will be applied:"
echo "  - Fast startup with cached verification"
echo "  - Fused snapshots for instant loading" 
echo "  - Offline mode when model in blockchain"
echo "  - Parallel block fetching"

# Check if param_index exists
if [ -f "./data/param_index.json" ]; then
    LAYER_COUNT=$(python3 -c "import json; print(len(json.load(open('./data/param_index.json'))))" 2>/dev/null || echo "0")
    echo "📊 Parameter index has $LAYER_COUNT layers"
    
    if [ "$LAYER_COUNT" -ge "38" ]; then
        echo "✅ Full model available in blockchain - skipping HF download"
    fi
fi

# Check for fused snapshot
if [ -d "./data/models/fused" ]; then
    SNAPSHOT_COUNT=$(ls ./data/models/fused/*.safetensors 2>/dev/null | wc -l)
    if [ "$SNAPSHOT_COUNT" -gt "0" ]; then
        echo "✅ Found $SNAPSHOT_COUNT fused snapshot(s) - instant boot available"
    fi
fi

# Run the GPU node with optimizations
echo ""
echo "Starting GPU node with optimizations..."
python3 run_gpu_node.py