#!/bin/bash
# Complete MoE Model Upload Script for Blyan Evolution System
# This script uploads the tiny_mistral_moe model as individual Expert blocks

echo "🧬 Blyan Evolution System - Complete MoE Upload"
echo "============================================="

# Get meta hash
META_HASH=$(curl -s http://127.0.0.1:8000/chain/A/blocks | grep -o '"hash":"[^"]*"' | head -1 | cut -d'"' -f4)
echo "📦 Meta hash: $META_HASH"

# Check if model exists
if [ ! -d "./models/tiny_mistral_moe" ]; then
    echo "❌ Model not found: ./models/tiny_mistral_moe"
    exit 1
fi

echo "🔍 Model found: ./models/tiny_mistral_moe"

# First, run dry-run to see what will be uploaded
echo ""
echo "🧪 Running dry-run to analyze model structure..."
python miner/upload_moe_parameters.py \
    --address alice \
    --model-file ./models/tiny_mistral_moe \
    --meta-hash $META_HASH \
    --candidate-loss 0.8 \
    --dry-run

echo ""
echo "📊 Analysis complete. Press Enter to proceed with actual upload, or Ctrl+C to cancel..."
read -p ""

# Actual upload with development optimizations
echo ""
echo "🚀 Starting complete MoE upload..."
echo "   - Skipping PoW for faster upload"
echo "   - Reusing existing experts if any"
echo ""

python miner/upload_moe_parameters.py \
    --address alice \
    --model-file ./models/tiny_mistral_moe \
    --meta-hash $META_HASH \
    --candidate-loss 0.8 \
    --skip-pow \
    --reuse-existing

# Check results
echo ""
echo "🔍 Checking upload results..."
EXPERT_COUNT=$(curl -s http://127.0.0.1:8000/chain/B/blocks | jq length 2>/dev/null || echo "N/A")
echo "📈 Total Expert blocks in chain: $EXPERT_COUNT"

# Test the uploaded model
echo ""
echo "🧪 Testing MoE inference with uploaded experts..."
curl -X POST "http://localhost:8000/chat" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello evolved Blyan!", "use_moe": true, "top_k_experts": 2}' \
    -w "\nResponse time: %{time_total}s\n"

echo ""
echo "✅ Complete MoE upload finished!"
echo "🧬 Blyan is now ready for evolutionary inference!"