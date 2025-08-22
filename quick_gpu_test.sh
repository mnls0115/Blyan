#!/bin/bash
# Quick GPU Chat Test - Uses small model for fast testing

echo "🚀 Quick GPU Chat Test"
echo "====================="

# Kill any existing processes
echo "Stopping any existing nodes..."
pkill -f "python.*run_gpu" || true
sleep 2

# Start with small model for quick testing
echo "Starting GPU node with small model (GPT-2)..."
USE_SMALL_MODEL=true python run_gpu_node.py > test_gpu.log 2>&1 &
NODE_PID=$!

echo "Node PID: $NODE_PID"
echo ""

# Wait for startup (much faster with small model)
echo "Waiting for node to start (30 seconds max)..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8002/health > /dev/null 2>&1; then
        echo "✅ Node is ready!"
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# Check if it started
if ! curl -s http://127.0.0.1:8002/health > /dev/null 2>&1; then
    echo "❌ Node failed to start. Check logs: tail -f test_gpu.log"
    exit 1
fi

echo ""
echo "Testing endpoints:"
echo "=================="

# 1. Health check
echo "1. Health Check:"
curl -s http://127.0.0.1:8002/health | python3 -m json.tool 2>/dev/null | head -10

# 2. Quick chat test
echo ""
echo "2. Chat Test:"
curl -X POST http://127.0.0.1:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello! Tell me a joke", "max_new_tokens": 30}' \
  -s | python3 -m json.tool 2>/dev/null

echo ""
echo "✅ Test complete!"
echo ""
echo "Commands:"
echo "  Stop node:  kill $NODE_PID"
echo "  View logs:  tail -f test_gpu.log"
echo "  Test chat:  curl -X POST http://127.0.0.1:8002/chat -H 'Content-Type: application/json' -d '{\"prompt\": \"Your text here\"}'"