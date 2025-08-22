#!/bin/bash

echo "🚀 Starting GPU Node and Running MoE Debug"
echo "=========================================="

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "⚠️  jq not found, installing..."
    apt-get update && apt-get install -y jq
fi

echo ""
echo "1. Starting GPU Node in background..."
echo "   (This may take several minutes to load 6192 experts)"

# Start the GPU node in background
python run_gpu_node.py > gpu_node.log 2>&1 &
NODE_PID=$!

echo "   Node started with PID: $NODE_PID"
echo "   Logs: gpu_node.log"

# Wait for the node to start up
echo ""
echo "2. Waiting for node to initialize..."
sleep 10

# Check if node is responding
echo ""
echo "3. Checking if node is responding..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8002/health > /dev/null 2>&1; then
        echo "   ✅ Node is responding!"
        break
    fi
    echo "   ⏳ Waiting for node to start... ($i/30)"
    sleep 10
done

echo ""
echo "4. Running MoE debug tests..."

# Run the debug tests
echo ""
echo "   Testing health endpoint:"
curl -s http://127.0.0.1:8002/health | jq . 2>/dev/null || curl -s http://127.0.0.1:8002/health

echo ""
echo "   Testing MoE status:"
curl -s http://127.0.0.1:8002/debug/moe-status \
  -H "X-User-Address: 0x1234567890abcdef1234567890abcdef12345678" \
  -H "User-Agent: Debug-Script" | jq . 2>/dev/null || echo "   (jq not available, raw output:)" && curl -s http://127.0.0.1:8002/debug/moe-status \
  -H "X-User-Address: 0x1234567890abcdef1234567890abcdef12345678" \
  -H "User-Agent: Debug-Script"

echo ""
echo "   Testing chat with MoE:"
curl -X POST http://127.0.0.1:8002/chat \
  -H "Content-Type: application/json" \
  -H "X-User-Address: 0x1234567890abcdef1234567890abcdef12345678" \
  -d '{"prompt": "Hello world", "use_moe": true, "max_new_tokens": 50}' | jq . 2>/dev/null || echo "   (jq not available, raw output:)" && curl -X POST http://127.0.0.1:8002/chat \
  -H "Content-Type: application/json" \
  -H "X-User-Address: 0x1234567890abcdef1234567890abcdef12345678" \
  -d '{"prompt": "Hello world", "use_moe": true, "max_new_tokens": 50}'

echo ""
echo "5. Recent logs (last 20 lines):"
echo "   ========================================="
tail -20 gpu_node.log

echo ""
echo "6. Node is still running in background."
echo "   To stop it later: kill $NODE_PID"
echo "   To check status: curl http://127.0.0.1:8002/health"
echo ""
echo "✅ Debug complete!"
