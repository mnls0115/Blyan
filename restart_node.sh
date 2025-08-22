#!/bin/bash

echo "🔄 Restarting GPU Node with Updated MoE Endpoints"
echo "================================================="

# Kill any existing node processes
echo "1. Stopping any existing nodes..."
pkill -f "python run_gpu_node.py" || true
sleep 2

# Start the node in background with logging
echo "2. Starting GPU node with updated endpoints..."
python run_gpu_node.py > gpu_node.log 2>&1 &
NODE_PID=$!

echo "   Node started with PID: $NODE_PID"
echo "   Logs: gpu_node.log"

# Wait for the node to initialize
echo "3. Waiting for node to start up..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8002/health > /dev/null 2>&1; then
        echo "   ✅ Node is responding!"
        break
    fi
    echo "   ⏳ Waiting for startup... ($i/30)"
    sleep 10
done

# Test the new endpoints
echo "4. Testing updated endpoints..."
echo ""
echo "   Health:"
curl -s http://127.0.0.1:8002/health

echo ""
echo "   MoE Status:"
curl -s http://127.0.0.1:8002/debug/moe-status

echo ""
echo "   Quick Chat Test:"
curl -X POST http://127.0.0.1:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "use_moe": true, "max_new_tokens": 30}'

echo ""
echo "5. Recent logs (last 10 lines):"
echo "   ==================================="
tail -10 gpu_node.log

echo ""
echo "✅ Restart complete!"
echo "   Node is running with PID: $NODE_PID"
echo "   Test with: ./quick_test.sh"
