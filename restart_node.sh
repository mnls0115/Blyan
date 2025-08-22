#!/bin/bash

echo "🔄 Restarting GPU Node with Updated MoE Endpoints"
echo "================================================="

# Check GPU availability first
echo "0. Checking GPU status..."
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1
else
    echo "   ⚠️  nvidia-smi not found - GPU may not be available"
fi

# Kill any existing node processes
echo "1. Stopping any existing nodes..."
pkill -f "python run_gpu_node.py" || true
sleep 3

# Check if process is still running
if pgrep -f "python run_gpu_node.py" > /dev/null; then
    echo "   ⚠️  Force killing remaining processes..."
    pkill -9 -f "python run_gpu_node.py" || true
    sleep 2
fi

# Start the node in background with logging
echo "2. Starting GPU node with updated endpoints..."
echo "   (This may take 5-10 minutes to load 6192 experts)"
echo "   Check progress with: tail -f gpu_node.log"

python run_gpu_node.py > gpu_node.log 2>&1 &
NODE_PID=$!

echo "   Node started with PID: $NODE_PID"
echo "   Logs: gpu_node.log"

# Wait for the node to initialize (shorter timeout)
echo "3. Waiting for node to start up..."
for i in {1..60}; do  # 10 minutes max
    if curl -s http://127.0.0.1:8002/health > /dev/null 2>&1; then
        echo "   ✅ Node is responding!"
        break
    fi

    # Show progress every 10 iterations
    if [ $((i % 10)) -eq 0 ]; then
        echo "   ⏳ Still waiting... ($i/60) - $(($i * 10))s elapsed"
        echo "   Recent logs:"
        tail -3 gpu_node.log 2>/dev/null || echo "   No logs yet"
    fi

    sleep 10
done

# Check if we timed out
if ! curl -s http://127.0.0.1:8002/health > /dev/null 2>&1; then
    echo "   ❌ Node failed to start within 10 minutes"
    echo "   Checking if process is still running..."
    if ps -p $NODE_PID > /dev/null 2>&1; then
        echo "   Node process is still running (PID: $NODE_PID)"
        echo "   Check logs: tail -f gpu_node.log"
        echo "   Try again later or check GPU status"
    else
        echo "   Node process has crashed"
        echo "   Check logs: tail -20 gpu_node.log"
    fi
    exit 1
fi

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
