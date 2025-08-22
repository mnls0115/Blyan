#!/bin/bash
# Test Qwen3-30B Model - For GPUs with 32GB+ memory

echo "🚀 Qwen3-30B GPU Test"
echo "===================="
echo "⚠️  Requires 32GB+ GPU memory!"
echo ""

# Check GPU memory first
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "❌ No NVIDIA GPU detected!"
    exit 1
fi

# Kill existing processes
echo "Stopping any existing nodes..."
pkill -f "python.*run_gpu" || true
sleep 2

# Start with full Qwen3-30B model
echo "Starting Qwen3-30B node..."
echo "(This will take 5-10 minutes to load)"
echo ""

# Set environment for better performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0

# Start the node
python run_gpu_node.py > qwen3_gpu.log 2>&1 &
NODE_PID=$!

echo "Node PID: $NODE_PID"
echo "Logs: tail -f qwen3_gpu.log"
echo ""

# Monitor startup progress
echo "Loading progress:"
for i in {1..120}; do  # 20 minutes max
    if curl -s http://127.0.0.1:8002/health > /dev/null 2>&1; then
        echo ""
        echo "✅ Node is ready!"
        break
    fi
    
    # Show progress every 30 seconds
    if [ $((i % 30)) -eq 0 ]; then
        echo ""
        echo "⏳ Still loading... ($((i/6))/20 minutes elapsed)"
        echo "Recent logs:"
        tail -2 qwen3_gpu.log 2>/dev/null | grep -v "^$"
    else
        echo -n "."
    fi
    sleep 10
done

# Check if started
if ! curl -s http://127.0.0.1:8002/health > /dev/null 2>&1; then
    echo ""
    echo "❌ Failed to start after 20 minutes"
    echo "Check logs: tail -20 qwen3_gpu.log"
    exit 1
fi

echo ""
echo "Testing Qwen3-30B:"
echo "=================="

# Test inference
echo "Chat test with Qwen3-30B:"
time curl -X POST http://127.0.0.1:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing in one sentence", "max_new_tokens": 50, "use_moe": true}' \
  -s | python3 -m json.tool

echo ""
echo "✅ Qwen3-30B is running!"
echo ""
echo "Commands:"
echo "  Stop:  kill $NODE_PID"
echo "  Logs:  tail -f qwen3_gpu.log"
echo "  Chat:  curl -X POST http://127.0.0.1:8002/chat -H 'Content-Type: application/json' -d '{\"prompt\": \"Your text\", \"use_moe\": true}'"