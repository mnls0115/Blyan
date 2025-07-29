#!/bin/bash

# AI-Block MoE Test Runner
# Convenience script to run tests with proper environment

echo "🚀 AI-Block MoE Blockchain Test Runner"
echo "======================================"

# Activate virtual environment
source aiblock_env/bin/activate

# Check if API server is running
echo "🔍 Checking if API server is running..."
if curl -s http://127.0.0.1:8000/docs > /dev/null 2>&1; then
    echo "✅ API server is running"
else
    echo "⚠️  API server not running. Starting in background..."
    uvicorn api.server:app --reload --host 127.0.0.1 --port 8000 &
    SERVER_PID=$!
    echo "📝 Server PID: $SERVER_PID"
    
    # Wait for server to start
    echo "⏳ Waiting for server to start..."
    for i in {1..10}; do
        if curl -s http://127.0.0.1:8000/docs > /dev/null 2>&1; then
            echo "✅ API server started successfully"
            break
        fi
        sleep 2
        echo "   Attempt $i/10..."
    done
fi

echo ""
echo "🧪 Running MoE Blockchain Tests..."
echo "=================================="

# Check which test to run
if [ "$1" = "full" ]; then
    echo "🔄 Running full end-to-end test..."
    python scripts/demo_full_moe_flow.py
elif [ "$1" = "inference" ]; then
    echo "🧠 Running inference-only test..."
    python scripts/test_inference_only.py
elif [ "$1" = "debug" ]; then
    echo "🔧 Running debug test..."
    python scripts/test_inference_only.py debug
elif [ "$1" = "status" ]; then
    echo "📊 Checking expert status..."
    python scripts/test_inference_only.py status
else
    echo "📋 Available test options:"
    echo "  ./run_test.sh full      - Complete end-to-end test"
    echo "  ./run_test.sh inference - Inference testing only"
    echo "  ./run_test.sh debug     - Quick debug test"
    echo "  ./run_test.sh status    - Expert status check"
    echo ""
    echo "🧠 Running inference test by default..."
    python scripts/test_inference_only.py
fi

echo ""
echo "✅ Test completed!"
echo ""
echo "💡 Next steps:"
echo "  - Visit http://127.0.0.1:8000/docs for API documentation"
echo "  - Check ./test_data/ for generated files"
echo "  - Try uploading real MoE models with miner/upload_moe_parameters.py"