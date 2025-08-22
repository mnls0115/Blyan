#!/bin/bash
# Optimized RunPod GPU Node Startup Script

echo "🚀 RunPod Optimized GPU Node Startup"
echo "===================================="

# 1. Check GPU availability
echo "1. Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✅ nvidia-smi found"
    nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv
else
    echo "   ❌ nvidia-smi not found - GPU may not be available!"
    echo "   Make sure you're on a GPU pod, not CPU!"
    exit 1
fi

# 2. Set environment for optimal GPU usage
echo ""
echo "2. Setting GPU environment..."
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"  # Support various GPU architectures
echo "   ✅ GPU environment configured"

# 3. Test PyTorch GPU access
echo ""
echo "3. Testing PyTorch GPU access..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'   ✅ PyTorch can see {torch.cuda.device_count()} GPU(s)')
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('   ❌ PyTorch cannot see GPU!')
    exit(1)
" || exit 1

# 4. Create minimal config for faster startup
echo ""
echo "4. Creating optimized configuration..."
cat > runpod_config.json <<EOF
{
    "model": "gpt2",
    "mode": "test",
    "skip_blockchain": true,
    "skip_expert_loading": true,
    "port": 8002
}
EOF
echo "   ✅ Config created"

# 5. Start minimal GPU node for testing
echo ""
echo "5. Starting minimal GPU node..."
echo "   (Using small model for quick testing)"

# Create a minimal test server
cat > minimal_gpu_server.py <<'PYEOF'
#!/usr/bin/env python3
import os
import torch
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global model state
model = None
gpu_info = {}

def init_gpu():
    """Initialize GPU info."""
    global gpu_info
    if torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "utilization": "active"
        }
    else:
        gpu_info = {"available": False, "error": "CUDA not available"}
    return gpu_info

def load_small_model():
    """Load a small model for testing."""
    global model
    try:
        from transformers import AutoModelForCausalLM
        logger.info("Loading small GPT-2 model for testing...")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        logger.info("✅ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    logger.info("Starting GPU node...")
    init_gpu()
    if gpu_info.get("available"):
        logger.info(f"GPU detected: {gpu_info['name']} with {gpu_info['memory_gb']:.1f} GB")
        load_small_model()
    else:
        logger.warning("No GPU detected - running in CPU mode")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpu": gpu_info,
        "model_loaded": model is not None
    })

@app.get("/gpu-status")
async def gpu_status():
    """Detailed GPU status."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return JSONResponse({
            "gpu_available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory": {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
            },
            "model_loaded": model is not None
        })
    else:
        return JSONResponse({"gpu_available": False, "error": "CUDA not available"})

@app.post("/test-inference")
async def test_inference():
    """Test GPU inference with small model."""
    if not model:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Test generation
        inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return JSONResponse({
            "success": True,
            "input": "Hello world",
            "output": result,
            "gpu_memory_used_gb": torch.cuda.memory_allocated() / 1e9
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    logger.info(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
PYEOF

echo "   ✅ Server script created"

# 6. Start the server
echo ""
echo "6. Starting server..."
python3 minimal_gpu_server.py &
SERVER_PID=$!

echo "   Server started with PID: $SERVER_PID"
echo ""
echo "7. Waiting for server to initialize..."
sleep 5

# 8. Test endpoints
echo ""
echo "8. Testing endpoints..."
echo "   Health check:"
curl -s http://localhost:8002/health | python3 -m json.tool

echo ""
echo "   GPU status:"
curl -s http://localhost:8002/gpu-status | python3 -m json.tool

echo ""
echo "✅ RunPod GPU node is running!"
echo ""
echo "Test commands:"
echo "  curl http://localhost:8002/health"
echo "  curl http://localhost:8002/gpu-status"
echo "  curl -X POST http://localhost:8002/test-inference"
echo ""
echo "Server PID: $SERVER_PID"
echo "Stop with: kill $SERVER_PID"