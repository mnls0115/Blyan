#!/bin/bash
# RunPod GPU Diagnostic Script

echo "================================================"
echo "🔍 RunPod GPU Diagnostics"
echo "================================================"
echo ""

# 1. System Info
echo "1️⃣ SYSTEM INFO"
echo "---------------"
echo "Hostname: $(hostname)"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "Kernel: $(uname -r)"
echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# 2. GPU Detection
echo "2️⃣ GPU DETECTION"
echo "----------------"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi is available"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv
else
    echo "❌ nvidia-smi NOT found - No NVIDIA driver!"
fi
echo ""

# 3. CUDA Check
echo "3️⃣ CUDA CHECK"
echo "--------------"
if [ -d "/usr/local/cuda" ]; then
    echo "✅ CUDA installation found at /usr/local/cuda"
    if [ -f "/usr/local/cuda/version.txt" ]; then
        echo "CUDA Version: $(cat /usr/local/cuda/version.txt)"
    fi
else
    echo "⚠️ CUDA not found at /usr/local/cuda"
fi

# Check nvcc
if command -v nvcc &> /dev/null; then
    echo "NVCC Version: $(nvcc --version | grep release | cut -d, -f2)"
else
    echo "⚠️ nvcc not found in PATH"
fi
echo ""

# 4. Python & PyTorch Check
echo "4️⃣ PYTHON & PYTORCH"
echo "--------------------"
python3 --version

echo ""
echo "Checking PyTorch GPU support..."
python3 -c "
import sys
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)')
else:
    print('❌ PyTorch cannot access GPU!')
    print('This usually means:')
    print('  1. You are on a CPU-only pod')
    print('  2. PyTorch was installed without CUDA support')
    print('  3. CUDA driver issues')
" 2>&1 || echo "❌ PyTorch test failed!"
echo ""

# 5. Memory Check
echo "5️⃣ MEMORY STATUS"
echo "-----------------"
free -h
echo ""

# 6. Disk Space
echo "6️⃣ DISK SPACE"
echo "--------------"
df -h / /tmp 2>/dev/null
echo ""

# 7. Network Check
echo "7️⃣ NETWORK CHECK"
echo "-----------------"
echo "Testing connection to main node..."
if curl -s --max-time 5 http://165.227.221.225:8000/health > /dev/null; then
    echo "✅ Can reach main node API"
else
    echo "❌ Cannot reach main node API"
fi
echo ""

# 8. Process Check
echo "8️⃣ RUNNING PROCESSES"
echo "---------------------"
echo "Python processes:"
ps aux | grep python | grep -v grep | head -5
echo ""

# 9. Environment Variables
echo "9️⃣ RELEVANT ENV VARS"
echo "---------------------"
env | grep -E "CUDA|GPU|PYTORCH|NODE" | sort
echo ""

# 10. Recommendations
echo "🔧 RECOMMENDATIONS"
echo "==================="
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ CRITICAL: No GPU driver detected!"
    echo "   - Make sure you selected a GPU pod, not CPU"
    echo "   - Try restarting the pod"
    echo "   - Contact RunPod support if issue persists"
elif ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "⚠️ PyTorch cannot see GPU!"
    echo "   Fix with:"
    echo "   pip uninstall torch torchvision torchaudio -y"
    echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
else
    echo "✅ GPU setup looks good!"
    echo "   Ready to run: ./run_gpu_node_fast.py"
fi
echo ""
echo "================================================"
echo "Diagnostics complete!"
echo "================================================"