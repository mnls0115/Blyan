#!/bin/bash
# Benchmark UV vs pip installation speed

echo "🏁 Benchmarking UV vs pip installation speed"
echo ""

# Create test requirements
cat > test_requirements.txt << 'EOF'
numpy
requests
pyyaml
tqdm
psutil
EOF

echo "Testing with a small set of packages:"
cat test_requirements.txt
echo ""

# Test pip
echo "⏱️  Testing pip..."
rm -rf test_pip_env
python3 -m venv test_pip_env
source test_pip_env/bin/activate
time pip install -r test_requirements.txt >/dev/null 2>&1
deactivate
pip_time=$SECONDS

echo ""

# Test UV
echo "⚡ Testing UV..."
# Install UV if not available
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    export PATH="$HOME/.cargo/bin:$PATH"
fi

rm -rf test_uv_env
SECONDS=0
uv venv test_uv_env >/dev/null 2>&1
source test_uv_env/bin/activate
uv pip install -r test_requirements.txt >/dev/null 2>&1
deactivate
uv_time=$SECONDS

echo ""
echo "📊 Results:"
echo "  pip:  ${pip_time}s"
echo "  UV:   ${uv_time}s"

if [ $uv_time -gt 0 ]; then
    speedup=$(( pip_time / uv_time ))
    echo "  UV is ~${speedup}x faster! 🚀"
else
    echo "  UV is blazingly fast! ⚡"
fi

# Clean up
rm -rf test_pip_env test_uv_env test_requirements.txt

echo ""
echo "For GPU nodes with large ML packages, the speedup is even more dramatic!"
echo "Use './setup_gpu_fast.sh' for ultra-fast GPU node setup."