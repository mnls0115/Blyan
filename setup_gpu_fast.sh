#!/bin/bash
# Fast GPU Node Setup with UV (10-100x faster than pip!)

set -e

echo "🚀 Setting up DNAI GPU Node with UV (Ultra-Fast Package Manager)"
echo ""

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add UV to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Source the new PATH
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
    
    echo "✅ UV installed successfully!"
else
    echo "✅ UV already installed"
fi

# Create virtual environment with UV
echo "🐍 Creating virtual environment..."
uv venv .venv --python 3.10
source .venv/bin/activate

echo "⚡ Installing PyTorch with CUDA support (UV is super fast!)..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "📚 Installing GPU node dependencies..."
uv pip install -r requirements-gpu.txt

echo "🔧 Setting up model configuration..."
# Source model config if available
if [ -f ".env.model" ]; then
    source .env.model
    echo "📋 Using model: ${MODEL_NAME:-Qwen/Qwen1.5-MoE-A2.7B}"
else
    echo "📋 Using default model: Qwen/Qwen1.5-MoE-A2.7B"
fi

echo "🎯 Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

try:
    import transformers
    print(f'Transformers version: {transformers.__version__}')
except:
    print('Transformers not available')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the GPU node:"
echo "  source .venv/bin/activate"
echo "  python3 run_gpu_node.py"
echo ""
echo "To change models quickly:"
echo "  ./switch_model.sh"
echo ""
echo "To clear model cache if needed:"
echo "  ./clear_cache.sh"