#!/bin/bash
# Clear HuggingFace cache to fix corrupted files

# Source model config if available
if [ -f ".env.model" ]; then
    source .env.model
fi

# Use environment variable or default
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen1.5-MoE-A2.7B"}

# Convert model name to cache format (replace / with --)
CACHE_NAME=$(echo $MODEL_NAME | sed 's/\//-/g')

echo "Clearing HuggingFace cache for model: $MODEL_NAME"

# Clear main HF cache
rm -rf ~/.cache/huggingface/hub/models--${CACHE_NAME}

# Clear local data cache if exists
rm -rf ./data/.hf/hub/models--${CACHE_NAME}
rm -rf /workspace/blyan/data/.hf/hub/models--${CACHE_NAME}

echo "Cache cleared. The model will be re-downloaded on next run."
echo ""
echo "To change the model, edit .env.model or set:"
echo "  export MODEL_NAME=EleutherAI/gpt-j-6b"
echo "  export MODEL_NAME=bigscience/bloom-7b1"