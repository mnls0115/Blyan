#!/bin/bash
# RunPod 노드 설정 및 실행 스크립트
# GPT-OSS-20B 모델을 강제로 사용하도록 설정

echo "🚀 RunPod Blyan Node Setup Script"
echo "================================="

# 환경 변수 설정
export MODEL_NAME="openai/gpt-oss-20b"  # GPT-OSS-20B 강제 사용
export MODEL_QUANTIZATION="none"         # 양자화 비활성화 (GPU 메모리 충분)
export NODE_ID="runpod-20b-01"          # 노드 ID
export NODE_PORT="8002"                 # 포트 (8001이 사용중이면 8002로)

# Blyan 서버 설정
export MAIN_SERVER_URL="https://blyan.com/api"
export BLYAN_API_KEY="NZRLQqc7HcWVaLEjHLoaUyGBdODCzALjX6h5Bh28dmA"  # API 키

# HuggingFace 토큰 (있으면 설정)
# export HF_TOKEN="your_huggingface_token"

# 모델 관련 설정
export TRUST_REMOTE_CODE="1"            # Remote code 신뢰
export MODEL_AUTO_QUANTIZE="0"          # 자동 양자화 비활성화
export ALLOW_MOCK_FALLBACK="1"          # Mock 폴백 허용

# 퍼블릭 IP (RunPod가 자동으로 제공)
export RUNPOD_PUBLIC_IP="${RUNPOD_PUBLIC_IP:-$(curl -s ifconfig.me)}"

echo "📋 Configuration:"
echo "   Model: $MODEL_NAME"
echo "   Node ID: $NODE_ID"
echo "   Port: $NODE_PORT"
echo "   Main Server: $MAIN_SERVER_URL"
echo "   Public IP: $RUNPOD_PUBLIC_IP"
echo ""

# Python 가상환경 확인 및 생성
if [ ! -d ".venv" ]; then
    echo "🔧 Creating Python virtual environment..."
    python3 -m venv .venv
fi

# 가상환경 활성화
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# 필요한 패키지 설치
echo "📦 Installing required packages..."
pip install -q --upgrade pip
pip install -q --upgrade \
    torch torchvision torchaudio \
    transformers>=4.42 \
    tokenizers>=0.15.2 \
    huggingface_hub>=0.23 \
    safetensors \
    accelerate \
    bitsandbytes \
    fastapi \
    uvicorn \
    aiohttp \
    pydantic

# RunPod 노드 실행
echo ""
echo "🚀 Starting Blyan GPU Node..."
echo "================================="
python -u run_blyan_node.py