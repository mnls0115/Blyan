#!/bin/bash

# 🔧 CONFIGURE YOUR RUNPOD GPU NODE ADDRESS HERE
RUNPOD_IP="195.26.233.56"  # e.g., 123.456.78.90
RUNPOD_PORT="8002"               # Usually 8002 for GPU nodes

echo "🔍 Debugging Remote GPU Node MoE System Status"
echo "=============================================="
echo "Target: $RUNPOD_IP:$RUNPOD_PORT"
echo ""

if [ "$RUNPOD_IP" = "195.26.233.56" ]; then
    echo "❌ Please edit this script and replace '195.26.233.56' with your actual RunPod IP address!"
    echo "   Example: RUNPOD_IP=\"123.456.78.90\""
    echo ""
    echo "   You can find your RunPod IP in your RunPod dashboard or by running:"
    echo "   curl ifconfig.me"
    echo ""
    exit 1
fi

REMOTE_URL="http://$RUNPOD_IP:$RUNPOD_PORT"

echo "1. Checking server health:"
curl -s "$REMOTE_URL/health" | jq .

echo ""
echo "2. Checking MoE Status Endpoint:"
curl -s "$REMOTE_URL/debug/moe-status" \
  -H "X-User-Address: 0x1234567890abcdef1234567890abcdef12345678" \
  -H "User-Agent: Debug-Script" | jq .

echo ""
echo "3. Testing Chat with MoE enabled:"
curl -X POST "$REMOTE_URL/chat" \
  -H "Content-Type: application/json" \
  -H "X-User-Address: 0x1234567890abcdef1234567890abcdef12345678" \
  -d '{"prompt": "Hello world", "use_moe": true, "max_new_tokens": 50}' | jq .

echo ""
echo "4. Testing Chat with MoE disabled (fallback):"
curl -X POST "$REMOTE_URL/chat" \
  -H "Content-Type: application/json" \
  -H "X-User-Address: 0x1234567890abcdef1234567890abcdef12345678" \
  -d '{"prompt": "Hello world", "use_moe": false, "max_new_tokens": 50}' | jq .
