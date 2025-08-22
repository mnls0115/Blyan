#!/bin/bash

echo "🔍 Internal GPU Node MoE Debug (RunPod Container)"
echo "================================================="

echo ""
echo "1. Testing localhost (internal connection):"
curl -s http://127.0.0.1:8002/health | jq .

echo ""
echo "2. Testing MoE status (internal):"
curl -s http://127.0.0.1:8002/debug/moe-status \
  -H "X-User-Address: 0x1234567890abcdef1234567890abcdef12345678" \
  -H "User-Agent: Debug-Script" | jq .

echo ""
echo "3. Testing chat with MoE enabled:"
curl -X POST http://127.0.0.1:8002/chat \
  -H "Content-Type: application/json" \
  -H "X-User-Address: 0x1234567890abcdef1234567890abcdef12345678" \
  -d '{"prompt": "Hello world", "use_moe": true, "max_new_tokens": 50}' | jq .

echo ""
echo "4. Testing chat without MoE (fallback):"
curl -X POST http://127.0.0.1:8002/chat \
  -H "Content-Type: application/json" \
  -H "X-User-Address: 0x1234567890abcdef1234567890abcdef12345678" \
  -d '{"prompt": "Hello world", "use_moe": false, "max_new_tokens": 50}' | jq .

echo ""
echo "5. Check server logs for MoE initialization:"
echo "   Look for lines containing 'moe_model_manager' or 'experts'"
