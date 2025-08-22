#!/bin/bash

echo "🔍 Quick MoE Test"
echo "================="

echo ""
echo "1. Health check:"
curl -s http://127.0.0.1:8002/health

echo ""
echo ""
echo "2. MoE Status:"
curl -s http://127.0.0.1:8002/debug/moe-status

echo ""
echo ""
echo "3. Chat test (MoE enabled):"
curl -X POST http://127.0.0.1:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "use_moe": true, "max_new_tokens": 50}'

echo ""
echo ""
echo "4. Chat test (MoE disabled):"
curl -X POST http://127.0.0.1:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "use_moe": false, "max_new_tokens": 50}'

echo ""
echo "✅ Test complete!"
