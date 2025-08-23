#!/bin/bash

echo "🔑 Blyan GPU Node API Key Generator"
echo "===================================="
echo ""
echo "This script generates an API key for a Blyan GPU node to connect to the main node."
echo ""

# Check if main node is accessible
echo "Checking main node availability..."
HEALTH_CHECK=$(curl -s https://blyan.com/api/health 2>/dev/null)

if [ -z "$HEALTH_CHECK" ]; then
    echo "❌ Cannot connect to main node at https://blyan.com/api"
    echo "Please ensure the main node is running."
    exit 1
fi

echo "✅ Main node is online"
echo ""

# Generate a unique identifier for this GPU node
TIMESTAMP=$(date +%s)
NODE_ID="gpu_node_${TIMESTAMP}"

echo "Generating API key for node: $NODE_ID"
echo ""

# Try to create API key using the keys endpoint
API_RESPONSE=$(curl -s -X POST "https://blyan.com/api/keys/create" \
  -H "Content-Type: application/json" \
  -d '{
    "key_type": "api_key",
    "description": "GPU Node '${NODE_ID}'"
  }' 2>/dev/null)

# Check if we got a valid response
if echo "$API_RESPONSE" | grep -q "api_key"; then
    API_KEY=$(echo "$API_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['api_key'])" 2>/dev/null)
    
    if [ ! -z "$API_KEY" ]; then
        echo "✅ API Key generated successfully!"
        echo ""
        echo "=========================================="
        echo "YOUR GPU NODE API KEY:"
        echo "$API_KEY"
        echo "=========================================="
        echo ""
        echo "To use this key in your RunPod deployment:"
        echo "1. Copy the API key above"
        echo "2. Run: ./deploy_to_runpod.sh"
        echo "3. Paste the API key when prompted"
        echo ""
        echo "⚠️  IMPORTANT: Save this key securely. You won't be able to retrieve it again."
        exit 0
    fi
fi

# If the keys endpoint doesn't work, provide manual instructions
echo "⚠️  Could not generate API key automatically."
echo ""
echo "The API key system may not be deployed on the main node yet."
echo ""
echo "For now, you can deploy the GPU node without an API key by:"
echo "1. Modifying the runpod_node.py to skip authentication"
echo "2. Or waiting for the API key system to be deployed"
echo ""
echo "To deploy the API key system on the main node:"
echo "1. SSH to the main node: ssh root@165.227.221.225"
echo "2. Deploy the auth system from the api_key_system.py"
echo ""