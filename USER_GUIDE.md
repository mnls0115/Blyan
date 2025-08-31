# User Guide
# Getting Started with Blyan Network

## Why Blyan?

We built Blyan because AI shouldn't be controlled by a few corporations. Every AI decision should be transparent, verifiable, and owned by the community that uses it.

**What makes us different:**
- Every model weight is on the blockchain - fully transparent
- Anyone can contribute compute power and earn rewards
- The AI evolves through community contributions
- No black boxes, no hidden algorithms

## Quick Start

### 1. Try It Now (No Setup)
Visit [blyan.com](https://blyan.com) and start chatting immediately.

### 2. Connect Your GPU (Earn Rewards)

#### Docker Method (Easiest)
```bash
# Standardized Docker command (first run requires JOIN_CODE)
docker run -d --name blyan-node \
  --gpus all \
  -p 8001:8001 \
  -v /var/lib/blyan/data:/data \
  -e JOIN_CODE=YOUR_CODE_HERE \
  -e MAIN_NODE_URL=https://api.blyan.com \
  -e PUBLIC_HOST=$(curl -s https://checkip.amazonaws.com) \
  -e PUBLIC_PORT=8001 \
  -e JOB_CAPACITY=1 \
  -e NODE_ENV=production \
  --restart unless-stopped \
  mnls0115/blyan-node:latest

# Check it's running
docker logs -f blyan-node
```

Notes:
- CUDA 12.x base, runs as non-root inside the container.
- Credentials persist in `/data/credentials.json` after first enrollment.

#### Python Method
```bash
# Clone and setup
git clone https://github.com/blyan-network/blyan.git
cd blyan
./setup_gpu_fast.sh

# Configure
export MAIN_NODE_URL=https://blyan.com/api
export NODE_PORT=8001

# Run
python run_gpu_node.py
```

### 3. Use the API

#### Get API Access

API keys are required for authenticated access to the main API endpoints. They are NOT needed for running a GPU node.

Create an API Key:
```bash
curl -X POST https://api.blyan.com/auth/v2/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-api-key",
    "key_type": "basic",
    "metadata": {"purpose": "testing"}
  }'
```

Validate Your Key:
```bash
curl -X GET https://api.blyan.com/auth/v2/validate \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Refresh Your Key:
```bash
curl -X POST https://api.blyan.com/auth/v2/refresh \
  -H "Content-Type: application/json" \
  -d '{"current_key": "YOUR_CURRENT_API_KEY"}'
```

For detailed API key management instructions, see the [Contribute page](https://blyan.com/contribute#api-key-section).

#### Basic Usage
```python
import requests

# Simple chat with API key
api_key = "YOUR_API_KEY"
response = requests.post(
    "https://api.blyan.com/chat",
    json={"prompt": "Hello, Blyan!", "max_tokens": 100},
    headers={"Authorization": f"Bearer {api_key}"}
)
print(response.json()["text"])
```

#### JavaScript/Web
```javascript
const apiKey = 'YOUR_API_KEY';
const response = await fetch('https://api.blyan.com/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`
  },
  body: JSON.stringify({
    prompt: 'Hello, Blyan!',
    max_tokens: 100
  })
});
const data = await response.json();
console.log(data.text);
```

## GPU Node Operators

### Requirements
- **GPU**: 16GB+ VRAM (RTX 3090, 4090, A5000, etc.)
- **Internet**: 100Mbps+ with public IP or port forwarding
- **OS**: Linux (Ubuntu 20.04+ recommended)

### Earning Rewards
Your earnings depend on:
- **Uptime**: Stay online to receive more requests
- **Performance**: Faster inference = higher rewards
- **Quality**: Accurate responses earn bonus rewards

### Monitoring Your Node
```bash
# Check status
curl http://localhost:8001/

# View earnings
curl https://blyan.com/api/balance/your_address

# Monitor performance
docker stats blyan-node
```

### Optimization Tips
1. **Use FP8 quantization** for 2x speed
2. **Enable CUDA graphs** for lower latency
3. **Keep drivers updated** (NVIDIA 545+)
4. **Run multiple GPUs** for higher throughput

## Developers

### SDK Installation
```bash
# Python
pip install blyan-sdk

# JavaScript
npm install @blyan/sdk

# Go
go get github.com/blyan-network/go-sdk
```

### Example Applications

#### Chatbot
```python
from blyan import Client

client = Client()  # Uses free tier by default

def chatbot():
    while True:
        user_input = input("You: ")
        response = client.chat(user_input)
        print(f"Blyan: {response.text}")

chatbot()
```

#### Content Generation
```python
# Generate blog post
post = client.generate(
    prompt="Write a blog post about quantum computing",
    max_tokens=500,
    temperature=0.7
)

# Generate code
code = client.generate(
    prompt="Python function to sort a list",
    max_tokens=200,
    temperature=0.2
)
```

#### Streaming Responses
```python
# Stream tokens as they generate
for token in client.stream_chat("Tell me a story"):
    print(token, end='', flush=True)
```

## Mining & Contributing

### Submit Model Improvements
```python
# Train your improvement
improved_weights = train_model(dataset)

# Submit to network
result = client.mine(
    weights=improved_weights,
    validation_loss=0.92,
    training_proof=proof_data
)

print(f"Earned {result.reward_bly} BLY tokens!")
```

### Contribute Datasets
```bash
# Upload quality data
curl -X POST https://blyan.com/api/datasets/upload \
  -F "file=@dataset.jsonl" \
  -F "description=Medical Q&A pairs"
```

## Free Tier Limits

| Feature | Free | Standard | Pro |
|---------|------|----------|-----|
| Requests/day | 100 | 10,000 | Unlimited |
| Tokens/request | 100 | 1,000 | 4,000 |
| Streaming | ❌ | ✅ | ✅ |
| Priority | Low | Medium | High |
| Cost | $0 | $29/mo | $299/mo |

## Troubleshooting

### GPU Node Issues

**Node not connecting:**
```bash
# Check connectivity
curl https://blyan.com/api/health

# Check ports
sudo netstat -tlnp | grep 8001

# Check logs
docker logs blyan-node --tail 50
```

**Low earnings:**
- Ensure stable internet connection
- Check GPU utilization: `nvidia-smi`
- Verify node is registered: `curl https://blyan.com/api/p2p/nodes`

### API Issues

**Rate limited (429):**
- Reduce request frequency
- Upgrade to paid tier
- Implement exponential backoff

**Authentication failed (401):**
- Verify API key is correct
- Check key hasn't expired
- Ensure proper header format

## Security Best Practices

1. **Never share your private keys or API keys**
2. **Use environment variables for sensitive data**
3. **Keep your node software updated**
4. **Monitor for unusual activity**
5. **Use HTTPS for all API calls**

## Community & Support

### Get Help
- **Discord**: [discord.gg/blyan](https://discord.gg/blyan)
- **Documentation**: [docs.blyan.com](https://docs.blyan.com)
- **GitHub Issues**: [github.com/blyan-network/blyan/issues](https://github.com/blyan-network/blyan/issues)

### Stay Updated
- **Twitter**: [@BlyanNetwork](https://twitter.com/BlyanNetwork)
- **Blog**: [blog.blyan.com](https://blog.blyan.com)
- **Newsletter**: Subscribe at [blyan.com/newsletter](https://blyan.com/newsletter)

## What's Next?

### Our Roadmap
- **Q1 2025**: Mobile apps, 10,000+ nodes
- **Q2 2025**: Multi-modal AI (images, audio)
- **Q3 2025**: Full decentralization, DAO governance
- **2026**: 100B+ parameter models, cross-chain bridges

### Join the Revolution
Every interaction makes the network smarter. Every node makes it stronger. Every contribution makes it more democratic.

**Together, we're building AI that belongs to everyone.**

---
*Start your journey: [blyan.com](https://blyan.com)*
*Version 2.0.0 | January 2025*
