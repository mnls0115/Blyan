# API Documentation
# Blyan Network REST API v2.0

## Base URL
- Production: `https://blyan.com/api`
- Local: `http://localhost:8000`

## Authentication

### API Key
```bash
curl -H "X-API-Key: your_api_key_here" https://blyan.com/api/endpoint
```

### SIWE (Sign-In with Ethereum)
```javascript
// Frontend example
const message = await createSiweMessage(address, statement, chainId);
const signature = await signer.signMessage(message);
const response = await fetch('/auth/siwe', {
  method: 'POST',
  body: JSON.stringify({ message, signature })
});
```

## Core Endpoints

### Chat & Inference

#### POST `/chat`
**Generate AI response**
```bash
curl -X POST https://blyan.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "max_tokens": 100,
    "temperature": 0.7,
    "use_moe": true,
    "stream": false
  }'
```

**Response:**
```json
{
  "text": "Quantum computing uses quantum bits...",
  "tokens_used": 85,
  "experts_used": ["layer0.expert3", "layer1.expert7"],
  "cost_bly": 0.085,
  "latency_ms": 342
}
```

#### GET `/chat/stream`
**Stream response via SSE**
```javascript
const eventSource = new EventSource('/chat/stream?prompt=Hello');
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.token);
};
```

#### POST `/chat/atomic`
**Thread-safe atomic chat**
```json
{
  "prompt": "test",
  "session_id": "uuid-here",
  "context_window": 2048
}
```

### Mining & Model Upload

#### POST `/mine`
**Submit model improvement**
```bash
curl -X POST https://blyan.com/api/mine \
  -H "Content-Type: application/json" \
  -d '{
    "address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0",
    "block_data": "base64_encoded_weights",
    "nonce": 12345,
    "candidate_loss": 0.92
  }'
```

#### POST `/upload_parameters`
**Upload model parameters**
```python
import requests
import json

files = {
    'model_file': open('model.bin', 'rb'),
    'metadata': json.dumps({
        'model_name': 'gpt_oss_20b',
        'layer': 'layer_0',
        'version': '1.0.0'
    })
}

response = requests.post(
    'https://blyan.com/api/upload_parameters',
    files=files,
    headers={'X-API-Key': 'your_key'}
)
```

### P2P Network Management

#### POST `/p2p/register`
**Register GPU node**
```json
{
  "node_id": "gpu-node-001",
  "host": "203.0.113.42",
  "port": 8001,
  "available_experts": [
    "layer0.expert0",
    "layer0.expert1",
    "layer1.expert0"
  ],
  "capabilities": {
    "gpu_memory": 24576,
    "gpu_model": "RTX 4090",
    "quantization": ["fp16", "int8"]
  },
  "donor_mode": false
}
```

#### GET `/p2p/nodes`
**List active nodes**
```bash
curl https://blyan.com/api/p2p/nodes
```

**Response:**
```json
{
  "nodes": [
    {
      "node_id": "gpu-node-001",
      "status": "online",
      "load": 0.45,
      "available_experts": 12,
      "latency_ms": 23
    }
  ],
  "total": 47,
  "online": 42
}
```

#### POST `/p2p/heartbeat/{node_id}`
**Send heartbeat**
```bash
curl -X POST https://blyan.com/api/p2p/heartbeat/gpu-node-001 \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "gpu_usage": 89.1,
    "active_experts": 8
  }'
```

### Blockchain Operations

#### GET `/chain/{chain_id}/blocks`
**Get blockchain blocks**
```bash
# Get blocks from chain A (meta-chain)
curl "https://blyan.com/api/chain/A/blocks?offset=0&limit=10"

# Get blocks from chain B (parameter-chain)
curl "https://blyan.com/api/chain/B/blocks?offset=0&limit=10"
```

#### GET `/chain/{chain_id}/block/{index}`
**Get specific block**
```bash
curl https://blyan.com/api/chain/A/block/42
```

**Response:**
```json
{
  "index": 42,
  "hash": "0x3f4e5a...",
  "previous_hash": "0x2d3c4b...",
  "timestamp": 1704067200,
  "block_type": "expert",
  "data": {
    "expert_id": "layer0.expert3",
    "version": "1.2.0",
    "weight_cid": "QmXoypiz..."
  },
  "depends_on": ["0x1a2b3c..."],
  "signature": "0x9f8e7d..."
}
```

### Rewards & Economics

#### GET `/balance/{address}`
**Get BLY balance**
```bash
curl https://blyan.com/api/balance/0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0
```

**Response:**
```json
{
  "address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0",
  "balance": "15234.56",
  "pending_rewards": "123.45",
  "staked": "5000.00",
  "last_updated": 1704067200
}
```

#### POST `/transfer`
**Transfer BLY tokens**
```json
{
  "from": "0x742d35Cc...",
  "to": "0x8B3a4Df2...",
  "amount": "100.50",
  "nonce": 42,
  "signature": "0x1a2b3c..."
}
```

### Training & Learning

#### GET `/training/status`
**Get training job status**
```bash
curl https://blyan.com/api/training/status
```

**Response:**
```json
{
  "active": true,
  "current_epoch": 3,
  "current_round": 42,
  "loss": 0.234,
  "improvement": 0.023,
  "estimated_completion": "2024-01-15T10:30:00Z"
}
```

#### POST `/training/start`
**Start training job**
```json
{
  "dataset_id": "dataset_001",
  "expert_ids": ["layer0.expert0"],
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
  }
}
```

#### POST `/learning/delta/submit`
**Submit training delta**
```json
{
  "node_id": "gpu-node-001",
  "round_id": "round_42",
  "expert_id": "layer0.expert0",
  "delta_hash": "0xabc123...",
  "improvement_metric": 0.015,
  "validation_loss": 0.891
}
```

### Expert Management

#### GET `/layers/stats/{layer_name}`
**Get layer statistics**
```bash
curl https://blyan.com/api/layers/stats/layer0.expert3
```

**Response:**
```json
{
  "layer_name": "layer0.expert3",
  "usage_count": 15234,
  "total_tokens": 1523456,
  "average_quality": 0.94,
  "last_updated": "2024-01-15T10:00:00Z",
  "hosting_nodes": ["gpu-001", "gpu-005", "gpu-012"]
}
```

#### GET `/layers/top`
**Get top performing experts**
```bash
curl "https://blyan.com/api/layers/top?limit=10"
```

### Security & Monitoring

#### GET `/health`
**Health check**
```bash
curl https://blyan.com/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1704067200.123,
  "uptime_seconds": 86400,
  "api_response_time_ms": 2.3
}
```

#### GET `/security/dashboard`
**Security metrics**
```bash
curl https://blyan.com/api/security/dashboard
```

#### POST `/security/verify_expert`
**Verify expert integrity**
```json
{
  "expert_id": "layer0.expert0",
  "weight_hash": "0xdef456...",
  "activation_beacon": "test_input_123"
}
```

### Authentication V2

#### POST `/auth/v2/register`
**Register new API key**
```json
{
  "email": "user@example.com",
  "tier": "standard",
  "description": "Production key for app"
}
```

**Response:**
```json
{
  "api_key": "bly_k_1234567890abcdef",
  "tier": "standard",
  "rate_limit": {
    "requests_per_minute": 100,
    "tokens_per_day": 1000000
  },
  "expires_at": "2024-12-31T23:59:59Z"
}
```

## WebSocket Endpoints

### Real-time Updates
```javascript
const ws = new WebSocket('wss://blyan.com/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['blocks', 'nodes', 'rewards']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  switch(data.type) {
    case 'new_block':
      console.log('New block:', data.block);
      break;
    case 'node_update':
      console.log('Node status:', data.node);
      break;
  }
};
```

## Rate Limits

| Tier | Requests/Min | Tokens/Day | Cost/Month |
|------|-------------|------------|------------|
| Free | 10 | 10,000 | $0 |
| Standard | 100 | 1,000,000 | $29 |
| Professional | 1,000 | 10,000,000 | $299 |
| Enterprise | Unlimited | Unlimited | Custom |

## Error Codes

| Code | Description | Action |
|------|-------------|--------|
| 400 | Bad Request | Check request format |
| 401 | Unauthorized | Verify API key |
| 403 | Forbidden | Check permissions |
| 404 | Not Found | Verify endpoint |
| 429 | Rate Limited | Reduce request rate |
| 500 | Server Error | Retry with backoff |
| 502 | Bad Gateway | Service temporarily unavailable |
| 503 | Service Unavailable | Maintenance mode |

## SDK Examples

### Python
```python
from blyan import Client

client = Client(api_key='your_key')

# Chat
response = client.chat(
    prompt="Hello, world!",
    max_tokens=100
)

# Mining
result = client.mine(
    address='0x...',
    weights=model_weights,
    loss=0.92
)

# P2P
nodes = client.get_nodes()
```

### JavaScript
```javascript
import { BlyanClient } from '@blyan/sdk';

const client = new BlyanClient({ apiKey: 'your_key' });

// Chat
const response = await client.chat({
  prompt: 'Hello, world!',
  maxTokens: 100
});

// Streaming
const stream = await client.streamChat({
  prompt: 'Tell me a story',
  onToken: (token) => console.log(token)
});
```

### cURL
```bash
# Simple inference
curl -X POST https://blyan.com/api/chat \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "max_tokens": 50}'

# Upload expert
curl -X POST https://blyan.com/api/upload_moe_experts \
  -H "X-API-Key: $API_KEY" \
  -F "file=@expert.bin" \
  -F "metadata={\"layer\":\"layer0\",\"expert\":0}"

# Monitor node
while true; do
  curl https://blyan.com/api/p2p/nodes | jq '.online'
  sleep 5
done
```

## Pagination

Most list endpoints support pagination:

```bash
GET /endpoint?offset=0&limit=20&sort=created_at&order=desc
```

## Filtering

Use query parameters for filtering:

```bash
GET /p2p/nodes?status=online&min_gpu_memory=16384&region=us-west
```

## Batch Operations

Some endpoints support batch operations:

```json
POST /batch
{
  "operations": [
    {"method": "GET", "path": "/balance/0x123..."},
    {"method": "GET", "path": "/balance/0x456..."},
    {"method": "GET", "path": "/balance/0x789..."}
  ]
}
```

## Webhooks

Configure webhooks for events:

```json
POST /webhooks
{
  "url": "https://your-server.com/webhook",
  "events": ["block.created", "node.online", "reward.distributed"],
  "secret": "your_webhook_secret"
}
```

---
*API Version: 2.0.0*
*Last Updated: January 2025*