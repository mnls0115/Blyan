# API Module Guidelines

## Overview
This directory contains the REST API server implementation for Blyan Network.

## Key Files
- `server.py` - Main FastAPI application with all endpoints
- `chat_atomic.py` - Thread-safe chat implementation
- `streaming.py` - SSE streaming for real-time responses
- `dense_learning_api.py` - Dense model training endpoints
- `rewards_api.py` - Token rewards endpoints

## Critical Reminders

### Production Code Standards
- **NO MOCK RESPONSES**: Never return hardcoded strings
- **NO TEST DATA**: All responses must come from real blockchain inference
- **PROPER ERROR CODES**: Return appropriate HTTP status codes
- **RATE LIMITING**: Implement real rate limiting, not placeholders

### Authentication
- Main node (blyan.com) bypasses authentication
- Regular nodes need X-API-Key header
- Health endpoint (`/health`) has no auth

### Inference Requirements
```python
# ❌ FORBIDDEN - Mock responses
@app.post("/chat")
async def chat():
    return {"text": "Hello, this is a test response"}  # NEVER!

# ✅ REQUIRED - Real blockchain inference
@app.post("/chat")
async def chat(request: ChatRequest):
    # Load model from blockchain
    model_manager = get_model_manager()
    if not model_manager.is_loaded:
        model_manager.load_from_blockchain()
    
    # Generate real response
    response = await model_manager.generate(
        prompt=request.prompt,
        max_tokens=request.max_tokens
    )
    
    # Track usage for rewards
    await track_inference_usage(response.tokens_used)
    
    return response
```

### Common Endpoints
```python
# Core inference
POST /chat - Generate response
GET /chat/stream - Stream tokens

# Blockchain
GET /chain/{chain_id}/blocks - Get blocks
POST /mine - Submit improvement

# P2P Network
POST /p2p/register - Register GPU node
GET /p2p/nodes - List active nodes

# Economics
GET /balance/{address} - Get BLY balance
POST /transfer - Transfer tokens
```

## Testing
```bash
# Test locally
curl http://localhost:8000/health

# Test with auth
curl -H "X-API-Key: test_key" http://localhost:8000/chat \
  -d '{"prompt": "test"}'
```

## Common Issues
- 502 errors: Check systemd service status
- Rate limiting: Implement exponential backoff
- CORS issues: Check allowed origins in server.py