# Blyan: AI Blockchain Platform

An innovative platform that runs AI models directly on blockchain. Unlike centralized AI services, all AI models are stored on blockchain providing transparent and verifiable AI services.

## 🎯 Key Features

### 🌐 Website Features
- **`frontend/index.html`** → AI Chat: Chat with AI directly in your browser
- **`frontend/explorer.html`** → Block Explorer: View blockchain information
- **`frontend/contribute.html`** → AI Model Contribution: Upload new AI models
- **`frontend/dataset_explorer.html`** → Dataset Explorer: Browse training data
- **`frontend/chat.html`** → Advanced Chat: Use expert AI modes
- **`frontend/home.html`** → Main Page: Access all features

### 💡 AI Features
- **Smart AI Chat**: Conversation with AI running directly from blockchain
- **Expert AI Mode**: Choose AI specialists for specific domains
- **Transparent Processing**: All AI response processes verifiable on blockchain

### 🔗 Blockchain Features
- **Block Exploration**: View all AI models and data block by block
- **Model Contribution**: Users can directly upload and improve AI models
- **Verifiable**: All AI processing results recorded on blockchain, tamper-proof

## 🚀 Quick Start

### Environment Setup
```bash
# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Initial Setup
```bash
# Initialize meta-chain (one-time setup)
python -c "
import json
from pathlib import Path
from backend.core.chain import Chain

root_dir = Path('./data')
meta_chain = Chain(root_dir, 'A')
spec = {
    'model_name': 'distilbert-base-uncased',
    'architecture': 'mixture-of-experts',
    'num_layers': 4,
    'num_experts': 8,
    'routing_strategy': 'top2'
}
meta_chain.add_block(json.dumps(spec).encode(), block_type='meta')
print('✅ Meta-chain initialized successfully.')
"
```

### Using the Web Interface
1. **Start Backend Server**:
   ```bash
   python -m api.server
   ```

2. **Open Frontend**:
   - Open `frontend/index.html` file in your browser
   - Use the chat interface to interact with AI

## 💡 Usage Examples

### Basic AI Conversation
Simply enter prompts in the web interface and receive AI responses.

### Programmatic Usage (Python SDK)

You can interact with Blyan programmatically without using the web interface:

#### Install Client
```bash
pip install aiohttp
```

#### Basic Usage
```python
import asyncio
from client.blyan_client import BlyanClient

async def main():
    # Connect to Blyan API
    async with BlyanClient("http://localhost:8000") as client:
        # Run inference
        response = await client.chat(
            "What is artificial intelligence?",
            use_moe=True,
            top_k_experts=2
        )
        print(response)
        
        # Get expert statistics
        stats = await client.get_expert_stats("layer0.expert0")
        print(stats)

asyncio.run(main())
```

#### Register as Expert Node
```python
from client.blyan_client import BlyanNode, NodeRunner

# Define your node configuration
node = BlyanNode(
    node_id="my-gpu-node",
    host="192.168.1.100",  # Your node's IP
    port=8001,
    available_experts=["layer0.expert0", "layer1.expert0"]
)

# Run node with automatic heartbeat
runner = NodeRunner(node, api_url="http://api.blyan.com")
await runner.run()  # Runs until Ctrl+C
```

#### Distributed Inference
```python
async with BlyanClient("http://api.blyan.com") as client:
    # Run secure distributed inference
    response = await client.chat(
        "Explain quantum computing",
        use_distributed=True,
        use_secure=True,
        required_experts=["layer0.expert0", "layer1.expert1"]
    )
    print(response)
```

### API Usage (cURL)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "use_moe": true}'
```

### Blockchain Information Query
```bash
curl -X GET "http://localhost:8000/chain/A/blocks"
```

## 🏗️ System Architecture

- **Meta Chain (A)**: Stores AI model architecture and routing rules
- **Parameter Chain (B)**: Stores individual expert weights as DAG blocks
- **DAG Structure**: Dependency graph enabling parallel expert evolution
- **Selective Loading**: Load only necessary experts to memory for efficiency

## 📁 Key Files

- `frontend/index.html` - Web user interface
- `api/server.py` - REST API server
- `backend/core/` - Blockchain core logic
- `backend/model/` - AI model management
- `miner/` - Block creation tools

## 🎯 System Features

- **🔄 Autonomous Evolution**: Independent performance improvement at expert level
- **🤝 Distributed Cooperation**: P2P expert sharing and load balancing
- **📈 Continuous Learning**: Real-time performance monitoring and adaptive routing
- **🧬 Organic Growth**: Parallel expert development through DAG structure
- **💰 Economic Incentives**: Usage-based automatic reward distribution

## 📚 Additional Documentation

For more detailed information about the project, please refer to:
- `CLAUDE.md` - Developer guide
- `TESTING_GUIDE.md` - Testing methods
- `POL_SYSTEM_GUIDE.md` - Proof-of-Learning system

---

Blyan is an innovative platform for the future of transparent and decentralized AI. 🤖✨