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

### API Usage (Optional)
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