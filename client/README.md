# Blyan Python Client SDK

Programmatic interface for Blyan AI Blockchain - no web interface needed!

## 🚀 Quick Start

### Installation
```bash
pip install aiohttp
```

### Basic Usage

1. **Simple Chat**
```python
import asyncio
from blyan_client import BlyanClient

async def main():
    async with BlyanClient("http://localhost:8000") as client:
        response = await client.chat("Hello, what is AI?")
        print(response)

asyncio.run(main())
```

2. **Run Interactive Chat** (No Web UI needed!)
```bash
cd examples
python simple_inference.py
```

3. **Contribute Your GPU**
```bash
cd examples
python run_expert_node.py
```

## 📋 API Reference

### BlyanClient

Main client for interacting with Blyan network.

#### Methods

- `chat(prompt, use_moe=False, use_distributed=False)` - Run inference
- `register_node(node)` - Register an expert node
- `list_nodes()` - List all active nodes
- `get_expert_stats(expert_name)` - Get expert performance stats
- `get_chain_blocks(chain_id)` - Get blockchain blocks

### BlyanNode

Configuration for an expert node.

```python
node = BlyanNode(
    node_id="my-gpu",
    host="192.168.1.100",
    port=8001,
    available_experts=["layer0.expert0"]
)
```

### NodeRunner

Helper to run a node with automatic heartbeat.

```python
runner = NodeRunner(node, api_url="http://api.blyan.com")
await runner.run()  # Runs until Ctrl+C
```

## 🔧 Advanced Examples

### Distributed Inference
```python
response = await client.chat(
    "Complex question here",
    use_distributed=True,
    use_secure=True,
    required_experts=["layer0.expert0", "layer1.expert1"]
)
```

### Custom API Key
```python
client = BlyanClient(
    api_url="http://api.blyan.com",
    api_key="your-api-key-here"
)
```

### Monitor Network
```python
# Get optimization insights
insights = await client.get_optimization_insights()

# View expert groups
groups = await client.get_expert_groups()

# Check top experts
top_experts = await client.get_top_experts(limit=5)
```

## 🌐 Network Endpoints

- Local development: `http://localhost:8000`
- Production: `http://api.blyan.com` (when deployed)

## 🤝 Contributing

To contribute compute power:

1. Ensure you have a GPU with sufficient memory
2. Configure your node with available experts
3. Run the node runner script
4. Keep your node online to earn rewards!

## 📝 License

Same as Blyan project - see main repository.