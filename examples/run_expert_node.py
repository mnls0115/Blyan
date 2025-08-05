#!/usr/bin/env python3
"""
Run your computer as a Blyan expert node.
Contribute GPU power to the network!
"""

import asyncio
import sys
import socket
sys.path.append('..')  # Add parent directory to path

from client.blyan_client import BlyanNode, NodeRunner


def get_local_ip():
    """Get local IP address."""
    try:
        # Connect to a public DNS to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


async def main():
    # Configuration
    NODE_ID = "expert-node-1"  # Change this to a unique ID
    API_URL = "http://localhost:8000"  # or "http://api.blyan.com"
    NODE_PORT = 8001  # Port for your node
    
    # Available experts (adjust based on your GPU memory)
    AVAILABLE_EXPERTS = [
        "layer0.expert0",
        "layer0.expert1", 
        "layer1.expert0",
        "layer1.expert1"
    ]
    
    # Expert groups for optimization (optional)
    EXPERT_GROUPS = [{
        "experts": ["layer0.expert0", "layer0.expert1"],
        "usage_count": 0  # Will be updated based on usage
    }]
    
    print("🖥️  Blyan Expert Node Runner")
    print("=" * 50)
    
    # Get local IP
    local_ip = get_local_ip()
    print(f"📍 Local IP: {local_ip}")
    print(f"🔌 Port: {NODE_PORT}")
    print(f"🤖 Node ID: {NODE_ID}")
    print(f"📦 Available Experts: {len(AVAILABLE_EXPERTS)}")
    print(f"🌐 API Server: {API_URL}")
    print("=" * 50)
    
    # Create node configuration
    node = BlyanNode(
        node_id=NODE_ID,
        host=local_ip,
        port=NODE_PORT,
        available_experts=AVAILABLE_EXPERTS,
        expert_groups=EXPERT_GROUPS,
        region="us-west"  # Change to your region
    )
    
    # Create and run node runner
    runner = NodeRunner(
        node=node,
        api_url=API_URL,
        heartbeat_interval=30  # Send heartbeat every 30 seconds
    )
    
    print("\n✅ Starting expert node...")
    print("💡 Press Ctrl+C to stop\n")
    
    try:
        # Run node (blocks until interrupted)
        await runner.run()
    except KeyboardInterrupt:
        print("\n🛑 Stopping node...")
        runner.stop()
        print("👋 Node stopped successfully!")


if __name__ == "__main__":
    # Check if running as actual node server
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        # TODO: Implement actual expert inference server
        print("⚠️  Server mode not yet implemented")
        print("This would start the actual inference server on port", NODE_PORT)
    else:
        # Run as registration client
        asyncio.run(main())