#!/usr/bin/env python3
"""Register gpu_node_2256 with the Redis-backed manager."""

import asyncio
import httpx
import sys
sys.path.insert(0, '/Users/mnls/projects/blyan')


async def register_node():
    """Register gpu_node_2256 with proper details."""
    
    print("Registering gpu_node_2256")
    print("=" * 60)
    
    # Node details from your RunPod instance
    node_data = {
        "node_id": "gpu_node_2256",
        "api_url": "https://wexdjh8n3k29kv-8000.proxy.runpod.net",
        "capabilities": {
            "gpu": "NVIDIA RTX 6000 Ada Generation",
            "gpu_memory_gb": 48,  # RTX 6000 Ada has 48GB
            "model": "Qwen/Qwen3-8B",
            "layers": list(range(32)),  # Full model support
            "dense_model_ready": True
        }
    }
    
    # You need to set your actual API key here
    api_key = "YOUR_BLYAN_API_KEY"  # Replace with actual key
    
    print(f"Node ID: {node_data['node_id']}")
    print(f"API URL: {node_data['api_url']}")
    print(f"GPU: {node_data['capabilities']['gpu']}")
    print(f"VRAM: {node_data['capabilities']['gpu_memory_gb']}GB")
    
    # First check if node is accessible
    print("\nChecking node health...")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{node_data['api_url']}/health")
            if response.status_code == 200:
                health = response.json()
                print(f"✅ Node is healthy: {health.get('status')}")
                print(f"   Model ready: {health.get('model_ready')}")
                print(f"   GPU available: {health.get('gpu_available')}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return
    except Exception as e:
        print(f"❌ Cannot reach node: {e}")
        return
    
    # Register with the API
    print("\nRegistering with Blyan API...")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://blyan.com/api/gpu/register",
                json=node_data,
                headers={"Authorization": f"Bearer {api_key}"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Registration successful!")
                print(f"   Message: {result.get('message', 'Registered')}")
                if result.get('layers_assigned'):
                    print(f"   Layers assigned: {result['layers_assigned']}")
            else:
                print(f"❌ Registration failed: {response.status_code}")
                print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Registration error: {e}")
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Make sure your GPU node sends regular heartbeats")
    print("2. Test inference with: curl -X POST https://blyan.com/api/chat")
    print("3. Monitor with: curl https://blyan.com/api/gpu/status")


if __name__ == "__main__":
    print("\n⚠️ IMPORTANT: Edit this script to add your actual BLYAN_API_KEY")
    print("The key should be the same one your GPU node uses.\n")
    
    # Uncomment after setting the API key
    # asyncio.run(register_node())