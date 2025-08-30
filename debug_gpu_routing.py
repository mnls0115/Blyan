#!/usr/bin/env python3
"""Debug GPU routing issue - why is gpu_node_2256 not receiving inference?"""

import asyncio
import json
import sys
sys.path.insert(0, '/Users/mnls/projects/blyan')

from backend.p2p.gpu_node_manager_redis import get_gpu_node_manager


async def debug_routing():
    """Debug why GPU node isn't receiving inference requests."""
    
    print("GPU Routing Debug")
    print("=" * 60)
    
    # Get manager
    manager = await get_gpu_node_manager()
    
    # 1. Check active nodes
    print("\n1. ACTIVE NODES CHECK")
    print("-" * 30)
    active_nodes = await manager.get_active_nodes()
    print(f"Total active nodes: {len(active_nodes)}")
    
    gpu_nodes = [n for n in active_nodes if n.get("node_type") == "gpu"]
    print(f"GPU-type nodes: {len(gpu_nodes)}")
    
    # Find our specific node
    our_node = None
    for node in active_nodes:
        if "2256" in node.get("node_id", ""):
            our_node = node
            break
    
    if our_node:
        print(f"\n✅ Found gpu_node_2256:")
        print(f"  - Node ID: {our_node.get('node_id')}")
        print(f"  - Node Type: {our_node.get('node_type', 'MISSING!')}")
        print(f"  - API URL: {our_node.get('api_url')}")
        print(f"  - Status: {our_node.get('status')}")
        print(f"  - Layers: {our_node.get('layers_assigned', [])}")
        print(f"  - Load: {our_node.get('current_load', 0)}")
        
        # Check if it's in GPU nodes list
        if our_node in gpu_nodes:
            print("  ✅ Node IS in GPU nodes list")
        else:
            print("  ❌ Node NOT in GPU nodes list - THIS IS THE PROBLEM!")
            print(f"     node_type field: '{our_node.get('node_type', 'MISSING')}'")
    else:
        print("\n❌ gpu_node_2256 not found in active nodes!")
    
    # 2. Test forwarding logic
    print("\n2. FORWARDING LOGIC TEST")
    print("-" * 30)
    
    # Simulate what forward_to_gpu does
    test_nodes = [n for n in active_nodes if n.get("node_type") == "gpu"]
    print(f"Nodes that would be considered for forwarding: {len(test_nodes)}")
    
    if test_nodes:
        print("Available GPU nodes for inference:")
        for node in test_nodes:
            print(f"  - {node.get('node_id')}: {node.get('api_url')}")
    else:
        print("❌ No nodes pass the node_type=='gpu' filter!")
        print("\nAll active nodes and their types:")
        for node in active_nodes:
            print(f"  - {node.get('node_id')}: type='{node.get('node_type', 'NONE')}'")
    
    # 3. Check raw Redis data
    print("\n3. RAW REDIS DATA")
    print("-" * 30)
    
    if our_node:
        node_key = f"{manager.NODE_PREFIX}{our_node['node_id']}"
        raw_data = await manager.redis_client.get(node_key)
        if raw_data:
            node_data = json.loads(raw_data)
            print(f"Raw Redis data for gpu_node_2256:")
            print(f"  node_type: '{node_data.get('node_type', 'MISSING')}'")
            print(f"  Full data keys: {list(node_data.keys())}")
        else:
            print("❌ No raw data in Redis for this node")
    
    # 4. Test selection
    print("\n4. SELECTION TEST")
    print("-" * 30)
    
    try:
        # Try to forward (will fail but show the error)
        result = await manager.forward_to_gpu(
            prompt="Test",
            max_tokens=5,
            temperature=0.1
        )
        print(f"✅ Forward succeeded: {result.get('node_id')}")
    except Exception as e:
        print(f"❌ Forward failed: {e}")
        
        # Check if it's because no GPU nodes
        if "No GPU nodes available" in str(e):
            print("\n⚠️ DIAGNOSIS: The node_type filter is blocking all nodes!")
            print("Solution: Either fix node_type in registration or remove filter")
    
    await manager.close()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(debug_routing())