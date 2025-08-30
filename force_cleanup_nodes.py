#!/usr/bin/env python3
"""Force cleanup all GPU nodes except the currently running one."""

import asyncio
import httpx
import time
import os
import sys
sys.path.insert(0, '/Users/mnls/projects/blyan')


async def identify_and_cleanup():
    """Identify the real running node and remove all others."""
    
    print("GPU Node Cleanup")
    print("=" * 50)
    
    # First, check which nodes are listed
    async with httpx.AsyncClient() as client:
        response = await client.get("https://blyan.com/api/gpu/status")
        if response.status_code == 200:
            status = response.json()
            print(f"Current nodes: {status['total_nodes']} total, {status['active_nodes']} active")
            
            active_nodes = [n for n in status['nodes'] if n['status'] == 'active']
            inactive_nodes = [n for n in status['nodes'] if n['status'] == 'inactive']
            
            print(f"\nActive nodes to check:")
            for node in active_nodes:
                print(f"  - {node['node_id']}")
            
            print(f"\nInactive nodes to remove:")
            for node in inactive_nodes:
                print(f"  - {node['node_id']}")
    
    # Now connect to Redis directly and clean up
    from backend.p2p.gpu_node_manager_redis import GPUNodeManagerRedis
    
    # Use production Redis URL
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    print(f"\nConnecting to Redis at {redis_url}")
    
    manager = GPUNodeManagerRedis(
        redis_url=redis_url,
        stale_timeout_hours=0.001  # 3.6 seconds - very short for immediate cleanup
    )
    
    await manager.initialize()
    
    try:
        # Get all nodes from Redis
        all_node_ids = await manager.redis_client.smembers(manager.ALL_NODES_SET)
        print(f"\nNodes in Redis ALL_NODES_SET: {len(all_node_ids)}")
        
        # Get active nodes from Redis
        active_node_ids = await manager.redis_client.smembers(manager.ACTIVE_NODES_SET)
        print(f"Nodes in Redis ACTIVE_NODES_SET: {len(active_node_ids)}")
        
        # Check heartbeats
        nodes_to_remove = []
        current_time = time.time()
        
        for node_id_bytes in all_node_ids:
            node_id = node_id_bytes.decode()
            
            # Check heartbeat
            heartbeat_key = f"{manager.HEARTBEAT_PREFIX}{node_id}"
            heartbeat_exists = await manager.redis_client.exists(heartbeat_key)
            
            if heartbeat_exists:
                # Get heartbeat time
                hb_time = await manager.redis_client.get(heartbeat_key)
                if hb_time:
                    hb_timestamp = float(hb_time)
                    age = current_time - hb_timestamp
                    print(f"  {node_id}: heartbeat age = {age:.1f}s")
                    
                    # If heartbeat is old, mark for removal
                    if age > 60:  # More than 1 minute old
                        nodes_to_remove.append(node_id)
            else:
                print(f"  {node_id}: NO HEARTBEAT")
                nodes_to_remove.append(node_id)
        
        if nodes_to_remove:
            print(f"\nRemoving {len(nodes_to_remove)} stale nodes:")
            for node_id in nodes_to_remove:
                print(f"  Removing {node_id}...")
                success = await manager.unregister_node(node_id)
                if success:
                    print(f"    ✓ Removed")
                else:
                    print(f"    ✗ Failed")
        
        # Force cleanup any remaining stale nodes
        print("\nForcing cleanup of all inactive nodes...")
        result = await manager.cleanup_stale_nodes(force=True)
        print(f"Cleanup result: {result.get('total_removed', 0)} nodes removed")
        
        # Final check
        final_status = await manager.get_node_status()
        print(f"\nFinal status: {final_status['total_nodes']} total, {final_status['active_nodes']} active")
        
        if final_status['nodes']:
            print("Remaining nodes:")
            for node in final_status['nodes']:
                print(f"  - {node['node_id']}: {node['status']}")
    
    finally:
        await manager.close()
    
    print("\n" + "=" * 50)
    print("Cleanup complete!")
    
    # Show how to register your node
    print("\nTo register your GPU node, run:")
    print("  python run_gpu_node.py")
    print("\nOr manually register with:")
    print('  curl -X POST https://blyan.com/api/gpu/register \\')
    print('    -H "Authorization: Bearer YOUR_API_KEY" \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"node_id": "your-gpu-node", "api_url": "https://your-node-url", "capabilities": {...}}\'')


if __name__ == "__main__":
    asyncio.run(identify_and_cleanup())