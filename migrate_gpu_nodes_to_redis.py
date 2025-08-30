#!/usr/bin/env python3
"""Migrate GPU nodes from in-memory registry to Redis."""

import asyncio
import json
import sys
import time
sys.path.insert(0, '/Users/mnls/projects/blyan')


async def migrate_nodes():
    """Migrate existing GPU nodes to Redis-backed manager."""
    
    print("GPU Node Migration to Redis")
    print("=" * 60)
    
    # Get the current state from the API
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get("https://blyan.com/api/gpu/status")
        if response.status_code != 200:
            print("Failed to get current GPU status")
            return
        
        current_status = response.json()
        all_nodes = current_status.get("nodes", [])
        
        print(f"Found {len(all_nodes)} nodes to migrate")
        print(f"  - Active: {current_status.get('active_nodes', 0)}")
        print(f"  - Inactive: {current_status.get('inactive_nodes', 0)}")
    
    # Get Redis manager
    from backend.p2p.gpu_node_manager_redis import get_gpu_node_manager
    manager = await get_gpu_node_manager()
    
    # Check what's already in Redis
    redis_status = await manager.get_node_status()
    print(f"\nCurrent Redis state: {redis_status['total_nodes']} nodes")
    
    # Migrate each active node
    migrated = 0
    skipped = 0
    failed = 0
    
    for node in all_nodes:
        node_id = node.get("node_id")
        status = node.get("status", "inactive")
        
        # Only migrate active nodes
        if status != "active":
            print(f"  Skipping {node_id} (inactive)")
            skipped += 1
            continue
        
        print(f"\nMigrating {node_id}...")
        
        # Determine API URL based on node ID
        if node_id == "gpu_node_2256":
            # Your RunPod node
            api_url = "https://wexdjh8n3k29kv-8000.proxy.runpod.net"
            api_key = "test_key"  # You'll need to set the real key
        elif node_id == "test-gpu-1756539828":
            # Test node
            api_url = "http://localhost:8001"
            api_key = "test_key"
        elif node_id == "gpu_node_993":
            # Another node (might be stale)
            api_url = "http://localhost:8002"
            api_key = "test_key"
        else:
            print(f"  ⚠️ Unknown node {node_id}, skipping")
            skipped += 1
            continue
        
        # Build capabilities from node data
        capabilities = {
            "layers": node.get("layers", []),
            "gpu_memory_gb": 24,  # Default assumption
            "model": "Qwen/Qwen3-8B"
        }
        
        # Register in Redis (without health check since we know they're active)
        try:
            # Direct insert without health check
            node_info = {
                "node_id": node_id,
                "api_url": api_url,
                "api_key": api_key,
                "node_type": "gpu",
                "capabilities": capabilities,
                "status": "active",
                "registered_at": time.time(),
                "last_heartbeat": time.time(),
                "layers_assigned": node.get("layers", [])
            }
            
            # Store directly in Redis
            node_key = f"{manager.NODE_PREFIX}{node_id}"
            await manager.redis_client.setex(
                node_key,
                manager.NODE_DATA_TTL,
                json.dumps(node_info).encode()
            )
            
            # Add to sets
            await manager.redis_client.sadd(manager.ALL_NODES_SET, node_id.encode())
            await manager.redis_client.sadd(manager.ACTIVE_NODES_SET, node_id.encode())
            
            # Set heartbeat
            heartbeat_key = f"{manager.HEARTBEAT_PREFIX}{node_id}"
            await manager.redis_client.setex(
                heartbeat_key,
                manager.HEARTBEAT_TTL,
                str(time.time()).encode()
            )
            
            print(f"  ✅ Migrated {node_id}")
            migrated += 1
            
        except Exception as e:
            print(f"  ❌ Failed to migrate {node_id}: {e}")
            failed += 1
    
    # Final status
    print("\n" + "=" * 60)
    print("Migration Results:")
    print(f"  ✅ Migrated: {migrated}")
    print(f"  ⏭️ Skipped: {skipped}")
    print(f"  ❌ Failed: {failed}")
    
    # Check final Redis state
    final_status = await manager.get_node_status()
    print(f"\nFinal Redis state: {final_status['total_nodes']} nodes")
    print(f"  - Active: {final_status['active_nodes']}")
    
    if final_status['nodes']:
        print("\nNodes in Redis:")
        for node in final_status['nodes']:
            print(f"  - {node['node_id']}: {node['status']}")
    
    await manager.close()
    
    print("\nMigration complete!")
    print("\nIMPORTANT: Your GPU node needs to send heartbeats to stay active.")
    print("Make sure gpu_node_2256 is sending heartbeats to https://blyan.com/api/gpu/heartbeat")


if __name__ == "__main__":
    asyncio.run(migrate_nodes())