#!/usr/bin/env python3
"""Clean up stale GPU nodes from Redis registry."""

import asyncio
import os
import json
import time
from datetime import datetime

import redis.asyncio as aioredis


async def cleanup_stale_nodes():
    """Remove inactive GPU nodes that haven't sent heartbeat in over 1 hour."""
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    print("Connecting to Redis...")
    client = await aioredis.from_url(redis_url, decode_responses=False)
    
    try:
        # Find all GPU node keys
        node_pattern = "gpu:node:*"
        heartbeat_pattern = "gpu:heartbeat:*"
        
        print(f"\nSearching for nodes matching pattern: {node_pattern}")
        
        # Get all node keys
        cursor = b'0'
        node_keys = []
        while cursor:
            cursor, keys = await client.scan(cursor, match=node_pattern.encode(), count=100)
            node_keys.extend(keys)
            if cursor == b'0':
                break
        
        print(f"Found {len(node_keys)} total nodes in Redis")
        
        # Check each node
        stale_nodes = []
        active_nodes = []
        
        for node_key in node_keys:
            node_id = node_key.decode().replace("gpu:node:", "")
            
            # Get node data
            node_data = await client.get(node_key)
            if node_data:
                node_info = json.loads(node_data)
                last_heartbeat = node_info.get('last_heartbeat', 0)
                
                # Check if heartbeat is stale (> 1 hour old)
                time_since_heartbeat = time.time() - last_heartbeat
                hours_ago = time_since_heartbeat / 3600
                
                # Check if heartbeat key exists
                heartbeat_key = f"gpu:heartbeat:{node_id}".encode()
                has_heartbeat = await client.exists(heartbeat_key)
                
                if has_heartbeat:
                    print(f"  ✓ {node_id}: ACTIVE (has heartbeat key)")
                    active_nodes.append(node_id)
                elif time_since_heartbeat > 3600:  # More than 1 hour
                    print(f"  ✗ {node_id}: STALE ({hours_ago:.1f} hours since last heartbeat)")
                    stale_nodes.append((node_id, node_key, hours_ago))
                else:
                    print(f"  ? {node_id}: RECENT ({hours_ago:.1f} hours ago, no heartbeat key)")
                    active_nodes.append(node_id)
        
        if not stale_nodes:
            print("\nNo stale nodes to clean up!")
            return
        
        print(f"\nFound {len(stale_nodes)} stale nodes to clean up")
        print(f"Active nodes to keep: {len(active_nodes)}")
        
        # Confirm before deletion
        response = input("\nDo you want to remove these stale nodes? (yes/no): ")
        
        if response.lower() == 'yes':
            # Clean up stale nodes
            async with client.pipeline() as pipe:
                for node_id, node_key, hours_ago in stale_nodes:
                    print(f"  Removing {node_id} ({hours_ago:.1f} hours stale)...")
                    
                    # Get node data to clean up layer assignments
                    node_data = await client.get(node_key)
                    if node_data:
                        node_info = json.loads(node_data)
                        
                        # Remove from layer assignments
                        for layer_id in node_info.get('layers_assigned', []):
                            layer_key = f"gpu:layer:{layer_id}"
                            pipe.srem(layer_key, node_id.encode())
                    
                    # Remove node data
                    pipe.delete(node_key)
                    pipe.delete(f"gpu:heartbeat:{node_id}".encode())
                    pipe.delete(f"gpu:metrics:{node_id}".encode())
                    pipe.srem("gpu:active_nodes", node_id.encode())
                
                await pipe.execute()
            
            print(f"\n✅ Cleaned up {len(stale_nodes)} stale nodes")
            
            # Verify cleanup
            remaining_nodes = []
            cursor = b'0'
            while cursor:
                cursor, keys = await client.scan(cursor, match=node_pattern.encode(), count=100)
                remaining_nodes.extend(keys)
                if cursor == b'0':
                    break
            
            print(f"Remaining nodes: {len(remaining_nodes)}")
            
        else:
            print("Cleanup cancelled")
    
    finally:
        await client.close()


async def list_current_status():
    """List current GPU node status after cleanup."""
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client = await aioredis.from_url(redis_url, decode_responses=False)
    
    try:
        print("\n" + "="*60)
        print("Current GPU Node Status")
        print("="*60)
        
        # Get all nodes
        cursor = b'0'
        node_keys = []
        while cursor:
            cursor, keys = await client.scan(cursor, match=b"gpu:node:*", count=100)
            node_keys.extend(keys)
            if cursor == b'0':
                break
        
        for node_key in sorted(node_keys):
            node_id = node_key.decode().replace("gpu:node:", "")
            node_data = await client.get(node_key)
            
            if node_data:
                node_info = json.loads(node_data)
                heartbeat_key = f"gpu:heartbeat:{node_id}".encode()
                has_heartbeat = await client.exists(heartbeat_key)
                
                status = "ACTIVE" if has_heartbeat else "INACTIVE"
                layers = node_info.get('layers_assigned', [])
                
                if isinstance(layers, list) and len(layers) > 3:
                    layers_str = f"{layers[:3]}... ({len(layers)} total)"
                else:
                    layers_str = str(layers)
                
                print(f"  {node_id:25} {status:8} Layers: {layers_str}")
        
        print(f"\nTotal nodes: {len(node_keys)}")
        
    finally:
        await client.close()


if __name__ == "__main__":
    print("GPU Node Registry Cleanup Tool")
    print("-" * 30)
    
    # Run cleanup
    asyncio.run(cleanup_stale_nodes())
    
    # Show final status
    asyncio.run(list_current_status())