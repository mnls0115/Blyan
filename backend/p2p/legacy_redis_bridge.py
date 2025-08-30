"""
Self-healing bridge to sync legacy coordinator nodes to Redis.
This is a temporary migration tool that will be removed once all nodes use Redis directly.
"""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class LegacyToRedisBridge:
    """Bridge legacy in-memory nodes to Redis GPU manager."""
    
    def __init__(self, coordinator, gpu_manager):
        self.coordinator = coordinator
        self.gpu_manager = gpu_manager
        self.sync_interval = 30  # Sync every 30 seconds
        self.synthetic_heartbeat_ttl = 60  # Keep nodes alive for 60s
        self._sync_task = None
        self._running = False
        
    async def start(self):
        """Start the background sync task."""
        if self._sync_task is None:
            self._running = True
            self._sync_task = asyncio.create_task(self._sync_loop())
            logger.info("Started legacy->Redis bridge sync task")
    
    async def stop(self):
        """Stop the background sync task."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None
            logger.info("Stopped legacy->Redis bridge sync task")
    
    async def _sync_loop(self):
        """Continuously sync legacy nodes to Redis."""
        while self._running:
            try:
                await self.sync_once()
                await asyncio.sleep(self.sync_interval)
            except Exception as e:
                logger.error(f"Bridge sync error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def sync_once(self):
        """Perform one sync from legacy to Redis."""
        if not self.coordinator or not hasattr(self.coordinator, 'registry'):
            return
        
        if not self.coordinator.registry or not hasattr(self.coordinator.registry, 'nodes'):
            return
        
        legacy_nodes = self.coordinator.registry.nodes
        synced = 0
        skipped = 0
        
        for node_id, node in legacy_nodes.items():
            try:
                # Check if node is healthy (has recent heartbeat)
                if hasattr(node, 'last_heartbeat'):
                    age = time.time() - node.last_heartbeat
                    if age > 120:  # Skip nodes older than 2 minutes
                        skipped += 1
                        continue
                
                # Extract node details from legacy format
                api_url = self._extract_api_url(node)
                if not api_url:
                    skipped += 1
                    continue
                
                # Build capabilities from legacy node
                capabilities = {
                    "layers": getattr(node, 'layers', []),
                    "gpu_memory_gb": getattr(node, 'vram_gb', 16),
                    "model": "Qwen/Qwen3-8B"
                }
                
                # Check if already in Redis
                redis_nodes = await self.gpu_manager.get_active_nodes()
                already_exists = any(n['node_id'] == node_id for n in redis_nodes)
                
                if not already_exists:
                    # Register in Redis (without health check since we know it's active)
                    logger.info(f"Bridge: Syncing {node_id} from legacy to Redis")
                    
                    # Direct registration bypassing health check
                    await self._register_node_direct(
                        node_id=node_id,
                        api_url=api_url,
                        api_key=getattr(node, 'api_key', ''),
                        capabilities=capabilities
                    )
                    synced += 1
                else:
                    # Update heartbeat to keep it alive
                    await self.gpu_manager.update_heartbeat(node_id)
                    
            except Exception as e:
                logger.warning(f"Bridge: Failed to sync {node_id}: {e}")
        
        if synced > 0:
            logger.info(f"Bridge: Synced {synced} nodes from legacy to Redis ({skipped} skipped)")
    
    def _extract_api_url(self, node) -> Optional[str]:
        """Extract API URL from legacy node format."""
        # Try different attribute names
        if hasattr(node, 'endpoint'):
            return node.endpoint
        elif hasattr(node, 'api_url'):
            return node.api_url
        elif hasattr(node, 'host'):
            host = node.host
            if host.startswith('http'):
                return host
            else:
                port = getattr(node, 'port', 8000)
                return f"http://{host}:{port}"
        return None
    
    async def _register_node_direct(self, node_id: str, api_url: str, api_key: str, capabilities: dict):
        """Directly register node in Redis without health check."""
        import json
        
        node_info = {
            "node_id": node_id,
            "api_url": api_url.rstrip('/'),
            "api_key": api_key,
            "node_type": "gpu",
            "capabilities": capabilities,
            "status": "active",
            "registered_at": time.time(),
            "last_heartbeat": time.time(),
            "layers_assigned": capabilities.get("layers", []),
            "bridged": True  # Mark as bridged for debugging
        }
        
        # Store directly in Redis with TTL
        node_key = f"{self.gpu_manager.NODE_PREFIX}{node_id}"
        await self.gpu_manager.redis_client.setex(
            node_key,
            self.synthetic_heartbeat_ttl,
            json.dumps(node_info).encode()
        )
        
        # Add to sets
        await self.gpu_manager.redis_client.sadd(
            self.gpu_manager.ALL_NODES_SET,
            node_id.encode()
        )
        await self.gpu_manager.redis_client.sadd(
            self.gpu_manager.ACTIVE_NODES_SET,
            node_id.encode()
        )
        
        # Set synthetic heartbeat
        heartbeat_key = f"{self.gpu_manager.HEARTBEAT_PREFIX}{node_id}"
        await self.gpu_manager.redis_client.setex(
            heartbeat_key,
            self.synthetic_heartbeat_ttl,
            str(time.time()).encode()
        )
        
        logger.info(f"Bridge: Registered {node_id} with TTL={self.synthetic_heartbeat_ttl}s")


# Global bridge instance
_bridge_instance = None


async def start_legacy_bridge(coordinator, gpu_manager):
    """Start the legacy to Redis bridge."""
    global _bridge_instance
    
    if _bridge_instance is None:
        _bridge_instance = LegacyToRedisBridge(coordinator, gpu_manager)
        await _bridge_instance.start()
        logger.info("Legacy bridge started")
    return _bridge_instance


async def stop_legacy_bridge():
    """Stop the legacy bridge."""
    global _bridge_instance
    
    if _bridge_instance:
        await _bridge_instance.stop()
        _bridge_instance = None
        logger.info("Legacy bridge stopped")