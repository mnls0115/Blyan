"""GPU Node Manager with Redis Backend - Production Implementation
================================================================
Handles GPU node registration with shared state across API instances.
Uses Redis for distributed state management and automatic TTL-based health checks.
"""

import os
import time
import json
import asyncio
import logging
import hashlib
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import httpx
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.error("Redis not available - cannot use distributed GPU node manager")


@dataclass
class GPUNodeInfo:
    """Information about a registered GPU node."""
    node_id: str
    api_url: str
    api_key: str
    node_type: str = "gpu"  # 'gpu' or 'service'
    capabilities: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    status: str = "pending"  # pending, active, inactive, failed
    layers_assigned: List[int] = field(default_factory=list)
    current_load: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    average_latency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GPUNodeInfo':
        """Create from dictionary."""
        return cls(**data)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return 1.0 - (self.failed_requests / self.total_requests)
    
    @property
    def is_healthy(self) -> bool:
        """Check if node is healthy based on heartbeat."""
        return (time.time() - self.last_heartbeat) < 30  # 30 second timeout


class GPUNodeManagerRedis:
    """Redis-backed GPU node manager for distributed inference."""
    
    # Redis key prefixes
    NODE_PREFIX = "gpu:node:"
    LAYER_PREFIX = "gpu:layer:"
    HEARTBEAT_PREFIX = "gpu:heartbeat:"
    METRICS_PREFIX = "gpu:metrics:"
    LOCK_PREFIX = "gpu:lock:"
    
    # Configuration
    HEARTBEAT_TTL = 30  # seconds - how long before node is marked inactive
    NODE_DATA_TTL = 3600  # 1 hour for node data cache
    STALE_NODE_TIMEOUT = 3600  # 1 hour - remove nodes after this time without heartbeat
    LOCK_TTL = 10  # seconds for distributed locks
    
    def __init__(self, redis_url: Optional[str] = None, stale_timeout_hours: float = 1.0):
        """
        Initialize GPU node manager with Redis backend.
        
        Args:
            redis_url: Redis connection URL (defaults to REDIS_URL env var)
            stale_timeout_hours: Hours before inactive nodes are removed (default 1 hour)
        """
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis is required for distributed GPU node management")
        
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client: Optional[Redis] = None
        self._background_task = None
        self._initialized = False
        
        # Allow configurable stale timeout (convert hours to seconds)
        self.STALE_NODE_TIMEOUT = stale_timeout_hours * 3600
        
        # Get from environment if set
        env_timeout = os.getenv("GPU_NODE_STALE_TIMEOUT_HOURS")
        if env_timeout:
            try:
                self.STALE_NODE_TIMEOUT = float(env_timeout) * 3600
                logger.info(f"Using stale timeout from env: {env_timeout} hours")
            except ValueError:
                logger.warning(f"Invalid GPU_NODE_STALE_TIMEOUT_HOURS: {env_timeout}, using default")
        
        logger.info(f"🚀 GPU Node Manager initialized with Redis backend")
        logger.info(f"   Stale node timeout: {self.STALE_NODE_TIMEOUT/3600:.1f} hours")
    
    async def initialize(self):
        """Initialize Redis connection."""
        if self._initialized:
            return
        
        try:
            # Use standard redis.asyncio (redis>=4.5)
            self.redis_client = Redis.from_url(
                self.redis_url,
                decode_responses=False,
                max_connections=50
            )
            
            # Test connection
            await self.redis_client.ping()
            self._initialized = True
            
            # Start background tasks
            if not self._background_task:
                self._background_task = asyncio.create_task(self._cleanup_expired_nodes())
            
            logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def _ensure_initialized(self):
        """Ensure Redis is initialized before operations."""
        if not self._initialized:
            await self.initialize()
    
    async def register_node(
        self,
        node_id: str,
        api_url: str,
        api_key: str,
        capabilities: Optional[Dict[str, Any]] = None,
        node_type: str = "gpu"
    ) -> Dict[str, Any]:
        """Register a new GPU node with distributed locking."""
        await self._ensure_initialized()
        
        # Acquire distributed lock for registration
        lock_key = f"{self.LOCK_PREFIX}register:{node_id}"
        lock_acquired = await self._acquire_lock(lock_key)
        
        if not lock_acquired:
            return {
                "success": False,
                "message": "Registration already in progress"
            }
        
        try:
            # Validate node is accessible
            async with httpx.AsyncClient(timeout=10) as client:
                headers = {"Authorization": f"Bearer {api_key}"}
                response = await client.get(f"{api_url}/health", headers=headers)
                
                if response.status_code != 200:
                    raise ValueError(f"Node health check failed: {response.status_code}")
            
            # Create node info
            node = GPUNodeInfo(
                node_id=node_id,
                api_url=api_url.rstrip('/'),
                api_key=api_key,
                node_type=node_type,
                capabilities=capabilities or {},
                status="active",
                registered_at=time.time(),
                last_heartbeat=time.time()
            )
            
            # Assign layers if GPU node
            if node_type == "gpu" and capabilities:
                layers = capabilities.get('layers', [])
                gpu_memory = capabilities.get('gpu_memory_gb', 0)
                
                # Auto-assign layers if not specified
                if not layers and gpu_memory >= 16:
                    # Get current node count for distribution
                    existing_nodes = await self._get_all_gpu_nodes()
                    num_nodes = len([n for n in existing_nodes if n["node_type"] == "gpu"]) + 1
                    
                    # Distribute 32 layers across nodes
                    layers_per_node = max(1, 32 // num_nodes)
                    start_layer = (num_nodes - 1) * layers_per_node
                    layers = list(range(start_layer, min(32, start_layer + layers_per_node)))
                
                node.layers_assigned = layers
            
            # Store in Redis with pipeline for atomicity
            async with self.redis_client.pipeline() as pipe:
                # Store node data
                node_key = f"{self.NODE_PREFIX}{node_id}"
                pipe.setex(
                    node_key,
                    self.NODE_DATA_TTL,
                    json.dumps(node.to_dict()).encode()
                )
                
                # Store heartbeat with TTL
                heartbeat_key = f"{self.HEARTBEAT_PREFIX}{node_id}"
                pipe.setex(heartbeat_key, self.HEARTBEAT_TTL, str(time.time()).encode())
                
                # Update layer assignments
                for layer_id in node.layers_assigned:
                    layer_key = f"{self.LAYER_PREFIX}{layer_id}"
                    pipe.sadd(layer_key, node_id.encode())
                    pipe.expire(layer_key, self.NODE_DATA_TTL)
                
                # Add to active nodes set
                pipe.sadd("gpu:active_nodes", node_id.encode())
                
                # Execute pipeline
                await pipe.execute()
            
            # Publish registration event for other instances
            await self.redis_client.publish(
                "gpu:events",
                json.dumps({
                    "event": "node_registered",
                    "node_id": node_id,
                    "timestamp": time.time()
                }).encode()
            )
            
            logger.info(
                f"✅ Registered {node_type} node {node_id} "
                f"with {len(node.layers_assigned)} layers"
            )
            
            return {
                "success": True,
                "message": f"Node {node_id} registered successfully",
                "node_id": node_id,
                "layers_assigned": node.layers_assigned,
                "status": node.status
            }
            
        except Exception as e:
            logger.error(f"Failed to register node {node_id}: {e}")
            return {
                "success": False,
                "message": f"Registration failed: {str(e)}"
            }
        finally:
            await self._release_lock(lock_key)
    
    async def unregister_node(self, node_id: str) -> bool:
        """Unregister a GPU node."""
        await self._ensure_initialized()
        
        try:
            # Get node info
            node_data = await self._get_node_data(node_id)
            if not node_data:
                return False
            
            async with self.redis_client.pipeline() as pipe:
                # Remove from layer assignments
                for layer_id in node_data.get("layers_assigned", []):
                    layer_key = f"{self.LAYER_PREFIX}{layer_id}"
                    pipe.srem(layer_key, node_id.encode())
                
                # Remove node data and heartbeat
                pipe.delete(f"{self.NODE_PREFIX}{node_id}")
                pipe.delete(f"{self.HEARTBEAT_PREFIX}{node_id}")
                pipe.delete(f"{self.METRICS_PREFIX}{node_id}")
                
                # Remove from active nodes
                pipe.srem("gpu:active_nodes", node_id.encode())
                
                await pipe.execute()
            
            # Publish event
            await self.redis_client.publish(
                "gpu:events",
                json.dumps({
                    "event": "node_unregistered",
                    "node_id": node_id,
                    "timestamp": time.time()
                }).encode()
            )
            
            logger.info(f"📤 Unregistered node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister node {node_id}: {e}")
            return False
    
    async def update_heartbeat(self, node_id: str) -> bool:
        """Update node heartbeat with Redis TTL."""
        await self._ensure_initialized()
        
        try:
            async with self.redis_client.pipeline() as pipe:
                # Update heartbeat with TTL
                heartbeat_key = f"{self.HEARTBEAT_PREFIX}{node_id}"
                pipe.setex(heartbeat_key, self.HEARTBEAT_TTL, str(time.time()).encode())
                
                # Update node status
                node_key = f"{self.NODE_PREFIX}{node_id}"
                node_data = await self._get_node_data(node_id)
                
                if node_data:
                    node_data["last_heartbeat"] = time.time()
                    node_data["status"] = "active"
                    pipe.setex(
                        node_key,
                        self.NODE_DATA_TTL,
                        json.dumps(node_data).encode()
                    )
                
                await pipe.execute()
                return True
            
        except Exception as e:
            logger.error(f"Failed to update heartbeat for {node_id}: {e}")
            return False
    
    async def update_metrics(
        self,
        node_id: str,
        load: Optional[float] = None,
        latency: Optional[float] = None,
        success: Optional[bool] = None
    ):
        """Update node metrics."""
        await self._ensure_initialized()
        
        try:
            metrics_key = f"{self.METRICS_PREFIX}{node_id}"
            
            # Get current metrics
            metrics_data = await self.redis_client.get(metrics_key)
            if metrics_data:
                metrics = json.loads(metrics_data)
            else:
                metrics = {
                    "current_load": 0.0,
                    "total_requests": 0,
                    "failed_requests": 0,
                    "average_latency": 0.0
                }
            
            # Update metrics
            if load is not None:
                metrics["current_load"] = load
            
            if success is not None:
                metrics["total_requests"] += 1
                if not success:
                    metrics["failed_requests"] += 1
            
            if latency is not None:
                # Exponential moving average
                metrics["average_latency"] = (
                    metrics["average_latency"] * 0.9 + latency * 0.1
                )
            
            # Store updated metrics
            await self.redis_client.setex(
                metrics_key,
                self.NODE_DATA_TTL,
                json.dumps(metrics).encode()
            )
            
        except Exception as e:
            logger.error(f"Failed to update metrics for {node_id}: {e}")
    
    async def get_nodes_for_layer(self, layer_id: int) -> List[Dict[str, Any]]:
        """Get all healthy nodes that can handle a specific layer."""
        await self._ensure_initialized()
        
        try:
            layer_key = f"{self.LAYER_PREFIX}{layer_id}"
            node_ids = await self.redis_client.smembers(layer_key)
            
            nodes = []
            for node_id_bytes in node_ids:
                node_id = node_id_bytes.decode()
                
                # Check if node has active heartbeat
                heartbeat_key = f"{self.HEARTBEAT_PREFIX}{node_id}"
                if await self.redis_client.exists(heartbeat_key):
                    node_data = await self._get_node_data(node_id)
                    if node_data:
                        nodes.append(node_data)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to get nodes for layer {layer_id}: {e}")
            return []
    
    async def get_best_node_for_layer(self, layer_id: int) -> Optional[Dict[str, Any]]:
        """Get the best node for a specific layer based on load and performance."""
        nodes = await self.get_nodes_for_layer(layer_id)
        if not nodes:
            return None
        
        # Get metrics for each node
        for node in nodes:
            metrics = await self._get_node_metrics(node["node_id"])
            node.update(metrics)
        
        # Score nodes
        def node_score(node: Dict[str, Any]) -> float:
            score = 0.0
            if node.get("status") == "active":
                score += 10
            
            # Lower load is better
            score += (1.0 - node.get("current_load", 0)) * 5
            
            # Higher success rate is better
            total = node.get("total_requests", 0)
            failed = node.get("failed_requests", 0)
            if total > 0:
                success_rate = 1.0 - (failed / total)
                score += success_rate * 3
            
            # Lower latency is better
            score -= node.get("average_latency", 0) / 1000
            
            return score
        
        return max(nodes, key=node_score)
    
    async def get_active_nodes(self) -> List[Dict[str, Any]]:
        """Get all active nodes."""
        await self._ensure_initialized()
        
        try:
            # Get nodes with active heartbeats
            pattern = f"{self.HEARTBEAT_PREFIX}*"
            cursor = b'0'
            active_nodes = []
            
            while cursor:
                cursor, keys = await self.redis_client.scan(
                    cursor, match=pattern.encode(), count=100
                )
                
                for key in keys:
                    node_id = key.decode().replace(self.HEARTBEAT_PREFIX, "")
                    node_data = await self._get_node_data(node_id)
                    if node_data:
                        # Add metrics
                        metrics = await self._get_node_metrics(node_id)
                        node_data.update(metrics)
                        active_nodes.append(node_data)
                
                if cursor == b'0':
                    break
            
            return active_nodes
            
        except Exception as e:
            logger.error(f"Failed to get active nodes: {e}")
            return []
    
    async def get_node_status(self) -> Dict[str, Any]:
        """Get status of all nodes."""
        # Get ALL registered nodes (not just active)
        all_nodes = await self._get_all_gpu_nodes()
        active_nodes = await self.get_active_nodes()
        
        # Create lookup set for active node IDs
        active_node_ids = {node.get("node_id") for node in active_nodes}
        
        # Count layer coverage from active nodes only
        layer_coverage = set()
        for node in active_nodes:
            layer_coverage.update(node.get("layers_assigned", []))
        
        # Count actual active vs inactive
        active_count = len(active_nodes)
        inactive_count = len([n for n in all_nodes if n.get("node_id") not in active_node_ids])
        
        return {
            "total_nodes": len(all_nodes),
            "active_nodes": active_count,
            "inactive_nodes": inactive_count,
            "layer_coverage": len(layer_coverage),
            "nodes": [
                {
                    "node_id": node.get("node_id"),
                    "status": "active" if node.get("node_id") in active_node_ids else "inactive",
                    "type": node.get("node_type"),
                    "layers": node.get("layers_assigned", []),
                    "load": node.get("current_load", 0),
                    "success_rate": self._calculate_success_rate(node),
                    "last_heartbeat": datetime.fromtimestamp(
                        node.get("last_heartbeat", 0)
                    ).isoformat() if node.get("last_heartbeat") else None
                }
                for node in all_nodes
            ]
        }
    
    async def forward_to_gpu(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Forward inference request to an available GPU node."""
        # Get active GPU nodes
        gpu_nodes = [
            n for n in await self.get_active_nodes()
            if n.get("node_type") == "gpu" and n.get("status") == "active"
        ]
        
        if not gpu_nodes:
            raise RuntimeError("No GPU nodes available for inference")
        
        # Select node with lowest load
        node = min(gpu_nodes, key=lambda n: n.get("current_load", 0))
        
        # Update load
        await self.update_metrics(
            node["node_id"],
            load=min(1.0, node.get("current_load", 0) + 0.1)
        )
        
        try:
            # Forward request
            async with httpx.AsyncClient(timeout=30) as client:
                headers = {"Authorization": f"Bearer {node['api_key']}"}
                
                start_time = time.time()
                
                response = await client.post(
                    f"{node['api_url']}/chat",
                    json={
                        "prompt": prompt,
                        "max_new_tokens": max_tokens,
                        "temperature": temperature
                    },
                    headers=headers
                )
                
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Update metrics
                    await self.update_metrics(
                        node["node_id"],
                        latency=latency,
                        success=True,
                        load=max(0.0, node.get("current_load", 0) - 0.1)
                    )
                    
                    return {
                        "success": True,
                        "response": result.get("response", ""),
                        "node_id": node["node_id"],
                        "latency_ms": latency
                    }
                else:
                    raise ValueError(f"GPU node returned {response.status_code}")
                    
        except Exception as e:
            # Update failure metrics
            await self.update_metrics(
                node["node_id"],
                success=False,
                load=max(0.0, node.get("current_load", 0) - 0.1)
            )
            logger.error(f"Failed to forward to GPU node {node['node_id']}: {e}")
            raise
    
    # Helper methods
    
    async def _get_node_data(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data from Redis."""
        try:
            node_key = f"{self.NODE_PREFIX}{node_id}"
            data = await self.redis_client.get(node_key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get node data for {node_id}: {e}")
            return None
    
    async def _get_node_metrics(self, node_id: str) -> Dict[str, Any]:
        """Get node metrics from Redis."""
        try:
            metrics_key = f"{self.METRICS_PREFIX}{node_id}"
            data = await self.redis_client.get(metrics_key)
            if data:
                return json.loads(data)
            return {
                "current_load": 0.0,
                "total_requests": 0,
                "failed_requests": 0,
                "average_latency": 0.0
            }
        except Exception as e:
            logger.error(f"Failed to get metrics for {node_id}: {e}")
            return {}
    
    async def _get_all_gpu_nodes(self) -> List[Dict[str, Any]]:
        """Get all GPU nodes (including inactive)."""
        try:
            pattern = f"{self.NODE_PREFIX}*"
            cursor = b'0'
            nodes = []
            
            while cursor:
                cursor, keys = await self.redis_client.scan(
                    cursor, match=pattern.encode(), count=100
                )
                
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        nodes.append(json.loads(data))
                
                if cursor == b'0':
                    break
            
            return nodes
        except Exception as e:
            logger.error(f"Failed to get all GPU nodes: {e}")
            return []
    
    def _calculate_success_rate(self, node: Dict[str, Any]) -> float:
        """Calculate success rate for a node."""
        total = node.get("total_requests", 0)
        if total == 0:
            return 1.0
        failed = node.get("failed_requests", 0)
        return 1.0 - (failed / total)
    
    async def _acquire_lock(self, lock_key: str) -> bool:
        """Acquire a distributed lock."""
        try:
            return await self.redis_client.set(
                lock_key,
                "1".encode(),
                nx=True,
                ex=self.LOCK_TTL
            )
        except Exception as e:
            logger.error(f"Failed to acquire lock {lock_key}: {e}")
            return False
    
    async def _release_lock(self, lock_key: str):
        """Release a distributed lock."""
        try:
            await self.redis_client.delete(lock_key)
        except Exception as e:
            logger.error(f"Failed to release lock {lock_key}: {e}")
    
    async def _cleanup_expired_nodes(self):
        """Background task to clean up expired nodes."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Get all nodes
                all_nodes = await self._get_all_gpu_nodes()
                current_time = time.time()
                nodes_to_remove = []
                
                for node in all_nodes:
                    node_id = node.get("node_id")
                    if not node_id:
                        continue
                    
                    last_heartbeat = node.get("last_heartbeat", 0)
                    time_since_heartbeat = current_time - last_heartbeat
                    
                    # Check if heartbeat expired
                    heartbeat_key = f"{self.HEARTBEAT_PREFIX}{node_id}"
                    heartbeat_exists = await self.redis_client.exists(heartbeat_key)
                    
                    if not heartbeat_exists:
                        # Node is inactive
                        if time_since_heartbeat > self.STALE_NODE_TIMEOUT:
                            # Node has been inactive for too long - remove it
                            nodes_to_remove.append((node_id, node, time_since_heartbeat))
                            logger.info(f"🗑️ Scheduling removal of stale node {node_id} "
                                      f"(inactive for {time_since_heartbeat/3600:.1f} hours)")
                        elif node.get("status") != "inactive":
                            # Mark as inactive but keep it
                            node["status"] = "inactive"
                            node_key = f"{self.NODE_PREFIX}{node_id}"
                            await self.redis_client.setex(
                                node_key,
                                self.NODE_DATA_TTL,
                                json.dumps(node).encode()
                            )
                            logger.warning(f"⚠️ Node {node_id} marked as inactive "
                                         f"(no heartbeat for {time_since_heartbeat:.0f}s)")
                
                # Remove stale nodes
                if nodes_to_remove:
                    logger.info(f"🧹 Removing {len(nodes_to_remove)} stale nodes")
                    
                    async with self.redis_client.pipeline() as pipe:
                        for node_id, node_data, inactive_time in nodes_to_remove:
                            # Remove from layer assignments
                            for layer_id in node_data.get("layers_assigned", []):
                                layer_key = f"{self.LAYER_PREFIX}{layer_id}"
                                pipe.srem(layer_key, node_id.encode())
                            
                            # Remove all node data
                            pipe.delete(f"{self.NODE_PREFIX}{node_id}")
                            pipe.delete(f"{self.HEARTBEAT_PREFIX}{node_id}")
                            pipe.delete(f"{self.METRICS_PREFIX}{node_id}")
                            pipe.srem("gpu:active_nodes", node_id.encode())
                            
                            logger.info(f"✅ Removed stale node {node_id} "
                                      f"(inactive for {inactive_time/3600:.1f} hours)")
                        
                        await pipe.execute()
                    
                    # Publish cleanup event
                    await self.redis_client.publish(
                        "gpu:events",
                        json.dumps({
                            "event": "stale_nodes_removed",
                            "count": len(nodes_to_remove),
                            "node_ids": [node_id for node_id, _, _ in nodes_to_remove],
                            "timestamp": current_time
                        }).encode()
                    )
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(5)
    
    async def cleanup_stale_nodes(self, force: bool = False) -> Dict[str, Any]:
        """
        Manually trigger cleanup of stale nodes.
        
        Args:
            force: If True, remove all inactive nodes regardless of timeout
        
        Returns:
            Dict with cleanup results
        """
        await self._ensure_initialized()
        
        try:
            all_nodes = await self._get_all_gpu_nodes()
            current_time = time.time()
            nodes_removed = []
            nodes_marked_inactive = []
            
            async with self.redis_client.pipeline() as pipe:
                for node in all_nodes:
                    node_id = node.get("node_id")
                    if not node_id:
                        continue
                    
                    last_heartbeat = node.get("last_heartbeat", 0)
                    time_since_heartbeat = current_time - last_heartbeat
                    
                    # Check if heartbeat exists
                    heartbeat_key = f"{self.HEARTBEAT_PREFIX}{node_id}"
                    heartbeat_exists = await self.redis_client.exists(heartbeat_key)
                    
                    if not heartbeat_exists:
                        # Node is inactive
                        should_remove = force or (time_since_heartbeat > self.STALE_NODE_TIMEOUT)
                        
                        if should_remove:
                            # Remove the node
                            for layer_id in node.get("layers_assigned", []):
                                layer_key = f"{self.LAYER_PREFIX}{layer_id}"
                                pipe.srem(layer_key, node_id.encode())
                            
                            pipe.delete(f"{self.NODE_PREFIX}{node_id}")
                            pipe.delete(f"{self.HEARTBEAT_PREFIX}{node_id}")
                            pipe.delete(f"{self.METRICS_PREFIX}{node_id}")
                            pipe.srem("gpu:active_nodes", node_id.encode())
                            
                            nodes_removed.append({
                                "node_id": node_id,
                                "inactive_hours": time_since_heartbeat / 3600
                            })
                        else:
                            # Just mark as inactive
                            if node.get("status") != "inactive":
                                node["status"] = "inactive"
                                node_key = f"{self.NODE_PREFIX}{node_id}"
                                pipe.setex(
                                    node_key,
                                    self.NODE_DATA_TTL,
                                    json.dumps(node).encode()
                                )
                                nodes_marked_inactive.append(node_id)
                
                await pipe.execute()
            
            # Log results
            if nodes_removed:
                logger.info(f"🧹 Cleaned up {len(nodes_removed)} stale nodes")
            
            return {
                "success": True,
                "nodes_removed": nodes_removed,
                "nodes_marked_inactive": nodes_marked_inactive,
                "total_removed": len(nodes_removed),
                "total_marked_inactive": len(nodes_marked_inactive),
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup stale nodes: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def close(self):
        """Close Redis connection and cleanup."""
        if self._background_task:
            self._background_task.cancel()
        
        if self.redis_client:
            await self.redis_client.close()
        
        self._initialized = False


# Import the production version
from .gpu_node_manager_production import (
    GPUNodeManagerProduction,
    SelectionPolicy,
    get_gpu_node_manager as get_production_manager
)

# Legacy alias for backward compatibility
GPUNodeManagerRedis = GPUNodeManagerProduction

# Use production manager
async def get_gpu_node_manager() -> GPUNodeManagerProduction:
    """Get or create the singleton GPU node manager instance."""
    return await get_production_manager()