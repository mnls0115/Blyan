"""Production-Ready GPU Node Manager with Redis Backend
=======================================================
Addresses all production concerns:
- Accurate node counts
- Redis connection resilience
- Safe distributed locking
- Advanced node selection
- Race-free metrics
- Performance optimizations
"""

import os
import time
import json
import asyncio
import logging
import secrets
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import httpx
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis
    from redis.exceptions import ConnectionError, TimeoutError, RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.error("Redis not available - cannot use distributed GPU node manager")


class SelectionPolicy(Enum):
    """Node selection policies."""
    MIN_LOAD = "min_load"  # Select node with lowest load
    BEST_SCORE = "best_score"  # Select node with best combined score
    ROUND_ROBIN = "round_robin"  # Rotate through nodes
    RANDOM = "random"  # Random selection


@dataclass
class GPUNodeInfo:
    """Information about a registered GPU node."""
    node_id: str
    api_url: str
    api_key: str
    node_type: str = "gpu"
    capabilities: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    status: str = "pending"
    layers_assigned: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GPUNodeInfo':
        """Create from dictionary."""
        return cls(**data)


class GPUNodeManagerProduction:
    """Production-ready GPU node manager with Redis backend."""
    
    # Redis key prefixes
    NODE_PREFIX = "gpu:node:"
    LAYER_PREFIX = "gpu:layer:"
    HEARTBEAT_PREFIX = "gpu:heartbeat:"
    METRICS_PREFIX = "gpu:metrics:"
    LOCK_PREFIX = "gpu:lock:"
    ALL_NODES_SET = "gpu:all_nodes"
    ACTIVE_NODES_SET = "gpu:active_nodes"
    
    # Configuration
    HEARTBEAT_TTL = 30  # seconds
    NODE_DATA_TTL = 7200  # 2 hours for node data cache
    LOCK_TTL = 10  # seconds for distributed locks
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    
    # Lua scripts for atomic operations
    RELEASE_LOCK_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """
    
    UPDATE_METRICS_SCRIPT = """
    local key = KEYS[1]
    if ARGV[1] then redis.call("hset", key, "current_load", ARGV[1]) end
    if ARGV[2] then redis.call("hincrby", key, "total_requests", 1) end
    if ARGV[3] == "false" then redis.call("hincrby", key, "failed_requests", 1) end
    if ARGV[4] then 
        local old_lat = redis.call("hget", key, "average_latency") or "0"
        local new_lat = tonumber(old_lat) * 0.9 + tonumber(ARGV[4]) * 0.1
        redis.call("hset", key, "average_latency", tostring(new_lat))
    end
    redis.call("expire", key, 7200)
    return "OK"
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        stale_timeout_hours: float = 1.0,
        selection_policy: SelectionPolicy = SelectionPolicy.BEST_SCORE
    ):
        """
        Initialize GPU node manager with Redis backend.
        
        Args:
            redis_url: Redis connection URL
            stale_timeout_hours: Hours before inactive nodes are removed
            selection_policy: How to select nodes for inference
        """
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis is required for distributed GPU node management")
        
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client: Optional[Redis] = None
        self._background_task = None
        self._health_check_task = None
        self._initialized = False
        self._round_robin_counter = 0
        
        # Configuration
        self.stale_timeout = stale_timeout_hours * 3600
        self.selection_policy = selection_policy
        
        # Override from environment
        env_timeout = os.getenv("GPU_NODE_STALE_TIMEOUT_HOURS")
        if env_timeout:
            try:
                self.stale_timeout = float(env_timeout) * 3600
            except ValueError:
                pass
        
        env_policy = os.getenv("GPU_NODE_SELECTION_POLICY")
        if env_policy:
            try:
                self.selection_policy = SelectionPolicy(env_policy.lower())
            except ValueError:
                pass
        
        # Prepare Lua scripts
        self._release_lock_sha = None
        self._update_metrics_sha = None
        
        logger.info(f"🚀 GPU Node Manager (Production) initialized")
        logger.info(f"   Stale timeout: {self.stale_timeout/3600:.1f} hours")
        logger.info(f"   Selection policy: {self.selection_policy.value}")
    
    async def initialize(self):
        """Initialize Redis connection with retry logic."""
        if self._initialized:
            return
        
        for attempt in range(self.MAX_RETRIES):
            try:
                self.redis_client = Redis.from_url(
                    self.redis_url,
                    decode_responses=False,
                    max_connections=50,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # Test connection
                await self.redis_client.ping()
                
                # Load Lua scripts
                self._release_lock_sha = await self.redis_client.script_load(
                    self.RELEASE_LOCK_SCRIPT
                )
                self._update_metrics_sha = await self.redis_client.script_load(
                    self.UPDATE_METRICS_SCRIPT
                )
                
                self._initialized = True
                
                # Start background tasks
                if not self._background_task:
                    self._background_task = asyncio.create_task(self._cleanup_expired_nodes())
                if not self._health_check_task:
                    self._health_check_task = asyncio.create_task(self._health_check_loop())
                
                logger.info("✅ Redis connection established")
                return
                
            except (ConnectionError, TimeoutError, RedisError) as e:
                logger.warning(f"Redis connection attempt {attempt+1}/{self.MAX_RETRIES} failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (2 ** attempt))
                else:
                    logger.error("Failed to connect to Redis after all retries")
                    raise
    
    async def _ensure_initialized(self):
        """Ensure Redis is initialized, with reconnection if needed."""
        if not self._initialized or not self.redis_client:
            await self.initialize()
        
        # Test connection and reconnect if needed
        try:
            await self.redis_client.ping()
        except (ConnectionError, TimeoutError):
            logger.warning("Redis connection lost, reconnecting...")
            self._initialized = False
            await self.initialize()
    
    async def _with_retry(self, operation, *args, **kwargs):
        """Execute Redis operation with retry logic."""
        for attempt in range(self.MAX_RETRIES):
            try:
                await self._ensure_initialized()
                return await operation(*args, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Redis operation failed (attempt {attempt+1}/{self.MAX_RETRIES}): {e}")
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (2 ** attempt))
                    self._initialized = False  # Force reconnection
                else:
                    raise
    
    async def register_node(
        self,
        node_id: str,
        api_url: str,
        api_key: str,
        capabilities: Optional[Dict[str, Any]] = None,
        node_type: str = "gpu"
    ) -> Dict[str, Any]:
        """Register a new GPU node with safe distributed locking."""
        
        # Generate unique lock token
        lock_token = secrets.token_hex(16)
        lock_key = f"{self.LOCK_PREFIX}register:{node_id}"
        
        # Acquire lock with token
        lock_acquired = await self._with_retry(
            self._acquire_lock_with_token,
            lock_key,
            lock_token
        )
        
        if not lock_acquired:
            return {
                "success": False,
                "message": "Registration already in progress"
            }
        
        try:
            # Validate node accessibility
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
            
            # Auto-assign layers if needed
            if node_type == "gpu" and capabilities:
                layers = capabilities.get('layers', [])
                if not layers and capabilities.get('gpu_memory_gb', 0) >= 16:
                    # Get current nodes for distribution
                    all_nodes = await self._with_retry(
                        self.redis_client.scard,
                        self.ALL_NODES_SET
                    )
                    num_nodes = all_nodes + 1
                    layers_per_node = max(1, 32 // num_nodes)
                    start_layer = (num_nodes - 1) * layers_per_node
                    layers = list(range(start_layer, min(32, start_layer + layers_per_node)))
                
                node.layers_assigned = layers
            
            # Store in Redis atomically
            async def store_node():
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
                    
                    # Add to node sets
                    pipe.sadd(self.ALL_NODES_SET, node_id.encode())
                    pipe.sadd(self.ACTIVE_NODES_SET, node_id.encode())
                    
                    # Initialize metrics hash
                    metrics_key = f"{self.METRICS_PREFIX}{node_id}"
                    pipe.hset(metrics_key, mapping={
                        b"current_load": b"0.0",
                        b"total_requests": b"0",
                        b"failed_requests": b"0",
                        b"average_latency": b"0.0"
                    })
                    pipe.expire(metrics_key, self.NODE_DATA_TTL)
                    
                    await pipe.execute()
            
            await self._with_retry(store_node)
            
            # Publish event
            await self._with_retry(
                self.redis_client.publish,
                "gpu:events",
                json.dumps({
                    "event": "node_registered",
                    "node_id": node_id,
                    "layers": node.layers_assigned,
                    "timestamp": time.time()
                }).encode()
            )
            
            logger.info(f"✅ Registered {node_type} node {node_id} with {len(node.layers_assigned)} layers")
            
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
            # Release lock with token
            await self._with_retry(
                self._release_lock_with_token,
                lock_key,
                lock_token
            )
    
    async def update_heartbeat(self, node_id: str) -> bool:
        """Update node heartbeat and maintain active set."""
        try:
            async def update():
                async with self.redis_client.pipeline() as pipe:
                    # Update heartbeat TTL
                    heartbeat_key = f"{self.HEARTBEAT_PREFIX}{node_id}"
                    pipe.setex(heartbeat_key, self.HEARTBEAT_TTL, str(time.time()).encode())
                    
                    # Ensure in active set
                    pipe.sadd(self.ACTIVE_NODES_SET, node_id.encode())
                    
                    # Update node status
                    node_key = f"{self.NODE_PREFIX}{node_id}"
                    node_data = await self.redis_client.get(node_key)
                    
                    if node_data:
                        node_info = json.loads(node_data)
                        node_info["last_heartbeat"] = time.time()
                        node_info["status"] = "active"
                        pipe.setex(
                            node_key,
                            self.NODE_DATA_TTL,
                            json.dumps(node_info).encode()
                        )
                    
                    result = await pipe.execute()
                    return all(result)
            
            return await self._with_retry(update)
            
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
        """Update node metrics atomically using Redis hash."""
        try:
            metrics_key = f"{self.METRICS_PREFIX}{node_id}"
            
            # Prepare arguments for Lua script
            args = [
                str(load).encode() if load is not None else None,
                b"1" if success is not None else None,
                b"false" if success is False else b"true",
                str(latency).encode() if latency is not None else None
            ]
            
            await self._with_retry(
                self.redis_client.evalsha,
                self._update_metrics_sha,
                1,  # number of keys
                metrics_key.encode(),
                *args
            )
            
        except Exception as e:
            logger.error(f"Failed to update metrics for {node_id}: {e}")
    
    async def get_active_nodes(self) -> List[Dict[str, Any]]:
        """Get all active nodes efficiently using active set."""
        try:
            # Get node IDs from active set
            active_node_ids = await self._with_retry(
                self.redis_client.smembers,
                self.ACTIVE_NODES_SET
            )
            
            if not active_node_ids:
                return []
            
            active_nodes = []
            
            # Batch fetch node data
            async with self.redis_client.pipeline() as pipe:
                for node_id_bytes in active_node_ids:
                    node_id = node_id_bytes.decode()
                    pipe.get(f"{self.NODE_PREFIX}{node_id}")
                    pipe.exists(f"{self.HEARTBEAT_PREFIX}{node_id}")
                    pipe.hgetall(f"{self.METRICS_PREFIX}{node_id}")
                
                results = await pipe.execute()
            
            # Process results in groups of 3
            for i in range(0, len(results), 3):
                node_data = results[i]
                has_heartbeat = results[i+1]
                metrics = results[i+2]
                
                if node_data and has_heartbeat:
                    node_info = json.loads(node_data)
                    
                    # Add metrics
                    if metrics:
                        node_info["current_load"] = float(metrics.get(b"current_load", b"0"))
                        node_info["total_requests"] = int(metrics.get(b"total_requests", b"0"))
                        node_info["failed_requests"] = int(metrics.get(b"failed_requests", b"0"))
                        node_info["average_latency"] = float(metrics.get(b"average_latency", b"0"))
                    
                    active_nodes.append(node_info)
            
            return active_nodes
            
        except Exception as e:
            logger.error(f"Failed to get active nodes: {e}")
            return []
    
    async def get_node_status(self) -> Dict[str, Any]:
        """Get accurate status of all nodes."""
        try:
            # Get ALL nodes from the all_nodes set
            all_node_ids = await self._with_retry(
                self.redis_client.smembers,
                self.ALL_NODES_SET
            )
            
            if not all_node_ids:
                return {
                    "total_nodes": 0,
                    "active_nodes": 0,
                    "inactive_nodes": 0,
                    "layer_coverage": 0,
                    "nodes": []
                }
            
            # Get active nodes set
            active_node_ids_set = await self._with_retry(
                self.redis_client.smembers,
                self.ACTIVE_NODES_SET
            )
            active_ids = {nid.decode() for nid in active_node_ids_set}
            
            nodes = []
            layer_coverage = set()
            active_count = 0
            inactive_count = 0
            
            # Batch fetch all node data
            async with self.redis_client.pipeline() as pipe:
                for node_id_bytes in all_node_ids:
                    node_id = node_id_bytes.decode()
                    pipe.get(f"{self.NODE_PREFIX}{node_id}")
                    pipe.exists(f"{self.HEARTBEAT_PREFIX}{node_id}")
                    pipe.hgetall(f"{self.METRICS_PREFIX}{node_id}")
                
                results = await pipe.execute()
            
            # Process results
            for i in range(0, len(results), 3):
                node_data = results[i]
                has_heartbeat = results[i+1]
                metrics = results[i+2]
                
                if node_data:
                    node_info = json.loads(node_data)
                    node_id = node_info["node_id"]
                    
                    # Determine actual status
                    if has_heartbeat and node_id in active_ids:
                        node_info["status"] = "active"
                        active_count += 1
                        # Count layers for active nodes
                        layer_coverage.update(node_info.get("layers_assigned", []))
                    else:
                        node_info["status"] = "inactive"
                        inactive_count += 1
                    
                    # Add metrics
                    if metrics:
                        total_req = int(metrics.get(b"total_requests", b"0"))
                        failed_req = int(metrics.get(b"failed_requests", b"0"))
                        node_info["success_rate"] = 1.0 - (failed_req / total_req) if total_req > 0 else 1.0
                        node_info["current_load"] = float(metrics.get(b"current_load", b"0"))
                        node_info["average_latency"] = float(metrics.get(b"average_latency", b"0"))
                    
                    nodes.append({
                        "node_id": node_info["node_id"],
                        "status": node_info["status"],
                        "type": node_info.get("node_type", "gpu"),
                        "layers": node_info.get("layers_assigned", []),
                        "load": node_info.get("current_load", 0),
                        "success_rate": node_info.get("success_rate", 1.0),
                        "last_heartbeat": datetime.fromtimestamp(
                            node_info.get("last_heartbeat", 0)
                        ).isoformat() if node_info.get("last_heartbeat") else None
                    })
            
            return {
                "total_nodes": len(all_node_ids),
                "active_nodes": active_count,
                "inactive_nodes": inactive_count,
                "layer_coverage": len(layer_coverage),
                "nodes": sorted(nodes, key=lambda n: (n["status"] != "active", n["node_id"]))
            }
            
        except Exception as e:
            logger.error(f"Failed to get node status: {e}")
            return {
                "total_nodes": 0,
                "active_nodes": 0,
                "inactive_nodes": 0,
                "layer_coverage": 0,
                "nodes": [],
                "error": str(e)
            }
    
    async def get_best_node_for_inference(self) -> Optional[Dict[str, Any]]:
        """Select best node based on configured policy."""
        active_nodes = await self.get_active_nodes()
        
        if not active_nodes:
            return None
        
        if self.selection_policy == SelectionPolicy.MIN_LOAD:
            # Select node with minimum load
            return min(active_nodes, key=lambda n: n.get("current_load", 0))
        
        elif self.selection_policy == SelectionPolicy.BEST_SCORE:
            # Calculate composite score
            def calculate_score(node: Dict[str, Any]) -> float:
                load = node.get("current_load", 0)
                total_req = node.get("total_requests", 0)
                failed_req = node.get("failed_requests", 0)
                latency = node.get("average_latency", 0)
                
                success_rate = 1.0 - (failed_req / total_req) if total_req > 0 else 1.0
                
                # Weighted score (higher is better)
                score = 0.0
                score += (1.0 - load) * 40  # 40% weight on low load
                score += success_rate * 30  # 30% weight on success rate
                score += max(0, 10 - latency/100) * 20  # 20% weight on low latency
                score += min(10, total_req / 100) * 10  # 10% weight on experience
                
                return score
            
            return max(active_nodes, key=calculate_score)
        
        elif self.selection_policy == SelectionPolicy.ROUND_ROBIN:
            # Round-robin selection
            self._round_robin_counter = (self._round_robin_counter + 1) % len(active_nodes)
            return active_nodes[self._round_robin_counter]
        
        else:  # RANDOM
            import random
            return random.choice(active_nodes)
    
    async def forward_to_gpu(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Forward inference request to best available GPU node."""
        # Get best node based on policy
        node = await self.get_best_node_for_inference()
        
        if not node:
            raise RuntimeError("No GPU nodes available for inference")
        
        node_id = node["node_id"]
        
        # Update load
        await self.update_metrics(node_id, load=min(1.0, node.get("current_load", 0) + 0.1))
        
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
                        node_id,
                        latency=latency,
                        success=True,
                        load=max(0.0, node.get("current_load", 0) - 0.1)
                    )
                    
                    return {
                        "success": True,
                        "response": result.get("response", ""),
                        "node_id": node_id,
                        "latency_ms": latency,
                        "selection_policy": self.selection_policy.value
                    }
                else:
                    raise ValueError(f"GPU node returned {response.status_code}")
                    
        except Exception as e:
            # Update failure metrics
            await self.update_metrics(
                node_id,
                success=False,
                load=max(0.0, node.get("current_load", 0) - 0.1)
            )
            logger.error(f"Failed to forward to GPU node {node_id}: {e}")
            raise
    
    async def _cleanup_expired_nodes(self):
        """Background task to clean up expired nodes and maintain sets."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Get all nodes from set
                all_node_ids = await self.redis_client.smembers(self.ALL_NODES_SET)
                current_time = time.time()
                
                nodes_to_remove = []
                nodes_to_mark_inactive = []
                
                for node_id_bytes in all_node_ids:
                    node_id = node_id_bytes.decode()
                    
                    # Check heartbeat
                    heartbeat_key = f"{self.HEARTBEAT_PREFIX}{node_id}"
                    heartbeat_exists = await self.redis_client.exists(heartbeat_key)
                    
                    if not heartbeat_exists:
                        # Remove from active set
                        await self.redis_client.srem(self.ACTIVE_NODES_SET, node_id.encode())
                        
                        # Get node data to check staleness
                        node_data = await self.redis_client.get(f"{self.NODE_PREFIX}{node_id}")
                        if node_data:
                            node_info = json.loads(node_data)
                            time_since_heartbeat = current_time - node_info.get("last_heartbeat", 0)
                            
                            if time_since_heartbeat > self.stale_timeout:
                                nodes_to_remove.append((node_id, node_info, time_since_heartbeat))
                            elif node_info.get("status") != "inactive":
                                nodes_to_mark_inactive.append((node_id, node_info))
                
                # Remove stale nodes
                if nodes_to_remove:
                    async with self.redis_client.pipeline() as pipe:
                        for node_id, node_info, inactive_time in nodes_to_remove:
                            # Remove from all sets and keys
                            pipe.srem(self.ALL_NODES_SET, node_id.encode())
                            pipe.srem(self.ACTIVE_NODES_SET, node_id.encode())
                            
                            for layer_id in node_info.get("layers_assigned", []):
                                pipe.srem(f"{self.LAYER_PREFIX}{layer_id}", node_id.encode())
                            
                            pipe.delete(f"{self.NODE_PREFIX}{node_id}")
                            pipe.delete(f"{self.HEARTBEAT_PREFIX}{node_id}")
                            pipe.delete(f"{self.METRICS_PREFIX}{node_id}")
                            
                            logger.info(f"🗑️ Removed stale node {node_id} "
                                      f"(inactive for {inactive_time/3600:.1f} hours)")
                        
                        await pipe.execute()
                
                # Mark nodes as inactive
                if nodes_to_mark_inactive:
                    async with self.redis_client.pipeline() as pipe:
                        for node_id, node_info in nodes_to_mark_inactive:
                            node_info["status"] = "inactive"
                            pipe.setex(
                                f"{self.NODE_PREFIX}{node_id}",
                                self.NODE_DATA_TTL,
                                json.dumps(node_info).encode()
                            )
                            logger.debug(f"⚠️ Marked {node_id} as inactive")
                        
                        await pipe.execute()
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(5)
    
    async def _health_check_loop(self):
        """Periodic Redis health check to maintain connection."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self.redis_client.ping()
            except (ConnectionError, TimeoutError):
                logger.warning("Redis health check failed, triggering reconnection")
                self._initialized = False
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _acquire_lock_with_token(self, lock_key: str, token: str) -> bool:
        """Acquire distributed lock with unique token."""
        return await self.redis_client.set(
            lock_key.encode(),
            token.encode(),
            nx=True,
            ex=self.LOCK_TTL
        )
    
    async def _release_lock_with_token(self, lock_key: str, token: str) -> bool:
        """Safely release lock only if we own it."""
        try:
            result = await self.redis_client.evalsha(
                self._release_lock_sha,
                1,
                lock_key.encode(),
                token.encode()
            )
            return result == 1
        except Exception as e:
            logger.error(f"Failed to release lock {lock_key}: {e}")
            return False
    
    async def cleanup_stale_nodes(self, force: bool = False) -> Dict[str, Any]:
        """Manually trigger cleanup of stale nodes."""
        await self._ensure_initialized()
        
        try:
            # Get all nodes from set
            all_node_ids = await self.redis_client.smembers(self.ALL_NODES_SET)
            current_time = time.time()
            nodes_removed = []
            nodes_marked_inactive = []
            
            async with self.redis_client.pipeline() as pipe:
                for node_id_bytes in all_node_ids:
                    node_id = node_id_bytes.decode()
                    
                    # Check heartbeat
                    heartbeat_key = f"{self.HEARTBEAT_PREFIX}{node_id}"
                    heartbeat_exists = await self.redis_client.exists(heartbeat_key)
                    
                    if not heartbeat_exists:
                        # Get node data
                        node_data = await self.redis_client.get(f"{self.NODE_PREFIX}{node_id}")
                        if node_data:
                            node_info = json.loads(node_data)
                            time_since_heartbeat = current_time - node_info.get("last_heartbeat", 0)
                            
                            should_remove = force or (time_since_heartbeat > self.stale_timeout)
                            
                            if should_remove:
                                # Remove completely
                                pipe.srem(self.ALL_NODES_SET, node_id.encode())
                                pipe.srem(self.ACTIVE_NODES_SET, node_id.encode())
                                
                                for layer_id in node_info.get("layers_assigned", []):
                                    pipe.srem(f"{self.LAYER_PREFIX}{layer_id}", node_id.encode())
                                
                                pipe.delete(f"{self.NODE_PREFIX}{node_id}")
                                pipe.delete(f"{self.HEARTBEAT_PREFIX}{node_id}")
                                pipe.delete(f"{self.METRICS_PREFIX}{node_id}")
                                
                                nodes_removed.append({
                                    "node_id": node_id,
                                    "inactive_hours": time_since_heartbeat / 3600
                                })
                            else:
                                # Mark as inactive
                                if node_info.get("status") != "inactive":
                                    node_info["status"] = "inactive"
                                    pipe.setex(
                                        f"{self.NODE_PREFIX}{node_id}",
                                        self.NODE_DATA_TTL,
                                        json.dumps(node_info).encode()
                                    )
                                    pipe.srem(self.ACTIVE_NODES_SET, node_id.encode())
                                    nodes_marked_inactive.append(node_id)
                
                await pipe.execute()
            
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
        if self._health_check_task:
            self._health_check_task.cancel()
        if self.redis_client:
            await self.redis_client.close()
        self._initialized = False


# Singleton instance
_manager_instance: Optional[GPUNodeManagerProduction] = None


async def get_gpu_node_manager() -> GPUNodeManagerProduction:
    """Get or create the singleton GPU node manager instance."""
    global _manager_instance
    
    if _manager_instance is None:
        _manager_instance = GPUNodeManagerProduction()
        await _manager_instance.initialize()
    
    return _manager_instance