"""GPU Node Manager with Redis Backend - Production Implementation V2
====================================================================
This is the actual implementation with all production fixes applied.
"""

import os
import time
import json
import asyncio
import logging
import secrets
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
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
    MIN_LOAD = "min_load"
    BEST_SCORE = "best_score"
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"


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
    current_load: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    average_latency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GPUNodeInfo':
        return cls(**data)
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return 1.0 - (self.failed_requests / self.total_requests)
    
    @property
    def is_healthy(self) -> bool:
        return (time.time() - self.last_heartbeat) < 30


class GPUNodeManagerRedis:
    """Production GPU node manager with all fixes applied."""
    
    # Redis key prefixes
    NODE_PREFIX = "gpu:node:"
    LAYER_PREFIX = "gpu:layer:"
    HEARTBEAT_PREFIX = "gpu:heartbeat:"
    METRICS_PREFIX = "gpu:metrics:"
    LOCK_PREFIX = "gpu:lock:"
    ALL_NODES_SET = "gpu:all_nodes"
    ACTIVE_NODES_SET = "gpu:active_nodes"
    
    # Configuration
    HEARTBEAT_TTL = 30
    NODE_DATA_TTL = 7200
    LOCK_TTL = 10
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    
    # Lua scripts
    RELEASE_LOCK_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """
    
    UPDATE_METRICS_SCRIPT = """
    local key = KEYS[1]
    local load_delta = ARGV[1]
    local incr_req = ARGV[2]
    local is_success = ARGV[3]
    local latency = ARGV[4]
    
    if load_delta ~= "" then
        -- Use HINCRBYFLOAT for atomic load updates
        local new_load = redis.call("hincrbyfloat", key, "current_load", load_delta)
        -- Ensure load stays within bounds [0.0, 1.0]
        if tonumber(new_load) < 0 then
            redis.call("hset", key, "current_load", "0")
        elseif tonumber(new_load) > 1 then
            redis.call("hset", key, "current_load", "1")
        end
    end
    
    if incr_req == "1" then
        redis.call("hincrby", key, "total_requests", 1)
        if is_success == "0" then
            redis.call("hincrby", key, "failed_requests", 1)
        end
    end
    
    if latency ~= "" then
        local old_lat = redis.call("hget", key, "average_latency") or "0"
        local new_lat = tonumber(old_lat) * 0.9 + tonumber(latency) * 0.1
        redis.call("hset", key, "average_latency", tostring(new_lat))
    end
    
    redis.call("expire", key, 7200)
    return "OK"
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        data_dir: Optional[Path] = None,
        stale_timeout_hours: float = 1.0,
        redis_username: Optional[str] = None,
        redis_password: Optional[str] = None,
        redis_ssl: Optional[bool] = None
    ):
        """Initialize with production features.
        
        Args:
            redis_url: Redis connection URL
            data_dir: Data directory (deprecated, kept for compatibility)
            stale_timeout_hours: Hours before node is considered stale
            redis_username: Redis ACL username (for Redis 6+)
            redis_password: Redis password
            redis_ssl: Enable SSL/TLS connection
        """
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis is required for distributed GPU node management")
        
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_username = redis_username or os.getenv("REDIS_USERNAME")
        self.redis_password = redis_password or os.getenv("REDIS_PASSWORD")
        self.redis_ssl = redis_ssl if redis_ssl is not None else os.getenv("REDIS_SSL", "").lower() == "true"
        self.redis_client: Optional[Redis] = None
        self._background_task = None
        self._health_check_task = None
        self._initialized = False
        self._round_robin_counter = 0
        
        # Configuration
        self.stale_timeout = stale_timeout_hours * 3600
        
        # Get selection policy from environment
        policy_str = os.getenv("GPU_NODE_SELECTION_POLICY", "best_score").lower()
        try:
            self.selection_policy = SelectionPolicy(policy_str)
        except ValueError:
            self.selection_policy = SelectionPolicy.BEST_SCORE
        
        # Override stale timeout from environment
        env_timeout = os.getenv("GPU_NODE_STALE_TIMEOUT_HOURS")
        if env_timeout:
            try:
                self.stale_timeout = float(env_timeout) * 3600
            except ValueError:
                pass
        
        # Script SHAs for Lua scripts
        self._release_lock_sha = None
        self._update_metrics_sha = None
        
        logger.info(f"🚀 GPU Node Manager initialized")
        logger.info(f"   Selection policy: {self.selection_policy.value}")
        logger.info(f"   Stale timeout: {self.stale_timeout/3600:.1f} hours")
    
    async def initialize(self):
        """Initialize Redis connection with retry logic."""
        if self._initialized:
            return
        
        for attempt in range(self.MAX_RETRIES):
            try:
                # Build connection kwargs
                connection_kwargs = {
                    "decode_responses": False,
                    "max_connections": 50,
                    "socket_connect_timeout": 5,
                    "socket_timeout": 5,
                    "retry_on_timeout": True,
                    "health_check_interval": 30
                }
                
                # Add authentication if provided
                if self.redis_username:
                    connection_kwargs["username"] = self.redis_username
                if self.redis_password:
                    connection_kwargs["password"] = self.redis_password
                
                # Add SSL if enabled
                if self.redis_ssl:
                    connection_kwargs["ssl"] = True
                    connection_kwargs["ssl_cert_reqs"] = "required"
                    # Optional: specify SSL cert paths
                    ssl_certfile = os.getenv("REDIS_SSL_CERTFILE")
                    ssl_keyfile = os.getenv("REDIS_SSL_KEYFILE")
                    ssl_ca_certs = os.getenv("REDIS_SSL_CA_CERTS")
                    if ssl_certfile:
                        connection_kwargs["ssl_certfile"] = ssl_certfile
                    if ssl_keyfile:
                        connection_kwargs["ssl_keyfile"] = ssl_keyfile
                    if ssl_ca_certs:
                        connection_kwargs["ssl_ca_certs"] = ssl_ca_certs
                
                self.redis_client = Redis.from_url(
                    self.redis_url,
                    **connection_kwargs
                )
                
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
                    raise
    
    async def _ensure_initialized(self):
        """Ensure Redis is initialized with reconnection."""
        if not self._initialized or not self.redis_client:
            await self.initialize()
        
        try:
            await self.redis_client.ping()
        except (ConnectionError, TimeoutError):
            logger.warning("Redis connection lost, reconnecting...")
            self._initialized = False
            await self.initialize()
    
    async def _with_retry(self, operation, *args, **kwargs):
        """Execute operation with retry logic."""
        for attempt in range(self.MAX_RETRIES):
            try:
                await self._ensure_initialized()
                return await operation(*args, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Redis operation failed (attempt {attempt+1}): {e}")
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (2 ** attempt))
                    self._initialized = False
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
        """Register node with safe locking."""
        # Generate unique token for lock
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
            # Validate node
            async with httpx.AsyncClient(timeout=10) as client:
                headers = {"Authorization": f"Bearer {api_key}"}
                response = await client.get(f"{api_url}/health", headers=headers)
                if response.status_code != 200:
                    raise ValueError(f"Node health check failed: {response.status_code}")
            
            # Create node info - default to "gpu" type if not specified or if has GPU capabilities
            actual_node_type = node_type
            if not actual_node_type and capabilities:
                # Auto-detect type from capabilities
                if capabilities.get("gpu_memory_gb", 0) > 0:
                    actual_node_type = "gpu"
            
            # Default to "gpu" if still not set (backwards compatibility)
            if not actual_node_type:
                actual_node_type = "gpu"
            
            node = GPUNodeInfo(
                node_id=node_id,
                api_url=api_url.rstrip('/'),
                api_key=api_key,
                node_type=actual_node_type,
                capabilities=capabilities or {},
                status="active",
                registered_at=time.time(),
                last_heartbeat=time.time()
            )
            
            # Auto-assign layers
            if node_type == "gpu" and capabilities:
                layers = capabilities.get('layers', [])
                if not layers and capabilities.get('gpu_memory_gb', 0) >= 16:
                    all_nodes_count = await self._with_retry(
                        self.redis_client.scard,
                        self.ALL_NODES_SET
                    ) or 0
                    num_nodes = all_nodes_count + 1
                    layers_per_node = max(1, 32 // num_nodes)
                    start_layer = (num_nodes - 1) * layers_per_node
                    layers = list(range(start_layer, min(32, start_layer + layers_per_node)))
                node.layers_assigned = layers
            
            # Store in Redis atomically
            async def store_node():
                async with self.redis_client.pipeline() as pipe:
                    # Node data
                    node_key = f"{self.NODE_PREFIX}{node_id}"
                    pipe.setex(node_key, self.NODE_DATA_TTL, json.dumps(node.to_dict()).encode())
                    
                    # Heartbeat
                    heartbeat_key = f"{self.HEARTBEAT_PREFIX}{node_id}"
                    pipe.setex(heartbeat_key, self.HEARTBEAT_TTL, str(time.time()).encode())
                    
                    # Layer assignments
                    for layer_id in node.layers_assigned:
                        layer_key = f"{self.LAYER_PREFIX}{layer_id}"
                        pipe.sadd(layer_key, node_id.encode())
                        pipe.expire(layer_key, self.NODE_DATA_TTL)
                    
                    # Add to sets
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
            
            logger.info(f"✅ Registered {node_type} node {node_id}")
            
            return {
                "success": True,
                "message": f"Node {node_id} registered successfully",
                "node_id": node_id,
                "layers_assigned": node.layers_assigned,
                "status": node.status
            }
            
        except Exception as e:
            logger.error(f"Failed to register node {node_id}: {e}")
            return {"success": False, "message": str(e)}
        finally:
            # Release lock with token
            await self._with_retry(
                self._release_lock_with_token,
                lock_key,
                lock_token
            )
    
    async def unregister_node(self, node_id: str) -> bool:
        """Unregister a GPU node."""
        await self._ensure_initialized()
        
        try:
            node_data = await self._get_node_data(node_id)
            if not node_data:
                return False
            
            async with self.redis_client.pipeline() as pipe:
                # Remove from sets
                pipe.srem(self.ALL_NODES_SET, node_id.encode())
                pipe.srem(self.ACTIVE_NODES_SET, node_id.encode())
                
                # Remove from layers
                for layer_id in node_data.get("layers_assigned", []):
                    pipe.srem(f"{self.LAYER_PREFIX}{layer_id}", node_id.encode())
                
                # Delete keys
                pipe.delete(f"{self.NODE_PREFIX}{node_id}")
                pipe.delete(f"{self.HEARTBEAT_PREFIX}{node_id}")
                pipe.delete(f"{self.METRICS_PREFIX}{node_id}")
                
                await pipe.execute()
            
            logger.info(f"📤 Unregistered node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister node {node_id}: {e}")
            return False
    
    async def update_heartbeat(self, node_id: str) -> bool:
        """Update heartbeat with set maintenance."""
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
                        pipe.setex(node_key, self.NODE_DATA_TTL, json.dumps(node_info).encode())
                    
                    result = await pipe.execute()
                    return all(result)
            
            return await self._with_retry(update)
            
        except Exception as e:
            logger.error(f"Failed to update heartbeat for {node_id}: {e}")
            return False
    
    async def update_metrics(
        self,
        node_id: str,
        load_delta: Optional[float] = None,
        latency: Optional[float] = None,
        success: Optional[bool] = None
    ):
        """Update metrics atomically using Redis hash and Lua script.
        
        Args:
            node_id: Node identifier
            load_delta: Change in load (e.g., +0.1 or -0.1), not absolute value
            latency: Request latency in ms
            success: Whether the request succeeded
        """
        try:
            metrics_key = f"{self.METRICS_PREFIX}{node_id}"
            
            # Prepare arguments
            args = [
                str(load_delta).encode() if load_delta is not None else b"",
                b"1" if success is not None else b"0",
                b"0" if success is False else b"1",
                str(latency).encode() if latency is not None else b""
            ]
            
            await self._with_retry(
                self.redis_client.evalsha,
                self._update_metrics_sha,
                1,
                metrics_key.encode(),
                *args
            )
            
        except Exception as e:
            logger.error(f"Failed to update metrics for {node_id}: {e}")
    
    async def get_active_nodes(self) -> List[Dict[str, Any]]:
        """Get active nodes efficiently using sets."""
        try:
            # Get from active set
            active_node_ids = await self._with_retry(
                self.redis_client.smembers,
                self.ACTIVE_NODES_SET
            )
            
            if not active_node_ids:
                return []
            
            active_nodes = []
            
            # Batch fetch
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
        """Get accurate status using sets."""
        try:
            # Get ALL nodes
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
            
            # Get active set
            active_node_ids_set = await self._with_retry(
                self.redis_client.smembers,
                self.ACTIVE_NODES_SET
            )
            active_ids = {nid.decode() for nid in active_node_ids_set}
            
            nodes = []
            layer_coverage = set()
            active_count = 0
            inactive_count = 0
            
            # Batch fetch all nodes
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
                    
                    # Determine status
                    if has_heartbeat and node_id in active_ids:
                        node_info["status"] = "active"
                        active_count += 1
                        layer_coverage.update(node_info.get("layers_assigned", []))
                    else:
                        node_info["status"] = "inactive"
                        inactive_count += 1
                    
                    # Add metrics
                    if metrics:
                        total = int(metrics.get(b"total_requests", b"0"))
                        failed = int(metrics.get(b"failed_requests", b"0"))
                        node_info["success_rate"] = 1.0 - (failed / total) if total > 0 else 1.0
                        node_info["current_load"] = float(metrics.get(b"current_load", b"0"))
                    
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
    
    def _calculate_node_score(self, node: Dict[str, Any]) -> float:
        """Calculate composite score for node selection."""
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
    
    async def get_best_node_for_layer(self, layer_id: int) -> Optional[Dict[str, Any]]:
        """Get best node for a specific layer using selection policy."""
        active_nodes = await self.get_active_nodes()
        
        # Filter nodes with this layer
        nodes_with_layer = [
            n for n in active_nodes
            if layer_id in n.get("layers_assigned", [])
        ]
        
        if not nodes_with_layer:
            return None
        
        # Apply selection policy
        if self.selection_policy == SelectionPolicy.MIN_LOAD:
            return min(nodes_with_layer, key=lambda n: n.get("current_load", 0))
        
        elif self.selection_policy == SelectionPolicy.BEST_SCORE:
            return max(nodes_with_layer, key=self._calculate_node_score)
        
        elif self.selection_policy == SelectionPolicy.ROUND_ROBIN:
            self._round_robin_counter = (self._round_robin_counter + 1) % len(nodes_with_layer)
            return nodes_with_layer[self._round_robin_counter]
        
        else:  # RANDOM
            import random
            return random.choice(nodes_with_layer)
    
    async def forward_to_gpu(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Forward to best GPU node based on selection policy."""
        active_nodes = await self.get_active_nodes()
        
        # Filter for GPU nodes - accept nodes with node_type="gpu" OR no node_type (backwards compat)
        # Also accept nodes that have GPU capabilities regardless of node_type
        gpu_nodes = []
        for node in active_nodes:
            node_type = node.get("node_type", "")
            # Accept if: explicitly GPU, no type specified (legacy), or has GPU capabilities
            if (node_type == "gpu" or 
                node_type == "" or 
                node.get("capabilities", {}).get("gpu_memory_gb", 0) > 0 or
                node.get("layers_assigned", [])):
                gpu_nodes.append(node)
        
        if not gpu_nodes:
            # Log more details for debugging
            logger.error(f"No GPU nodes found. Active nodes: {len(active_nodes)}")
            for node in active_nodes[:3]:  # Log first 3 for debugging
                logger.error(f"  Node {node.get('node_id')}: type={node.get('node_type', 'none')}, layers={node.get('layers_assigned', [])}")
            raise RuntimeError("No GPU nodes available for inference")
        
        # Select based on policy
        if self.selection_policy == SelectionPolicy.MIN_LOAD:
            node = min(gpu_nodes, key=lambda n: n.get("current_load", 0))
        elif self.selection_policy == SelectionPolicy.BEST_SCORE:
            node = max(gpu_nodes, key=self._calculate_node_score)
        elif self.selection_policy == SelectionPolicy.ROUND_ROBIN:
            self._round_robin_counter = (self._round_robin_counter + 1) % len(gpu_nodes)
            node = gpu_nodes[self._round_robin_counter]
        else:  # RANDOM
            import random
            node = random.choice(gpu_nodes)
        
        node_id = node["node_id"]
        
        # Increment load atomically (+0.1)
        await self.update_metrics(node_id, load_delta=0.1)
        
        try:
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
                    
                    # Update metrics - decrement load and record success
                    await self.update_metrics(
                        node_id,
                        latency=latency,
                        success=True,
                        load_delta=-0.1  # Decrement load atomically
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
            # Decrement load on failure
            await self.update_metrics(
                node_id,
                success=False,
                load_delta=-0.1  # Decrement load atomically
            )
            logger.error(f"Failed to forward to GPU node {node_id}: {e}")
            raise
    
    async def _acquire_lock_with_token(self, lock_key: str, token: str) -> bool:
        """Acquire lock with unique token."""
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
    
    async def _cleanup_expired_nodes(self):
        """Background cleanup with set maintenance."""
        while True:
            try:
                await asyncio.sleep(60)
                
                all_node_ids = await self.redis_client.smembers(self.ALL_NODES_SET)
                current_time = time.time()
                
                nodes_to_remove = []
                nodes_to_mark_inactive = []
                
                for node_id_bytes in all_node_ids:
                    node_id = node_id_bytes.decode()
                    
                    heartbeat_key = f"{self.HEARTBEAT_PREFIX}{node_id}"
                    heartbeat_exists = await self.redis_client.exists(heartbeat_key)
                    
                    if not heartbeat_exists:
                        # Remove from active set
                        await self.redis_client.srem(self.ACTIVE_NODES_SET, node_id.encode())
                        
                        # Check if stale
                        node_data = await self.redis_client.get(f"{self.NODE_PREFIX}{node_id}")
                        if node_data:
                            node_info = json.loads(node_data)
                            time_since = current_time - node_info.get("last_heartbeat", 0)
                            
                            if time_since > self.stale_timeout:
                                nodes_to_remove.append((node_id, node_info))
                            elif node_info.get("status") != "inactive":
                                nodes_to_mark_inactive.append((node_id, node_info))
                
                # Remove stale nodes
                if nodes_to_remove:
                    async with self.redis_client.pipeline() as pipe:
                        for node_id, node_info in nodes_to_remove:
                            pipe.srem(self.ALL_NODES_SET, node_id.encode())
                            pipe.srem(self.ACTIVE_NODES_SET, node_id.encode())
                            
                            for layer_id in node_info.get("layers_assigned", []):
                                pipe.srem(f"{self.LAYER_PREFIX}{layer_id}", node_id.encode())
                            
                            pipe.delete(f"{self.NODE_PREFIX}{node_id}")
                            pipe.delete(f"{self.HEARTBEAT_PREFIX}{node_id}")
                            pipe.delete(f"{self.METRICS_PREFIX}{node_id}")
                            
                            logger.info(f"🗑️ Removed stale node {node_id}")
                        
                        await pipe.execute()
                
                # Mark inactive
                if nodes_to_mark_inactive:
                    async with self.redis_client.pipeline() as pipe:
                        for node_id, node_info in nodes_to_mark_inactive:
                            node_info["status"] = "inactive"
                            pipe.setex(
                                f"{self.NODE_PREFIX}{node_id}",
                                self.NODE_DATA_TTL,
                                json.dumps(node_info).encode()
                            )
                        await pipe.execute()
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(5)
    
    async def _health_check_loop(self):
        """Periodic Redis health check."""
        while True:
            try:
                await asyncio.sleep(30)
                await self.redis_client.ping()
            except (ConnectionError, TimeoutError):
                logger.warning("Redis health check failed, triggering reconnection")
                self._initialized = False
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def cleanup_stale_nodes(self, force: bool = False) -> Dict[str, Any]:
        """Manual cleanup of stale nodes."""
        await self._ensure_initialized()
        
        try:
            all_node_ids = await self.redis_client.smembers(self.ALL_NODES_SET)
            current_time = time.time()
            nodes_removed = []
            
            async with self.redis_client.pipeline() as pipe:
                for node_id_bytes in all_node_ids:
                    node_id = node_id_bytes.decode()
                    
                    heartbeat_exists = await self.redis_client.exists(f"{self.HEARTBEAT_PREFIX}{node_id}")
                    
                    if not heartbeat_exists:
                        node_data = await self.redis_client.get(f"{self.NODE_PREFIX}{node_id}")
                        if node_data:
                            node_info = json.loads(node_data)
                            time_since = current_time - node_info.get("last_heartbeat", 0)
                            
                            if force or time_since > self.stale_timeout:
                                pipe.srem(self.ALL_NODES_SET, node_id.encode())
                                pipe.srem(self.ACTIVE_NODES_SET, node_id.encode())
                                
                                for layer_id in node_info.get("layers_assigned", []):
                                    pipe.srem(f"{self.LAYER_PREFIX}{layer_id}", node_id.encode())
                                
                                pipe.delete(f"{self.NODE_PREFIX}{node_id}")
                                pipe.delete(f"{self.HEARTBEAT_PREFIX}{node_id}")
                                pipe.delete(f"{self.METRICS_PREFIX}{node_id}")
                                
                                nodes_removed.append({
                                    "node_id": node_id,
                                    "inactive_hours": time_since / 3600
                                })
                
                await pipe.execute()
            
            return {
                "success": True,
                "nodes_removed": nodes_removed,
                "total_removed": len(nodes_removed)
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup: {e}")
            return {"success": False, "error": str(e)}
    
    async def close(self):
        """Clean shutdown."""
        if self._background_task:
            self._background_task.cancel()
        if self._health_check_task:
            self._health_check_task.cancel()
        if self.redis_client:
            await self.redis_client.close()
        self._initialized = False


# Singleton
_manager_instance: Optional[GPUNodeManagerRedis] = None


async def get_gpu_node_manager() -> GPUNodeManagerRedis:
    """Get or create singleton instance."""
    global _manager_instance
    
    if _manager_instance is None:
        _manager_instance = GPUNodeManagerRedis()
        await _manager_instance.initialize()
    
    return _manager_instance