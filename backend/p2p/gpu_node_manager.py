"""GPU Node Manager - Production Implementation
================================================================
Handles GPU node registration, health monitoring, and routing.
Designed for both service nodes (forwarding) and GPU nodes.
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


class GPUNodeManager:
    """Manages GPU nodes for distributed inference."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("./data")
        self.nodes: Dict[str, GPUNodeInfo] = {}
        self.layer_assignments: Dict[int, List[str]] = {}  # layer_id -> node_ids
        self.pending_registrations: Set[str] = set()
        self.registration_lock = asyncio.Lock()
        self.persistence_file = self.data_dir / "gpu_nodes.json"
        
        # Configuration
        self.heartbeat_interval = 15  # seconds
        self.max_retries = 3
        self.timeout = 10  # seconds
        
        # Load persisted nodes
        self._load_nodes()
        
        # Start background tasks
        self._background_task = None
        
        logger.info(f"🚀 GPU Node Manager initialized with {len(self.nodes)} nodes")
    
    def _load_nodes(self):
        """Load persisted node information."""
        if self.persistence_file.exists():
            try:
                with open(self.persistence_file, 'r') as f:
                    data = json.load(f)
                    for node_id, node_data in data.get('nodes', {}).items():
                        self.nodes[node_id] = GPUNodeInfo.from_dict(node_data)
                    self.layer_assignments = data.get('layer_assignments', {})
                    # Convert layer keys to integers
                    self.layer_assignments = {
                        int(k): v for k, v in self.layer_assignments.items()
                    }
                logger.info(f"📂 Loaded {len(self.nodes)} nodes from persistence")
            except Exception as e:
                logger.warning(f"Failed to load persisted nodes: {e}")
    
    def _save_nodes(self):
        """Persist node information."""
        try:
            self.data_dir.mkdir(exist_ok=True, parents=True)
            data = {
                'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
                'layer_assignments': self.layer_assignments,
                'timestamp': time.time()
            }
            with open(self.persistence_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save nodes: {e}")
    
    async def register_node(
        self,
        node_id: str,
        api_url: str,
        api_key: str,
        capabilities: Optional[Dict[str, Any]] = None,
        node_type: str = "gpu"
    ) -> Dict[str, Any]:
        """Register a new GPU node."""
        async with self.registration_lock:
            # Check if already registering
            if node_id in self.pending_registrations:
                return {
                    "success": False,
                    "message": "Registration already in progress"
                }
            
            # Mark as pending
            self.pending_registrations.add(node_id)
            
            try:
                # Validate node is accessible
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    headers = {"Authorization": f"Bearer {api_key}"}
                    response = await client.get(f"{api_url}/health", headers=headers)
                    
                    if response.status_code != 200:
                        raise ValueError(f"Node health check failed: {response.status_code}")
                
                # Create or update node info
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
                    
                    # Auto-assign layers based on GPU memory if not specified
                    if not layers and gpu_memory >= 16:  # 16GB+ can handle layers
                        # Distribute 32 layers across available GPUs
                        # This is simplified - production would use smarter assignment
                        num_existing_nodes = len([n for n in self.nodes.values() if n.node_type == "gpu"])
                        layers_per_node = 32 // max(1, num_existing_nodes + 1)
                        start_layer = num_existing_nodes * layers_per_node
                        layers = list(range(start_layer, min(32, start_layer + layers_per_node)))
                    
                    node.layers_assigned = layers
                    
                    # Update layer assignments
                    for layer_id in layers:
                        if layer_id not in self.layer_assignments:
                            self.layer_assignments[layer_id] = []
                        if node_id not in self.layer_assignments[layer_id]:
                            self.layer_assignments[layer_id].append(node_id)
                
                # Store node
                self.nodes[node_id] = node
                self._save_nodes()
                
                logger.info(
                    f"✅ Registered {node_type} node {node_id} "
                    f"with {len(node.layers_assigned)} layers"
                )
                
                # Start health monitoring if not already running
                if not self._background_task:
                    self._background_task = asyncio.create_task(self._monitor_health())
                
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
                self.pending_registrations.discard(node_id)
    
    async def unregister_node(self, node_id: str) -> bool:
        """Unregister a GPU node."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            # Remove from layer assignments
            for layer_id in node.layers_assigned:
                if layer_id in self.layer_assignments:
                    self.layer_assignments[layer_id] = [
                        nid for nid in self.layer_assignments[layer_id] if nid != node_id
                    ]
            
            del self.nodes[node_id]
            self._save_nodes()
            
            logger.info(f"📤 Unregistered node {node_id}")
            return True
        return False
    
    async def update_heartbeat(self, node_id: str) -> bool:
        """Update node heartbeat timestamp."""
        if node_id in self.nodes:
            self.nodes[node_id].last_heartbeat = time.time()
            self.nodes[node_id].status = "active"
            return True
        return False
    
    async def _monitor_health(self):
        """Monitor health of registered nodes."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                for node_id, node in list(self.nodes.items()):
                    if not node.is_healthy:
                        if node.status == "active":
                            logger.warning(f"⚠️ Node {node_id} is unhealthy")
                            node.status = "inactive"
                        
                        # Try to ping the node
                        try:
                            async with httpx.AsyncClient(timeout=5) as client:
                                headers = {"Authorization": f"Bearer {node.api_key}"}
                                response = await client.get(
                                    f"{node.api_url}/health",
                                    headers=headers
                                )
                                if response.status_code == 200:
                                    node.last_heartbeat = time.time()
                                    node.status = "active"
                                    logger.info(f"✅ Node {node_id} recovered")
                        except:
                            pass
                
                # Save state periodically
                self._save_nodes()
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)
    
    def get_nodes_for_layer(self, layer_id: int) -> List[GPUNodeInfo]:
        """Get all nodes that can handle a specific layer."""
        node_ids = self.layer_assignments.get(layer_id, [])
        return [
            self.nodes[node_id]
            for node_id in node_ids
            if node_id in self.nodes and self.nodes[node_id].is_healthy
        ]
    
    def get_best_node_for_layer(self, layer_id: int) -> Optional[GPUNodeInfo]:
        """Get the best node for a specific layer based on load and performance."""
        nodes = self.get_nodes_for_layer(layer_id)
        if not nodes:
            return None
        
        # Sort by: health, load, success rate, latency
        def node_score(node: GPUNodeInfo) -> float:
            score = 0.0
            if node.status == "active":
                score += 10
            score += (1.0 - node.current_load) * 5  # Lower load is better
            score += node.success_rate * 3  # Higher success rate is better
            score -= node.average_latency / 1000  # Lower latency is better
            return score
        
        return max(nodes, key=node_score)
    
    def get_active_nodes(self) -> List[GPUNodeInfo]:
        """Get all active nodes."""
        return [
            node for node in self.nodes.values()
            if node.status == "active" and node.is_healthy
        ]
    
    def get_node_status(self) -> Dict[str, Any]:
        """Get status of all nodes."""
        active_nodes = self.get_active_nodes()
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": len(active_nodes),
            "inactive_nodes": len(self.nodes) - len(active_nodes),
            "layer_coverage": len(self.layer_assignments),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "status": node.status,
                    "type": node.node_type,
                    "layers": node.layers_assigned,
                    "load": node.current_load,
                    "success_rate": node.success_rate,
                    "last_heartbeat": datetime.fromtimestamp(node.last_heartbeat).isoformat()
                }
                for node in self.nodes.values()
            ]
        }
    
    async def forward_to_gpu(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Forward inference request to an available GPU node."""
        # Get an active GPU node
        gpu_nodes = [n for n in self.get_active_nodes() if n.node_type == "gpu"]
        
        if not gpu_nodes:
            raise RuntimeError("No GPU nodes available for inference")
        
        # Select node with lowest load
        node = min(gpu_nodes, key=lambda n: n.current_load)
        
        # Update node stats
        node.current_load = min(1.0, node.current_load + 0.1)
        node.total_requests += 1
        
        try:
            # Forward request
            async with httpx.AsyncClient(timeout=30) as client:
                headers = {"Authorization": f"Bearer {node.api_key}"}
                
                start_time = time.time()
                
                response = await client.post(
                    f"{node.api_url}/chat",
                    json={
                        "prompt": prompt,
                        # GPU node expects 'max_new_tokens'; keep backward compat with max_tokens fallback handled by node default
                        "max_new_tokens": max_tokens,
                        "temperature": temperature
                    },
                    headers=headers
                )
                
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Update node metrics
                    node.average_latency = (
                        node.average_latency * 0.9 + latency * 0.1
                    )  # Exponential moving average
                    
                    return {
                        "success": True,
                        "response": result.get("response", ""),
                        "node_id": node.node_id,
                        "latency_ms": latency
                    }
                else:
                    raise ValueError(f"GPU node returned {response.status_code}")
                    
        except Exception as e:
            node.failed_requests += 1
            logger.error(f"Failed to forward to GPU node {node.node_id}: {e}")
            raise
        finally:
            # Decrease load
            node.current_load = max(0.0, node.current_load - 0.1)
            self._save_nodes()
