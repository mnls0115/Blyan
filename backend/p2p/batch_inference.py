"""
GPU Node Request Forwarding
Handles forwarding inference requests from service node to GPU nodes
"""

import asyncio
import httpx
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class GPUNodeForwarder:
    """
    Forwards inference requests from service node to GPU nodes.
    """
    
    def __init__(self, coordinator=None):
        self.coordinator = coordinator
        self.node_health: Dict[str, float] = {}
        self.request_timeout = 30.0
        logger.info("🚀 GPU Node Forwarder initialized")
    
    def get_available_nodes(self) -> List[Dict[str, Any]]:
        """Get list of available GPU nodes from registry."""
        nodes = []
        
        if self.coordinator and hasattr(self.coordinator, 'registry'):
            for node_id, node in self.coordinator.registry.nodes.items():
                # Check if node is healthy
                if hasattr(node, 'last_heartbeat'):
                    if time.time() - node.last_heartbeat > 60:
                        continue
                
                # Build endpoint URL
                if hasattr(node, 'endpoint'):
                    endpoint = node.endpoint
                elif hasattr(node, 'host'):
                    host = node.host
                    if host.startswith('http'):
                        endpoint = host
                    else:
                        port = getattr(node, 'port', 8000)
                        endpoint = f"http://{host}:{port}"
                else:
                    continue
                
                nodes.append({
                    'node_id': node_id,
                    'endpoint': endpoint,
                    'vram_gb': getattr(node, 'vram_gb', 0),
                    'load': getattr(node, 'current_load', 0.5),
                    'health_score': self.node_health.get(node_id, 1.0)
                })
        
        return nodes
    
    def select_best_node(self, nodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best available node based on load and health."""
        if not nodes:
            return None
        
        # Score nodes (lower load + higher health = better)
        scored = []
        for node in nodes:
            score = (1.0 - node['load']) * node['health_score']
            scored.append((score, node))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
    
    async def forward_to_node(
        self,
        node: Dict[str, Any],
        prompt: str,
        max_new_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Forward request to a specific GPU node."""
        endpoint = node['endpoint'].rstrip('/')
        chat_url = f"{endpoint}/chat"
        
        logger.info(f"📡 Forwarding to GPU node {node['node_id']} at {chat_url}")
        
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                response = await client.post(
                    chat_url,
                    json={
                        "prompt": prompt,
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self._update_health(node['node_id'], True)
                    logger.info(f"✅ Received response from {node['node_id']}")
                    return {
                        "success": True,
                        "response": result.get("response", ""),
                        "inference_time": result.get("inference_time", 0),
                        "node_id": node['node_id']
                    }
                else:
                    self._update_health(node['node_id'], False)
                    return {"success": False, "error": f"Status {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Error forwarding to {node['node_id']}: {e}")
            self._update_health(node['node_id'], False)
            return {"success": False, "error": str(e)}
    
    def _update_health(self, node_id: str, success: bool):
        """Update node health score."""
        current = self.node_health.get(node_id, 1.0)
        if success:
            self.node_health[node_id] = min(1.0, current + 0.1)
        else:
            self.node_health[node_id] = max(0.0, current - 0.2)
    
    async def process_inference(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Process inference by forwarding to best GPU node."""
        nodes = self.get_available_nodes()
        
        if not nodes:
            logger.error("No GPU nodes available")
            return {
                "success": False,
                "error": "No GPU nodes available",
                "response": "Service temporarily unavailable"
            }
        
        logger.info(f"Found {len(nodes)} GPU nodes")
        
        # Try up to 3 nodes
        for attempt in range(min(3, len(nodes))):
            node = self.select_best_node(nodes)
            if not node:
                break
            
            logger.info(f"Attempt {attempt+1}: Using {node['node_id']}")
            result = await self.forward_to_node(node, prompt, max_new_tokens, temperature)
            
            if result.get("success"):
                return {
                    "response": result["response"],
                    "inference_time": result["inference_time"],
                    "node_used": result["node_id"]
                }
            
            # Remove failed node from list
            nodes = [n for n in nodes if n['node_id'] != node['node_id']]
        
        return {
            "success": False,
            "error": "All GPU nodes failed",
            "response": "Unable to process request"
        }


# Global instance
_forwarder = None

def get_gpu_forwarder(coordinator=None):
    """Get or create GPU forwarder."""
    global _forwarder
    if _forwarder is None:
        _forwarder = GPUNodeForwarder(coordinator)
    elif coordinator and not _forwarder.coordinator:
        _forwarder.coordinator = coordinator
    return _forwarder