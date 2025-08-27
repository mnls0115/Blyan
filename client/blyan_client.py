#!/usr/bin/env python3
"""
Blyan AI Blockchain Client SDK
Programmatic interface for node registration and distributed inference.
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BlyanNode:
    """Represents a Blyan expert node."""
    node_id: str
    host: str
    port: int
    available_experts: List[str]
    expert_groups: Optional[List[Dict]] = None
    region: Optional[str] = None


class BlyanClient:
    """
    Client for interacting with Blyan AI Blockchain.
    
    Example:
        client = BlyanClient(os.getenv('BLYAN_API_URL', 'http://165.227.221.225:8000'))
        
        # Register as expert node
        node = BlyanNode("my-node", "192.168.1.10", 8001, ["layer0.expert0"])
        await client.register_node(node)
        
        # Run inference
        result = await client.chat("What is AI?", use_moe=True)
        print(result)
    """
    
    def __init__(self, api_url: str = None, api_key: Optional[str] = None):
        """
        Initialize Blyan client.
        
        Args:
            api_url: Base URL of Blyan API server
            api_key: Optional API key for authentication
        """
        self.api_url = (api_url or os.getenv('BLYAN_API_URL', 'http://165.227.221.225:8000')).rstrip("/")
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            
        self.session = aiohttp.ClientSession(headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to API."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with BlyanClient() as client:'")
            
        url = f"{self.api_url}{endpoint}"
        async with self.session.request(method, url, **kwargs) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise Exception(f"API error {response.status}: {error_text}")
            return await response.json()
    
    # === Node Registration ===
    
    async def register_node(self, node: BlyanNode) -> Dict[str, Any]:
        """
        Register an expert node for distributed inference.
        
        Args:
            node: BlyanNode configuration
            
        Returns:
            Registration response with node details
        """
        payload = {
            "node_id": node.node_id,
            "host": node.host,
            "port": node.port,
            "available_experts": node.available_experts
        }
        
        # Use optimized registration if expert groups provided
        if node.expert_groups:
            payload["expert_groups"] = node.expert_groups
            payload["region"] = node.region
            endpoint = "/p2p/register_optimized"
        else:
            endpoint = "/p2p/register"
            
        logger.info(f"Registering node {node.node_id} at {node.host}:{node.port}")
        return await self._request("POST", endpoint, json=payload)
    
    async def unregister_node(self, node_id: str) -> Dict[str, Any]:
        """Unregister an expert node."""
        logger.info(f"Unregistering node {node_id}")
        return await self._request("DELETE", f"/p2p/nodes/{node_id}")
    
    async def list_nodes(self) -> List[Dict[str, Any]]:
        """List all registered nodes."""
        result = await self._request("GET", "/p2p/nodes")
        return result.get("nodes", [])
    
    async def send_heartbeat(self, node_id: str, load_factor: float = 0.0) -> Dict[str, Any]:
        """Send heartbeat for a node with optional device metrics."""
        try:
            import torch
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
        except Exception:
            vram_gb = None
        payload_params = {"load_factor": load_factor}
        query = f"/p2p/heartbeat/{node_id}?load_factor={load_factor}"
        # Optionally attach metrics via query for simplicity; could be JSON body too
        if vram_gb is not None:
            query += f"&vram_gb={vram_gb:.2f}"
        return await self._request("POST", query)
    
    # === Inference ===
    
    async def chat(
        self, 
        prompt: str, 
        max_new_tokens: int = 256,
        use_moe: bool = False,
        use_distributed: bool = False,
        use_secure: bool = False,
        top_k_experts: int = 2,
        required_experts: Optional[List[str]] = None
    ) -> str:
        """
        Run inference using Blyan AI.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            use_moe: Use Mixture of Experts
            use_distributed: Use distributed inference
            use_secure: Use secure distributed inference with integrity verification
            top_k_experts: Number of experts to use (for MoE)
            required_experts: Specific experts to use (for distributed)
            
        Returns:
            Generated text response
        """
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens
        }
        
        # Choose endpoint based on options
        if use_secure and use_distributed:
            endpoint = "/chat/distributed_secure"
            payload["required_experts"] = required_experts or []
            payload["enable_integrity_check"] = True
        elif use_distributed:
            endpoint = "/chat/distributed"
            payload["top_k_experts"] = top_k_experts
        else:
            endpoint = "/chat"
            payload["use_moe"] = use_moe
            payload["top_k_experts"] = top_k_experts
            
        logger.info(f"Running inference: {endpoint}")
        result = await self._request("POST", endpoint, json=payload)
        return result.get("response", "")
    
    # === Expert Management ===
    
    async def get_expert_stats(self, expert_name: str) -> Dict[str, Any]:
        """Get statistics for a specific expert."""
        return await self._request("GET", f"/experts/stats/{expert_name}")
    
    async def get_top_experts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing experts."""
        result = await self._request("GET", f"/experts/top?limit={limit}")
        return result.get("experts", [])
    
    # === Chain Operations ===
    
    async def get_chain_blocks(self, chain_id: str = "A") -> List[Dict[str, Any]]:
        """Get blocks from a specific chain."""
        return await self._request("GET", f"/chain/{chain_id}/blocks")
    
    async def get_latest_block(self, chain_id: str = "A") -> Dict[str, Any]:
        """Get latest block from a chain."""
        return await self._request("GET", f"/chain/{chain_id}/latest")
    
    # === P2P Insights ===
    
    async def get_optimization_insights(self) -> Dict[str, Any]:
        """Get P2P optimization insights."""
        return await self._request("GET", "/p2p/optimization_insights")
    
    async def get_expert_groups(self) -> Dict[str, Any]:
        """Get information about expert groups."""
        return await self._request("GET", "/p2p/expert_groups")


class NodeRunner:
    """
    Helper class to run an expert node with automatic heartbeat.
    
    Example:
        node = BlyanNode("gpu-node-1", "192.168.1.10", 8001, 
                        ["layer0.expert0", "layer1.expert1"])
        
        runner = NodeRunner(node, api_url=os.getenv('BLYAN_API_URL', 'http://165.227.221.225:8000'))
        await runner.run()  # Runs until interrupted
    """
    
    def __init__(
        self, 
        node: BlyanNode, 
        api_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        heartbeat_interval: int = 30
    ):
        self.node = node
        self.client = BlyanClient(api_url, api_key)
        self.heartbeat_interval = heartbeat_interval
        self.running = False
        
    async def run(self):
        """Run node with automatic heartbeat until interrupted."""
        async with self.client:
            # Register node
            try:
                await self.client.register_node(self.node)
                logger.info(f"Node {self.node.node_id} registered successfully")
            except Exception as e:
                logger.error(f"Failed to register node: {e}")
                return
            
            # Run heartbeat loop
            self.running = True
            try:
                while self.running:
                    # Send heartbeat
                    try:
                        # Calculate load factor (implement your own logic)
                        load_factor = self._calculate_load()
                        await self.client.send_heartbeat(self.node.node_id, load_factor)
                        logger.debug(f"Heartbeat sent, load: {load_factor:.2f}")
                    except Exception as e:
                        logger.warning(f"Heartbeat failed: {e}")
                    
                    # Wait for next heartbeat
                    await asyncio.sleep(self.heartbeat_interval)
                    
            except KeyboardInterrupt:
                logger.info("Shutting down node...")
            finally:
                # Unregister node
                try:
                    await self.client.unregister_node(self.node.node_id)
                    logger.info(f"Node {self.node.node_id} unregistered")
                except Exception as e:
                    logger.error(f"Failed to unregister node: {e}")
    
    def _calculate_load(self) -> float:
        """
        Calculate current load factor (0.0 to 1.0).
        Override this method to implement custom load calculation.
        """
        # Simple implementation - you can use GPU/CPU metrics
        import random
        return random.uniform(0.1, 0.5)
    
    def stop(self):
        """Stop the node runner."""
        self.running = False


# === Example Usage ===

async def example_basic_usage():
    """Basic usage example."""
    # Connect to Blyan API
    async with BlyanClient("http://localhost:8000") as client:
        # List current nodes
        nodes = await client.list_nodes()
        print(f"Current nodes: {len(nodes)}")
        
        # Run inference
        response = await client.chat(
            "What is artificial intelligence?",
            use_moe=True,
            top_k_experts=2
        )
        print(f"Response: {response}")
        
        # Get expert statistics
        stats = await client.get_expert_stats("layer0.expert0")
        print(f"Expert stats: {stats}")


async def example_node_registration():
    """Node registration example."""
    # Define your node
    node = BlyanNode(
        node_id="my-gpu-node",
        host="192.168.1.100",  # Your node's IP
        port=8001,
        available_experts=["layer0.expert0", "layer0.expert1", "layer1.expert0"],
        expert_groups=[{
            "experts": ["layer0.expert0", "layer0.expert1"],
            "usage_count": 10
        }],
        region="us-west"
    )
    
    # Register and run
    runner = NodeRunner(node, api_url="http://api.blyan.com")
    await runner.run()  # Runs until Ctrl+C


async def example_distributed_inference():
    """Distributed inference example."""
    async with BlyanClient(os.getenv('BLYAN_API_URL', 'http://165.227.221.225:8000')) as client:
        # Run secure distributed inference
        response = await client.chat(
            "Explain quantum computing",
            use_distributed=True,
            use_secure=True,
            required_experts=["layer0.expert0", "layer1.expert1"]
        )
        print(f"Secure response: {response}")
        
        # Get optimization insights
        insights = await client.get_optimization_insights()
        print(f"Network insights: {json.dumps(insights, indent=2)}")


if __name__ == "__main__":
    # Run examples
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "node":
        # Run as node: python blyan_client.py node
        asyncio.run(example_node_registration())
    else:
        # Run basic example
        asyncio.run(example_basic_usage())