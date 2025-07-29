from __future__ import annotations

import asyncio
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
from aiohttp import web

from backend.model.moe_infer import ExpertUsageTracker


@dataclass
class ExpertNode:
    """Represents a node that can serve specific experts."""
    node_id: str
    host: str
    port: int
    available_experts: List[str]
    load_factor: float = 0.0  # Current load (0.0 = idle, 1.0 = fully loaded)
    last_heartbeat: float = 0.0
    
    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class InferenceRequest:
    """Request for distributed inference."""
    request_id: str
    prompt: str
    required_experts: List[str]
    max_new_tokens: int = 64
    timeout: float = 30.0


@dataclass
class ExpertResponse:
    """Response from an expert node."""
    expert_name: str
    node_id: str
    result: Any
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class ExpertNodeRegistry:
    """Registry of available expert nodes."""
    
    def __init__(self):
        self.nodes: Dict[str, ExpertNode] = {}
        self.expert_to_nodes: Dict[str, List[str]] = {}  # expert_name -> [node_ids]
    
    def register_node(self, node: ExpertNode):
        """Register a new expert node."""
        self.nodes[node.node_id] = node
        node.last_heartbeat = time.time()
        
        # Update expert mappings
        for expert_name in node.available_experts:
            if expert_name not in self.expert_to_nodes:
                self.expert_to_nodes[expert_name] = []
            if node.node_id not in self.expert_to_nodes[expert_name]:
                self.expert_to_nodes[expert_name].append(node.node_id)
        
        print(f"✓ Registered node {node.node_id} with {len(node.available_experts)} experts")
    
    def unregister_node(self, node_id: str):
        """Remove a node from the registry."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        # Remove from expert mappings
        for expert_name in node.available_experts:
            if expert_name in self.expert_to_nodes:
                self.expert_to_nodes[expert_name] = [
                    nid for nid in self.expert_to_nodes[expert_name] if nid != node_id
                ]
                if not self.expert_to_nodes[expert_name]:
                    del self.expert_to_nodes[expert_name]
        
        del self.nodes[node_id]
        print(f"✗ Unregistered node {node_id}")
    
    def get_nodes_for_expert(self, expert_name: str) -> List[ExpertNode]:
        """Get all nodes that can serve a specific expert."""
        node_ids = self.expert_to_nodes.get(expert_name, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def select_best_node(self, expert_name: str) -> Optional[ExpertNode]:
        """Select the best node for an expert based on load balancing."""
        nodes = self.get_nodes_for_expert(expert_name)
        if not nodes:
            return None
        
        # Filter out stale nodes (no heartbeat in last 60 seconds)
        current_time = time.time()
        active_nodes = [n for n in nodes if current_time - n.last_heartbeat < 60.0]
        
        if not active_nodes:
            return None
        
        # Select node with lowest load
        return min(active_nodes, key=lambda n: n.load_factor)
    
    def update_node_load(self, node_id: str, load_factor: float):
        """Update the load factor for a node."""
        if node_id in self.nodes:
            self.nodes[node_id].load_factor = load_factor
    
    def heartbeat(self, node_id: str):
        """Update heartbeat for a node."""
        if node_id in self.nodes:
            self.nodes[node_id].last_heartbeat = time.time()


class DistributedInferenceCoordinator:
    """Coordinates distributed inference across expert nodes."""
    
    def __init__(self, usage_tracker: ExpertUsageTracker):
        self.registry = ExpertNodeRegistry()
        self.usage_tracker = usage_tracker
        self.active_requests: Dict[str, InferenceRequest] = {}
    
    async def distribute_inference(
        self, 
        prompt: str, 
        required_experts: List[str],
        max_new_tokens: int = 64
    ) -> Tuple[str, Dict[str, Any]]:
        """Distribute inference across multiple expert nodes."""
        
        # Generate request ID
        request_id = hashlib.sha256(f"{prompt}{time.time()}".encode()).hexdigest()[:16]
        
        request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            required_experts=required_experts,
            max_new_tokens=max_new_tokens
        )
        
        self.active_requests[request_id] = request
        
        try:
            # Assign experts to nodes
            expert_assignments = {}
            for expert_name in required_experts:
                node = self.registry.select_best_node(expert_name)
                if node:
                    expert_assignments[expert_name] = node
                else:
                    print(f"Warning: No available node for expert {expert_name}")
            
            if not expert_assignments:
                return "Error: No expert nodes available", {}
            
            # Execute inference on assigned nodes
            tasks = []
            for expert_name, node in expert_assignments.items():
                task = self._call_expert_node(node, expert_name, prompt, max_new_tokens)
                tasks.append(task)
            
            # Wait for all expert responses
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process responses
            successful_responses = []
            expert_usage = {}
            
            for i, response in enumerate(responses):
                if isinstance(response, ExpertResponse) and response.success:
                    successful_responses.append(response)
                    expert_usage[response.expert_name] = {
                        "node_id": response.node_id,
                        "processing_time": response.processing_time
                    }
                    
                    # Record usage
                    self.usage_tracker.record_usage(
                        expert_name=response.expert_name,
                        response_time=response.processing_time,
                        quality_score=0.8  # Mock quality score
                    )
                elif isinstance(response, Exception):
                    print(f"Expert call failed: {response}")
            
            # Combine results (simplified aggregation)
            if successful_responses:
                combined_result = f"Distributed inference using {len(successful_responses)} experts: "
                combined_result += ", ".join([r.expert_name for r in successful_responses])
                return combined_result, expert_usage
            else:
                return "Error: All expert nodes failed", {}
                
        finally:
            # Clean up
            if request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def _call_expert_node(
        self, 
        node: ExpertNode, 
        expert_name: str, 
        prompt: str, 
        max_new_tokens: int
    ) -> ExpertResponse:
        """Call a specific expert on a remote node."""
        start_time = time.time()
        
        try:
            payload = {
                "expert_name": expert_name,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{node.endpoint}/expert/inference",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30.0)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        processing_time = time.time() - start_time
                        
                        return ExpertResponse(
                            expert_name=expert_name,
                            node_id=node.node_id,
                            result=result,
                            processing_time=processing_time,
                            success=True
                        )
                    else:
                        error_text = await response.text()
                        return ExpertResponse(
                            expert_name=expert_name,
                            node_id=node.node_id,
                            result=None,
                            processing_time=time.time() - start_time,
                            success=False,
                            error_message=f"HTTP {response.status}: {error_text}"
                        )
        
        except Exception as e:
            return ExpertResponse(
                expert_name=expert_name,
                node_id=node.node_id,
                result=None,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )


class ExpertNodeServer:
    """Server that runs on expert nodes to serve specific experts."""
    
    def __init__(self, node_id: str, available_experts: List[str], port: int = 8001):
        self.node_id = node_id
        self.available_experts = available_experts
        self.port = port
        self.app = web.Application()
        self.current_load = 0.0
        
        # Setup routes
        self.app.router.add_post('/expert/inference', self.handle_inference)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/experts', self.handle_list_experts)
    
    async def handle_inference(self, request: web.Request) -> web.Response:
        """Handle inference request for a specific expert."""
        try:
            data = await request.json()
            expert_name = data.get('expert_name')
            prompt = data.get('prompt', '')
            max_new_tokens = data.get('max_new_tokens', 64)
            
            if expert_name not in self.available_experts:
                return web.json_response(
                    {"error": f"Expert {expert_name} not available on this node"},
                    status=404
                )
            
            # Simulate expert inference (replace with actual expert loading and inference)
            start_time = time.time()
            
            # Mock inference result
            result = f"Expert {expert_name} processed: '{prompt[:50]}...'" 
            
            processing_time = time.time() - start_time
            
            return web.json_response({
                "expert_name": expert_name,
                "result": result,
                "processing_time": processing_time,
                "node_id": self.node_id
            })
            
        except Exception as e:
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "node_id": self.node_id,
            "current_load": self.current_load,
            "available_experts": self.available_experts,
            "timestamp": time.time()
        })
    
    async def handle_list_experts(self, request: web.Request) -> web.Response:
        """List available experts on this node."""
        return web.json_response({
            "node_id": self.node_id,
            "experts": self.available_experts
        })
    
    def start_server(self):
        """Start the expert node server."""
        print(f"Starting expert node {self.node_id} on port {self.port}")
        print(f"Available experts: {', '.join(self.available_experts)}")
        web.run_app(self.app, host='0.0.0.0', port=self.port)


# Utility functions for testing
async def simulate_distributed_inference():
    """Simulate a distributed inference scenario."""
    
    # Create usage tracker
    usage_tracker = ExpertUsageTracker(Path("./test_usage.json"))
    
    # Create coordinator
    coordinator = DistributedInferenceCoordinator(usage_tracker)
    
    # Register some mock expert nodes
    node1 = ExpertNode(
        node_id="node1",
        host="localhost",
        port=8001,
        available_experts=["layer0.expert0", "layer0.expert1", "layer1.expert0"]
    )
    
    node2 = ExpertNode(
        node_id="node2", 
        host="localhost",
        port=8002,
        available_experts=["layer1.expert1", "layer2.expert0", "layer2.expert1"]
    )
    
    coordinator.registry.register_node(node1)
    coordinator.registry.register_node(node2)
    
    # Simulate inference request
    required_experts = ["layer0.expert0", "layer1.expert1", "layer2.expert0"]
    
    print("Starting distributed inference...")
    result, usage = await coordinator.distribute_inference(
        prompt="What is the meaning of life?",
        required_experts=required_experts
    )
    
    print(f"Result: {result}")
    print(f"Expert usage: {usage}")


if __name__ == "__main__":
    # Example of running an expert node server
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run as expert node server
        node_id = sys.argv[2] if len(sys.argv) > 2 else "test_node"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8001
        
        # Mock experts for testing
        experts = [f"layer{i}.expert{j}" for i in range(2) for j in range(2)]
        
        server = ExpertNodeServer(node_id, experts, port)
        server.start_server()
    
    else:
        # Run distributed inference simulation
        asyncio.run(simulate_distributed_inference())