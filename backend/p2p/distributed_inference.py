from __future__ import annotations

import asyncio
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
from aiohttp import web

from backend.model.moe_infer import ExpertUsageTracker
from backend.core.param_index import ParameterIndex
from .expert_group_optimizer import (
    ExpertGroupIndex, 
    DistributedInferenceRouter,
    NodeCapability,
    ExpertGroup
)
from backend.security.inference_integrity import (
    InferenceIntegrityCoordinator,
    SecurityBeacon,
    ActivationBeaconGenerator,
    RollingOutputCommitment,
    format_beacon_for_stream,
    parse_beacon_from_stream
)
from backend.security.security_orchestrator import SecurityOrchestrator, SecurityPolicy


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
    """Enhanced distributed inference coordinator with expert group optimization and security verification."""
    
    def __init__(self, usage_tracker: ExpertUsageTracker, param_index: Optional[ParameterIndex] = None):
        # Legacy registry for backward compatibility
        self.registry = ExpertNodeRegistry()
        self.usage_tracker = usage_tracker
        self.active_requests: Dict[str, InferenceRequest] = {}
        
        # New optimized components
        self.group_index = ExpertGroupIndex()
        self.smart_router = DistributedInferenceRouter(self.group_index, usage_tracker)
        
        # Security verification components
        self.param_index = param_index
        self.integrity_coordinator = None
        self.security_orchestrator = None
        if param_index:
            self.integrity_coordinator = InferenceIntegrityCoordinator(param_index)
            # Initialize security orchestrator with production-grade policies
            security_policy = SecurityPolicy(
                beacon_failure_threshold=2,
                integrity_score_threshold=0.75,
                auto_quarantine_enabled=True,
                max_retry_attempts=3
            )
            self.security_orchestrator = SecurityOrchestrator(
                self.integrity_coordinator, 
                self.group_index, 
                security_policy
            )
    
    def register_expert_group_node(self, node_capability: NodeCapability):
        """Register a node with expert group capabilities."""
        self.group_index.register_node(node_capability)
        
        # Also register in legacy registry for backward compatibility
        legacy_node = ExpertNode(
            node_id=node_capability.node_id,
            host=node_capability.host,
            port=node_capability.port,
            available_experts=list(node_capability.individual_experts),
            load_factor=node_capability.load_factor,
            last_heartbeat=node_capability.last_heartbeat
        )
        self.registry.register_node(legacy_node)
    
    async def distribute_inference_optimized(
        self, 
        prompt: str, 
        required_experts: List[str],
        max_new_tokens: int = 64,
        preferred_region: str = "default"
    ) -> Tuple[str, Dict[str, Any]]:
        """Optimized distributed inference using expert groups."""
        
        # Use smart router to find optimal node
        selected_node, matching_group, routing_info = self.smart_router.route_inference_request(
            prompt, required_experts, preferred_region
        )
        
        if not selected_node:
            return f"Error: {routing_info.get('error', 'No suitable nodes found')}", routing_info
        
        # Generate request ID
        request_id = hashlib.sha256(f"{prompt}{time.time()}".encode()).hexdigest()[:16]
        
        try:
            start_time = time.time()
            
            # Call the selected node for complete inference
            if matching_group:
                # Use expert group endpoint for optimized inference
                result = await self._call_expert_group_inference(
                    selected_node, matching_group, prompt, max_new_tokens
                )
            else:
                # Fallback to individual expert calls
                result = await self._call_individual_experts(
                    selected_node, required_experts, prompt, max_new_tokens
                )
            
            processing_time = time.time() - start_time
            
            # Record usage for all experts
            for expert_name in required_experts:
                self.usage_tracker.record_usage(
                    expert_name=expert_name,
                    response_time=processing_time / len(required_experts),
                    quality_score=0.8  # Mock quality score
                )
            
            # Update routing info with results
            routing_info.update({
                "processing_time": processing_time,
                "success": True,
                "experts_used": required_experts,
                "optimization_applied": matching_group is not None
            })
            
            return result, routing_info
            
        except Exception as e:
            routing_info.update({
                "error": str(e),
                "success": False
            })
            return f"Error during inference: {str(e)}", routing_info
    
    async def distribute_inference_secure(
        self, 
        prompt: str, 
        required_experts: List[str],
        max_new_tokens: int = 64,
        preferred_region: str = "default",
        enable_integrity_check: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """Secure distributed inference with real-time integrity verification."""
        
        if not self.integrity_coordinator and enable_integrity_check:
            return "Error: Integrity verification not available", {"error": "No integrity coordinator"}
        
        # Generate request ID
        request_id = hashlib.sha256(f"{prompt}{time.time()}".encode()).hexdigest()[:16]
        
        try:
            # Initialize security audit
            audit_context = None
            if enable_integrity_check:
                audit_context = self.integrity_coordinator.initialize_audit(
                    request_id=request_id,
                    prompt=prompt,
                    required_experts=required_experts
                )
            
            # Use smart router to find optimal node
            selected_node, matching_group, routing_info = self.smart_router.route_inference_request(
                prompt, required_experts, preferred_region
            )
            
            if not selected_node:
                return f"Error: {routing_info.get('error', 'No suitable nodes found')}", routing_info
            
            start_time = time.time()
            
            # Call node with security verification
            if enable_integrity_check:
                result, security_report = await self._call_node_with_verification(
                    selected_node, audit_context, prompt, max_new_tokens
                )
            else:
                # Fallback to standard call
                if matching_group:
                    result = await self._call_expert_group_inference(
                        selected_node, matching_group, prompt, max_new_tokens
                    )
                else:
                    result = await self._call_individual_experts(
                        selected_node, required_experts, prompt, max_new_tokens
                    )
                security_report = {"verification_enabled": False}
            
            processing_time = time.time() - start_time
            
            # Record usage for all experts
            for expert_name in required_experts:
                self.usage_tracker.record_usage(
                    expert_name=expert_name,
                    response_time=processing_time / len(required_experts),
                    quality_score=0.8
                )
            
            # Update routing info with security results
            routing_info.update({
                "processing_time": processing_time,
                "success": True,
                "experts_used": required_experts,
                "optimization_applied": matching_group is not None,
                "security_verification": security_report
            })
            
            # Cleanup audit context
            if audit_context and self.integrity_coordinator:
                self.integrity_coordinator.cleanup_audit(request_id)
            
            return result, routing_info
            
        except Exception as e:
            # Cleanup on error
            if audit_context and self.integrity_coordinator:
                self.integrity_coordinator.cleanup_audit(request_id)
            
            routing_info.update({
                "error": str(e),
                "success": False
            })
            return f"Error during secure inference: {str(e)}", routing_info
    
    async def distribute_inference_with_failover(
        self, 
        prompt: str, 
        required_experts: List[str],
        max_new_tokens: int = 64,
        preferred_region: str = "default"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Secure distributed inference with automatic failover and user-friendly messaging.
        """
        if not self.security_orchestrator:
            return await self.distribute_inference_optimized(
                prompt, required_experts, max_new_tokens, preferred_region
            )
        
        try:
            # Use security orchestrator for automatic failover
            result, routing_info = await self.security_orchestrator.secure_inference_with_failover(
                prompt=prompt,
                required_experts=required_experts,
                max_new_tokens=max_new_tokens,
                preferred_region=preferred_region
            )
            
            # Add user-friendly messaging for failover scenarios
            if routing_info.get("failover_occurred", False):
                attempts = routing_info.get("attempts", 1)
                user_message = (
                    f"✅ Successfully completed your request after trying {attempts} nodes. "
                    f"We automatically switched to a more secure node to ensure the best results."
                )
                
                routing_info["user_message"] = user_message
                routing_info["failover_explanation"] = (
                    "Some nodes had security verification issues, but we found a secure alternative."
                )
            
            # Handle case where all nodes failed
            if "error" in routing_info and "All nodes failed" in routing_info.get("error", ""):
                quarantined_count = len(routing_info.get("quarantined_nodes", []))
                user_message = (
                    f"⏳ We're temporarily having issues with our inference nodes. "
                    f"Please wait a moment while we restore service. "
                    f"({quarantined_count} nodes are being verified for security)"
                )
                
                routing_info["user_message"] = user_message
                routing_info["recovery_suggestion"] = "Please try again in 30 seconds"
                routing_info["status"] = "temporary_unavailable"
            
            return result, routing_info
            
        except Exception as e:
            return f"Error: {str(e)}", {"error": str(e), "user_message": "⚠️ Service temporarily unavailable. Please try again shortly."}
    
    async def _call_node_with_verification(
        self, 
        node: NodeCapability, 
        audit_context, 
        prompt: str, 
        max_new_tokens: int
    ) -> Tuple[str, Dict[str, Any]]:
        """Call node with full security verification."""
        
        # Generate header beacon
        header_beacon = self.integrity_coordinator.generate_header_beacon(audit_context)
        
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "security_audit": {
                "request_id": audit_context.request_id,
                "audit_nonce": audit_context.audit_nonce,
                "required_experts": audit_context.required_experts,
                "merkle_root": audit_context.merkle_root,
                "activation_checkpoints": audit_context.activation_checkpoints,
                "routing_seed": audit_context.routing_seed
            },
            "verification_enabled": True
        }
        
        received_beacons = [header_beacon]  # Start with our header beacon
        rolling_commitment = RollingOutputCommitment(audit_context.request_id)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{node.endpoint}/inference/secure",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60.0)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Secure inference failed: HTTP {response.status}: {error_text}")
                
                # Process streaming response with beacon verification
                full_response = ""
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    if line_str.startswith("BEACON:"):
                        # Parse and verify beacon
                        beacon = parse_beacon_from_stream(line_str)
                        if beacon:
                            received_beacons.append(beacon)
                    
                    elif line_str.startswith("TOKEN:"):
                        # Process output token
                        token = line_str[6:]  # Remove "TOKEN:" prefix
                        full_response += token
                        rolling_commitment.update(token)
                    
                    elif line_str.startswith("RESULT:"):
                        # Final result
                        final_result = line_str[7:]  # Remove "RESULT:" prefix
                        full_response = final_result
        
        # Analyze security beacons
        security_analysis = self.integrity_coordinator.analyze_beacon_stream(
            audit_context.request_id, received_beacons
        )
        
        security_report = {
            "verification_enabled": True,
            "beacon_count": len(received_beacons),
            "integrity_score": security_analysis.get("integrity_score", 0.0),
            "trust_level": security_analysis.get("trust_level", "UNKNOWN"),
            "anomalies": security_analysis.get("anomalies", []),
            "verified_components": security_analysis.get("verified_components", []),
            "rolling_hash": rolling_commitment.rolling_hash
        }
        
        return full_response, security_report
    
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
    
    async def _call_expert_group_inference(
        self, 
        node: NodeCapability, 
        expert_group: ExpertGroup, 
        prompt: str, 
        max_new_tokens: int
    ) -> str:
        """Call optimized expert group inference on a single node."""
        payload = {
            "group_id": expert_group.group_id,
            "experts": list(expert_group.experts),
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "optimization": "expert_group"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{node.endpoint}/inference/expert_group",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30.0)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("text", f"Expert group {expert_group.group_id} inference completed")
                else:
                    error_text = await response.text()
                    raise Exception(f"Expert group inference failed: HTTP {response.status}: {error_text}")
    
    async def _call_individual_experts(
        self, 
        node: NodeCapability, 
        required_experts: List[str], 
        prompt: str, 
        max_new_tokens: int
    ) -> str:
        """Fallback to individual expert calls on a single node."""
        payload = {
            "experts": required_experts,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "optimization": "individual_fallback"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{node.endpoint}/inference/multi_expert",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30.0)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("text", f"Multi-expert inference completed using {len(required_experts)} experts")
                else:
                    error_text = await response.text()
                    raise Exception(f"Multi-expert inference failed: HTTP {response.status}: {error_text}")
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights about the optimization performance."""
        return {
            "router_stats": self.smart_router.get_optimization_stats(),
            "replication_suggestions": self.smart_router.get_replication_suggestions(),
            "registered_nodes": len(self.group_index.nodes),
            "hot_expert_groups": [
                {
                    "group_id": group.group_id,
                    "experts": list(group.experts),
                    "usage_count": group.usage_count,
                    "co_occurrence_score": group.co_occurrence_score
                }
                for group in self.group_index.get_hot_expert_groups(limit=5)
            ]
        }


class ExpertNodeServer:
    """Enhanced server that runs on expert nodes with expert group support."""
    
    def __init__(self, node_id: str, available_experts: List[str], expert_groups: List[ExpertGroup] = None, port: int = 8001):
        self.node_id = node_id
        self.available_experts = available_experts
        self.expert_groups = expert_groups or []
        self.port = port
        self.app = web.Application()
        self.current_load = 0.0
        
        # Setup routes
        self.app.router.add_post('/expert/inference', self.handle_inference)
        self.app.router.add_post('/inference/expert_group', self.handle_expert_group_inference)
        self.app.router.add_post('/inference/multi_expert', self.handle_multi_expert_inference)
        self.app.router.add_post('/inference/secure', self.handle_secure_inference)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/experts', self.handle_list_experts)
        self.app.router.add_get('/expert_groups', self.handle_list_expert_groups)
    
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
    
    async def handle_expert_group_inference(self, request: web.Request) -> web.Response:
        """Handle optimized expert group inference."""
        try:
            data = await request.json()
            group_id = data.get('group_id')
            experts = data.get('experts', [])
            prompt = data.get('prompt', '')
            max_new_tokens = data.get('max_new_tokens', 64)
            
            # Verify this node has the expert group
            matching_group = None
            for group in self.expert_groups:
                if group.group_id == group_id or set(experts).issubset(group.experts):
                    matching_group = group
                    break
            
            if not matching_group:
                return web.json_response(
                    {"error": f"Expert group {group_id} not available on this node"},
                    status=404
                )
            
            # Simulate optimized group inference
            start_time = time.time()
            result_text = f"Expert group {group_id} inference: '{prompt[:50]}...' using {len(experts)} experts efficiently"
            processing_time = time.time() - start_time
            
            return web.json_response({
                "group_id": group_id,
                "experts_used": experts,
                "text": result_text,
                "processing_time": processing_time,
                "node_id": self.node_id,
                "optimization": "expert_group"
            })
            
        except Exception as e:
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def handle_multi_expert_inference(self, request: web.Request) -> web.Response:
        """Handle multi-expert inference fallback."""
        try:
            data = await request.json()
            experts = data.get('experts', [])
            prompt = data.get('prompt', '')
            max_new_tokens = data.get('max_new_tokens', 64)
            
            # Verify this node has all required experts
            missing_experts = [exp for exp in experts if exp not in self.available_experts]
            if missing_experts:
                return web.json_response(
                    {"error": f"Missing experts: {missing_experts}"},
                    status=404
                )
            
            # Simulate multi-expert inference
            start_time = time.time()
            result_text = f"Multi-expert inference: '{prompt[:50]}...' using {len(experts)} individual experts"
            processing_time = time.time() - start_time
            
            return web.json_response({
                "experts_used": experts,
                "text": result_text,
                "processing_time": processing_time,
                "node_id": self.node_id,
                "optimization": "individual_fallback"
            })
            
        except Exception as e:
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def handle_list_experts(self, request: web.Request) -> web.Response:
        """List available experts on this node."""
        return web.json_response({
            "node_id": self.node_id,
            "experts": self.available_experts
        })
    
    async def handle_list_expert_groups(self, request: web.Request) -> web.Response:
        """List available expert groups on this node."""
        return web.json_response({
            "node_id": self.node_id,
            "expert_groups": [
                {
                    "group_id": group.group_id,
                    "experts": list(group.experts),
                    "usage_count": group.usage_count,
                    "co_occurrence_score": group.co_occurrence_score
                }
                for group in self.expert_groups
            ]
        })
    
    async def handle_secure_inference(self, request: web.Request) -> web.Response:
        """Handle secure inference with real-time integrity verification."""
        try:
            data = await request.json()
            prompt = data.get('prompt', '')
            max_new_tokens = data.get('max_new_tokens', 64)
            security_audit = data.get('security_audit', {})
            verification_enabled = data.get('verification_enabled', True)
            
            if not verification_enabled:
                return web.json_response(
                    {"error": "Secure endpoint requires verification enabled"},
                    status=400
                )
            
            # Extract audit parameters
            request_id = security_audit.get('request_id', '')
            audit_nonce = security_audit.get('audit_nonce', '')
            required_experts = security_audit.get('required_experts', [])
            merkle_root = security_audit.get('merkle_root', '')
            activation_checkpoints = security_audit.get('activation_checkpoints', [])
            routing_seed = security_audit.get('routing_seed', 0)
            
            # Verify we have required experts
            missing_experts = [exp for exp in required_experts if exp not in self.available_experts]
            if missing_experts:
                return web.json_response(
                    {"error": f"Missing experts: {missing_experts}"},
                    status=404
                )
            
            # Create streaming response
            response = web.StreamResponse()
            response.headers['Content-Type'] = 'text/plain'
            await response.prepare(request)
            
            # Initialize security components
            audit_context = InferenceAuditContext(
                request_id=request_id,
                audit_nonce=audit_nonce,
                required_experts=required_experts,
                merkle_root=merkle_root,
                routing_seed=routing_seed,
                image_digest="sha256:mock_node_digest",
                activation_checkpoints=activation_checkpoints
            )
            
            beacon_generator = ActivationBeaconGenerator(audit_context)
            rolling_commitment = RollingOutputCommitment(request_id)
            
            # Send header beacon
            header_beacon = SecurityBeacon(
                beacon_type="header",
                timestamp=time.time(),
                request_id=request_id,
                beacon_data={
                    "node_id": self.node_id,
                    "image_digest": "sha256:mock_node_digest",
                    "merkle_root": merkle_root,
                    "experts_loaded": required_experts,
                    "verification_ready": True
                }
            )
            
            await response.write(format_beacon_for_stream(header_beacon).encode())
            
            # Simulate expert inference with periodic beacons
            start_time = time.time()
            
            # Simulate loading experts and generate weight proof beacon
            await asyncio.sleep(0.1)  # Simulate loading time
            
            weight_proof_beacon = SecurityBeacon(
                beacon_type="weight_proof",
                timestamp=time.time(),
                request_id=request_id,
                beacon_data={
                    "experts_verified": required_experts,
                    "merkle_proof_valid": True,
                    "proof_pages": [0, 1, 2],  # Mock page indices
                    "verification_hash": hashlib.sha256(f"{merkle_root}{audit_nonce}".encode()).hexdigest()[:16]
                }
            )
            
            await response.write(format_beacon_for_stream(weight_proof_beacon).encode())
            
            # Simulate inference with activation beacons
            mock_tokens = ["The", " answer", " to", " your", " question", " is", " 42", "."]
            
            for i, token in enumerate(mock_tokens):
                # Simulate processing time
                await asyncio.sleep(0.05)
                
                # Generate activation beacon at checkpoints
                if i in [2, 5]:  # Checkpoints at tokens 2 and 5
                    layer_idx = activation_checkpoints[min(i//3, len(activation_checkpoints)-1)]
                    
                    # Mock activation tensor (in real implementation, this would be actual layer output)
                    mock_activation = torch.randn(1, 10, 768)  # [batch, seq, hidden]
                    
                    activation_beacon = beacon_generator.generate_beacon(layer_idx, mock_activation)
                    if activation_beacon:
                        await response.write(format_beacon_for_stream(activation_beacon).encode())
                
                # Send token with rolling commitment
                rolling_hash = rolling_commitment.update(token)
                await response.write(f"TOKEN:{token}\n".encode())
                
                # Send rolling commitment beacon every few tokens
                if i % 3 == 0:
                    rolling_beacon = rolling_commitment.generate_beacon()
                    await response.write(format_beacon_for_stream(rolling_beacon).encode())
            
            # Send footer beacon
            processing_time = time.time() - start_time
            footer_beacon = SecurityBeacon(
                beacon_type="footer",
                timestamp=time.time(),
                request_id=request_id,
                beacon_data={
                    "used_experts": required_experts,
                    "processing_time": processing_time,
                    "final_rolling_hash": rolling_commitment.rolling_hash,
                    "token_count": rolling_commitment.token_count,
                    "node_signature": "mock_signature_" + self.node_id
                }
            )
            
            await response.write(format_beacon_for_stream(footer_beacon).encode())
            
            # Send final result
            result_text = "".join(mock_tokens)
            await response.write(f"RESULT:{result_text}\n".encode())
            
            return response
            
        except Exception as e:
            return web.json_response(
                {"error": f"Secure inference failed: {str(e)}"},
                status=500
            )
    
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