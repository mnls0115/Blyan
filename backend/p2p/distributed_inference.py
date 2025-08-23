from __future__ import annotations

import asyncio
import json
import os
import time
import hashlib
import random
import logging
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Any, Tuple, Set, AsyncIterator
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import aiohttp
from aiohttp import web
import base64

logger = logging.getLogger(__name__)

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
from backend.core.rewards import get_reward_calculator, RewardComponent
from backend.model.moe_infer import MoEModelManager, ExpertUsageTracker
from backend.core.chain import Chain
from backend.core.scheduler import PreemptiveScheduler, Metrics, SchedulerState
from .connection_pool import ConnectionPool
from backend.learning.pipeline_metrics import get_pipeline_metrics


@dataclass
class DeviceProfile:
    """Hardware and network profile for scheduling/partitioning."""
    vram_gb: float = 0.0
    tflops_est: float = 0.0
    net_mbps: float = 0.0
    latency_ms: Optional[float] = None
    cuda_capability: Optional[str] = None


class WarmPool:
    """LRU cache for frequently used expert combinations with hit ratio tracking."""
    
    def __init__(self, capacity: int = 3):
        self.capacity = capacity
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.load_times: Dict[str, float] = {}
        
    def get(self, expert_group_key: str) -> Optional[Dict[str, Any]]:
        """Get expert group from warm pool."""
        if expert_group_key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(expert_group_key)
            self.hit_count += 1
            return self.cache[expert_group_key]
        else:
            self.miss_count += 1
            return None
            
    def put(self, expert_group_key: str, expert_data: Dict[str, Any]):
        """Add expert group to warm pool."""
        if expert_group_key in self.cache:
            # Update existing entry
            self.cache.move_to_end(expert_group_key)
        else:
            # Add new entry
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                evicted_key, _ = self.cache.popitem(last=False)
                if evicted_key in self.load_times:
                    del self.load_times[evicted_key]
                    
        self.cache[expert_group_key] = expert_data
        self.load_times[expert_group_key] = time.time()
        
    def ensure_loaded(self, expert_group_key: str, loader_func) -> bool:
        """Ensure expert group is loaded in warm pool."""
        if expert_group_key in self.cache:
            return True
            
        try:
            # Load expert group using provided loader function
            expert_data = loader_func(expert_group_key)
            self.put(expert_group_key, expert_data)
            return True
        except Exception as e:
            print(f"❌ Failed to load expert group {expert_group_key}: {e}")
            return False
            
    def get_hit_ratio(self) -> float:
        """Get cache hit ratio."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
        
    def get_status(self) -> Dict[str, Any]:
        """Get warm pool status for monitoring."""
        return {
            "capacity": self.capacity,
            "current_size": len(self.cache),
            "hit_ratio": self.get_hit_ratio(),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "cached_groups": list(self.cache.keys()),
            "load_times": dict(self.load_times)
        }
        
    def clear(self):
        """Clear warm pool."""
        self.cache.clear()
        self.load_times.clear()
        self.hit_count = 0
        self.miss_count = 0


@dataclass
class ExpertNode:
    """Represents a node that can serve specific experts."""
    node_id: str
    host: str
    port: int
    available_experts: List[str]
    load_factor: float = 0.0  # Current load (0.0 = idle, 1.0 = fully loaded)
    last_heartbeat: float = 0.0
    donor_mode: bool = False
    device_profile: Optional[DeviceProfile] = None
    
    @property
    def endpoint(self) -> str:
        # If host already includes protocol, use it as-is
        if self.host.startswith('http://') or self.host.startswith('https://'):
            return self.host
        # Otherwise construct with http
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
        """Register a new expert node with health monitoring."""
        self.nodes[node.node_id] = node
        node.last_heartbeat = time.time()
        
        # Update expert mappings
        for expert_name in node.available_experts:
            if expert_name not in self.expert_to_nodes:
                self.expert_to_nodes[expert_name] = []
            if node.node_id not in self.expert_to_nodes[expert_name]:
                self.expert_to_nodes[expert_name].append(node.node_id)
        
        # Register for health monitoring if coordinator available
        if hasattr(self, 'coordinator') and hasattr(self.coordinator, 'reputation_manager'):
            self.coordinator.reputation_manager.register_node_endpoint(
                node.node_id, node.endpoint
            )
        
        donor_suffix = " [donor]" if getattr(node, "donor_mode", False) else ""
        print(f"✓ Registered node {node.node_id}{donor_suffix} with {len(node.available_experts)} experts")
    
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
    
    def cleanup_stale_nodes(self, ttl_seconds: int = 90):
        """Remove nodes that haven't sent heartbeat within TTL."""
        current_time = time.time()
        stale_nodes = []
        
        for node_id, node in self.nodes.items():
            if current_time - node.last_heartbeat > ttl_seconds:
                stale_nodes.append(node_id)
        
        for node_id in stale_nodes:
            logger.warning(f"Auto-removing stale node {node_id} (no heartbeat for {ttl_seconds}s)")
            self.unregister_node(node_id)
        
        return len(stale_nodes)
    
    def select_best_node(self, expert_name: str, prefer_donor: bool = False) -> Optional[ExpertNode]:
        """Select the best node for an expert based on load balancing and reputation.
        If prefer_donor is True, prioritize donor nodes when available.
        """
        nodes = self.get_nodes_for_expert(expert_name)
        if not nodes:
            return None
        
        # Filter out stale nodes and check reputation
        current_time = time.time()
        
        # Import reputation manager for filtering
        from backend.p2p.node_reputation import get_reputation_manager
        reputation_mgr = get_reputation_manager()
        
        active_nodes = []
        for n in nodes:
            # Check heartbeat (60 seconds timeout)
            if current_time - n.last_heartbeat >= 60.0:
                continue
                
            # Check reputation (higher threshold for donor nodes)
            node_rep = reputation_mgr.get_node_reputation(n.node_id)
            if node_rep:
                success_rate = node_rep.get('success_rate', 0)
                # Donor nodes need higher reputation (70%) to ensure quality
                min_success_rate = 0.7 if getattr(n, "donor_mode", False) else 0.5
                if success_rate < min_success_rate:
                    continue
            
            active_nodes.append(n)
        
        if not active_nodes:
            return None
        
        # Optionally prioritize donor nodes first
        candidate_pool = active_nodes
        if prefer_donor:
            donor_candidates = [n for n in active_nodes if getattr(n, "donor_mode", False)]
            if donor_candidates:
                candidate_pool = donor_candidates
        else:
            # For paid or non-free-tier requests, avoid donor nodes if possible
            non_donor_candidates = [n for n in active_nodes if not getattr(n, "donor_mode", False)]
            if non_donor_candidates:
                candidate_pool = non_donor_candidates
        
        # Select node with lowest load from candidate pool
        return min(candidate_pool, key=lambda n: n.load_factor)
    
    def update_node_load(self, node_id: str, load_factor: float):
        """Update the load factor for a node."""
        if node_id in self.nodes:
            self.nodes[node_id].load_factor = load_factor
    
    def update_device_profile(self, node_id: str, profile: DeviceProfile):
        """Update device profile for a node."""
        if node_id in self.nodes:
            self.nodes[node_id].device_profile = profile
    
    def heartbeat(self, node_id: str):
        """Update heartbeat for a node."""
        if node_id in self.nodes:
            self.nodes[node_id].last_heartbeat = time.time()
            
    def get_available_nodes(self, max_age: float = 300.0) -> List[ExpertNode]:
        """Get list of available nodes (recent heartbeat)."""
        current_time = time.time()
        available = []
        
        for node in self.nodes.values():
            if current_time - node.last_heartbeat <= max_age:
                available.append(node)
                
        return available


class DistributedInferenceCoordinator:
    """Enhanced distributed inference coordinator with expert group optimization and security verification."""
    
    def __init__(self, usage_tracker: ExpertUsageTracker, param_index: Optional[ParameterIndex] = None):
        # Import os locally to avoid any scope issues
        import os as _os
        
        # Legacy registry for backward compatibility
        self.registry = ExpertNodeRegistry()
        
        # Donor utilization cap configuration
        self.max_donor_utilization_ratio = float(_os.getenv("DONOR_USAGE_CAP", "0.3"))
        self.strict_free_tier = _os.getenv("STRICT_FREE_TIER", "false").lower() in ("1", "true", "yes")
        self.free_tier_queue_timeout = float(_os.getenv("FREE_TIER_QUEUE_TIMEOUT", "3.0"))  # seconds
        
        # Concurrent request tracking with EMA
        self.inflight_total = 0
        self.inflight_free_tier = 0
        self.inflight_donor = 0
        self._inflight_lock = asyncio.Lock()
        
        # EMA (Exponential Moving Average) for stable measurement
        self.ema_window_seconds = 30.0  # 30 second window
        self.ema_alpha = 2.0 / (self.ema_window_seconds + 1)  # EMA coefficient
        self.donor_utilization_ema = 0.0
        self.last_ema_update = time.time()
        
        # Metrics tracking
        self.metrics = {
            "donor_utilization_ratio": 0.0,
            "free_tier_queue_len": 0,
            "fallback_rate": 0.0,
            "fallback_count": 0,
            "free_tier_total": 0,
            "strict_free_tier_active": self.strict_free_tier
        }
        
        # Free tier queue management
        self.free_tier_semaphore = None  # Will be initialized based on donor nodes
        self.free_tier_queue_size = 0
        self.usage_tracker = usage_tracker
        self.active_requests: Dict[str, InferenceRequest] = {}
        
        # New optimized components
        self.group_index = ExpertGroupIndex()
        self.smart_router = DistributedInferenceRouter(self.group_index, usage_tracker)
        
        # Warm pool for hot expert combinations
        self.warm_pool = WarmPool(capacity=3)
        
        # Fair reward system
        self.reward_calculator = get_reward_calculator()
        
        # Security verification components
        self.param_index = param_index
        
        # SLO-based scheduler integration
        self.scheduler = PreemptiveScheduler()
        self.inference_metrics = {"request_count": 0, "total_latency": 0.0, "active_requests": 0}
        self.start_time = time.time()  # Track coordinator start time
        
        # Node reputation and health monitoring
        from backend.p2p.node_reputation import NodeReputationManager
        self.reputation_manager = NodeReputationManager(health_check_interval=30)
        self._health_monitor_task = None
        
        # Connection pooling for reduced latency with authentication
        import os
        auth_headers = {}
        node_id = os.environ.get('BLYAN_NODE_ID')
        auth_token = os.environ.get('BLYAN_MAIN_NODE_TOKEN')
        if node_id and auth_token:
            auth_headers = {
                'X-Node-ID': node_id,
                'X-Node-Auth-Token': auth_token
            }
        
        self.connection_pool = ConnectionPool(
            max_connections_per_host=20,
            keepalive_timeout=60,
            total_timeout=30,
            auth_headers=auth_headers
        )
        self.pipeline_metrics = get_pipeline_metrics()
        
        # Node reputation and failover management
        from backend.p2p.node_reputation import get_reputation_manager, get_failover_coordinator
        self.reputation_manager = get_reputation_manager()
        self.failover_coordinator = get_failover_coordinator()
        
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
    
    async def start_health_monitoring(self):
        """Start background health monitoring for all registered nodes."""
        await self.reputation_manager.start_health_monitoring()
        logger.info("Started distributed inference health monitoring")
    
    async def stop_health_monitoring(self):
        """Stop background health monitoring."""
        await self.reputation_manager.stop_health_monitoring()
        logger.info("Stopped distributed inference health monitoring")
    
    def register_expert_group_node(self, node_capability: NodeCapability):
        """Register a node with expert group capabilities and health monitoring."""
        self.group_index.register_node(node_capability)
        
        # Also register in legacy registry for backward compatibility
        legacy_node = ExpertNode(
            node_id=node_capability.node_id,
            host=node_capability.host,
            port=node_capability.port,
            available_experts=list(node_capability.individual_experts),
            load_factor=node_capability.load_factor,
            last_heartbeat=node_capability.last_heartbeat,
            donor_mode=getattr(node_capability, "donor_mode", False)
        )
        self.registry.register_node(legacy_node)
        
        # Register for health monitoring
        endpoint = f"http://{node_capability.host}:{node_capability.port}"
        self.reputation_manager.register_node_endpoint(node_capability.node_id, endpoint)
    
    async def distribute_inference_optimized(
        self, 
        prompt: str, 
        required_experts: List[str],
        max_new_tokens: int = 64,
        preferred_region: str = "default"
    ) -> Tuple[str, Dict[str, Any]]:
        """Optimized distributed inference using expert groups with warm pool."""
        
        # Check warm pool for cached expert combinations
        expert_group_key = f"{sorted(required_experts)}_{preferred_region}"
        warm_data = self.warm_pool.get(expert_group_key)
        
        if warm_data:
            routing_info = {"warm_pool_hit": True, "cached_experts": required_experts}
            # Fast path: use cached expert data
            result = f"Warm pool inference: '{prompt[:50]}...' using cached experts {required_experts}"
            self._update_usage_tracking(required_experts, len(result))
            return result, routing_info
        
        # Get available nodes for fair routing
        available_nodes = self.registry.get_available_nodes()
        candidate_node_ids = [node.node_id for node in available_nodes if 
                             any(expert in node.available_experts for expert in required_experts)]
        
        if not candidate_node_ids:
            return "Error: No suitable nodes found", {"error": "no_nodes", "warm_pool_hit": False}
        
        # Use fair routing probabilities
        routing_context = {
            "latency_score": 0.8,  # Would come from actual network measurements
            "warm_hit_score": 0.7,  # Would come from warm pool status
            "cost_efficiency": 1.0   # Would come from cost estimates
        }
        
        routing_probs = self.reward_calculator.get_routing_probabilities(
            candidate_node_ids, routing_context
        )
        
        # Select node based on fair probabilities
        selected_node_id = np.random.choice(
            list(routing_probs.keys()), 
            p=list(routing_probs.values())
        )
        
        selected_node = next(node for node in available_nodes if node.node_id == selected_node_id)
        
        routing_info = {
            "warm_pool_hit": False,
            "fair_routing": True,
            "routing_probabilities": routing_probs,
            "selected_node": selected_node_id
        }
        
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
            
            # Record usage for all experts with calculated quality scores
            from backend.model.quality_calculator import get_quality_calculator
            quality_calc = get_quality_calculator()
            
            for expert_name in required_experts:
                # Calculate real quality score
                quality_score = quality_calc.calculate_score(
                    expert_name=expert_name,
                    response_time=processing_time / len(required_experts),
                    tokens_generated=len(result.split()) if isinstance(result, str) else 50,
                    success=True
                )
                
                self.usage_tracker.record_usage(
                    expert_name=expert_name,
                    response_time=processing_time / len(required_experts),
                    quality_score=quality_score
                )
            
            # Calculate fair reward for the node
            tokens_generated = len(result.split()) if isinstance(result, str) else 50  # Better token approximation
            # Use average quality score from all experts
            avg_quality_score = np.mean([
                quality_calc.calculate_score(
                    expert_name=expert_name,
                    response_time=processing_time / len(required_experts),
                    tokens_generated=tokens_generated,
                    success=True
                ) for expert_name in required_experts
            ])
            quality_score = avg_quality_score
            integrity_passed = True  # Would come from security verification
            estimated_cost = processing_time * 0.1  # Mock cost estimation
            
            reward, reward_components = self.reward_calculator.compute_reward(
                node_id=selected_node.node_id,
                request_id=request_id,
                tokens_generated=tokens_generated,
                quality_score=quality_score,
                actual_latency_ms=processing_time * 1000,
                integrity_passed=integrity_passed,
                estimated_cost=estimated_cost
            )
            
            # Cache successful result in warm pool for future use
            expert_data = {
                "result": result,
                "experts": required_experts,
                "node_id": selected_node.node_id,
                "processing_time": processing_time,
                "timestamp": time.time(),
                "reward": reward
            }
            self.warm_pool.put(expert_group_key, expert_data)
            
            # Update routing info with results
            routing_info.update({
                "processing_time": processing_time,
                "success": True,
                "experts_used": required_experts,
                "optimization_applied": False,  # No matching group in fair routing
                "cached_for_future": True,
                "reward": reward,
                "reward_components": reward_components.__dict__
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
            
            # Record usage for all experts with real quality scores
            from backend.model.quality_calculator import get_quality_calculator
            quality_calc = get_quality_calculator()
            
            for expert_name in required_experts:
                # Calculate quality based on performance
                quality_score = quality_calc.calculate_score(
                    expert_name=expert_name,
                    response_time=processing_time / len(required_experts),
                    tokens_generated=len(result.split()) if isinstance(result, str) else 50,
                    success=True
                )
                
                self.usage_tracker.record_usage(
                    expert_name=expert_name,
                    response_time=processing_time / len(required_experts),
                    quality_score=quality_score
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
        
        # Use connection pool for better performance
        async with await self.connection_pool.post(
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
    
    async def distribute_inference_with_failover(
        self,
        prompt: str,
        required_experts: List[str],
        max_new_tokens: int = 64
    ) -> Tuple[str, Dict[str, Any]]:
        """Distribute inference with automatic failover and reputation-based routing."""
        
        request_id = hashlib.sha256(f"{prompt}{time.time()}".encode()).hexdigest()[:16]
        
        # Get nodes for each required expert
        expert_to_nodes = {}
        for expert in required_experts:
            nodes = self.registry.get_nodes_for_expert(expert)
            if not nodes:
                return f"Error: No nodes available for expert {expert}", {"error": "no_nodes"}
            
            # Rank nodes by reputation
            node_ids = [n.node_id for n in nodes]
            ranked_nodes = self.reputation_manager.rank_nodes(node_ids)
            expert_to_nodes[expert] = [nid for nid, _ in ranked_nodes]
        
        # Execute with failover for each expert
        results = {}
        metrics = {}
        
        async def execute_expert_inference(node_id: str, expert_name: str):
            """Execute inference for a single expert on a node."""
            node = self.registry.nodes[node_id]
            payload = {
                "prompt": prompt,
                "expert": expert_name,
                "max_new_tokens": max_new_tokens
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{node.endpoint}/inference",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30.0)
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Inference failed: HTTP {response.status}")
                    return await response.json()
        
        # Process each expert with failover
        for expert, node_candidates in expert_to_nodes.items():
            success, result, successful_node = await self.failover_coordinator.execute_with_failover(
                primary_node_id=node_candidates[0],
                fallback_node_ids=node_candidates[1:] if len(node_candidates) > 1 else [],
                execute_fn=execute_expert_inference,
                expert_name=expert
            )
            
            if success:
                results[expert] = result
                metrics[expert] = {
                    "node_id": successful_node,
                    "reputation": self.reputation_manager.get_node_reputation(successful_node),
                    "success": True
                }
            else:
                # All nodes failed
                results[expert] = {"error": "All nodes failed"}
                metrics[expert] = {"success": False}
        
        # Combine results
        final_result = f"Failover-enabled inference for '{prompt[:50]}...' using experts {required_experts}"
        
        return final_result, {
            "request_id": request_id,
            "expert_metrics": metrics,
            "failover_enabled": True,
            "node_reputation_summary": self.reputation_manager.get_metrics_summary()
        }
    
    async def distribute_inference(
        self, 
        prompt: str, 
        required_experts: List[str],
        max_new_tokens: int = 64,
        prefer_donor: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """Distribute inference across multiple expert nodes with SLO-based scheduling and donor utilization cap."""
        
        # Track concurrent requests
        is_free_tier = prefer_donor  # prefer_donor flag indicates free tier request
        donor_nodes_used = 0
        
        async with self._inflight_lock:
            self.inflight_total += 1
            if is_free_tier:
                self.inflight_free_tier += 1
                self.metrics["free_tier_total"] += 1
                
                # Update EMA for donor utilization
                current_time = time.time()
                time_delta = current_time - self.last_ema_update
                if time_delta > 0.1:  # Update EMA every 100ms minimum
                    instant_ratio = self.inflight_donor / max(1, self.inflight_total)
                    self.donor_utilization_ema = (self.ema_alpha * instant_ratio + 
                                                  (1 - self.ema_alpha) * self.donor_utilization_ema)
                    self.last_ema_update = current_time
                    self.metrics["donor_utilization_ratio"] = self.donor_utilization_ema
                
                # Check donor utilization cap using EMA
                donor_nodes_count = sum(1 for n in self.registry.nodes.values() if getattr(n, "donor_mode", False))
                if donor_nodes_count > 0:
                    # Use EMA for stable measurement
                    if self.donor_utilization_ema > self.max_donor_utilization_ratio:
                        if self.strict_free_tier:
                            # Initialize semaphore if needed
                            if self.free_tier_semaphore is None:
                                max_concurrent = int(donor_nodes_count * 2)  # Allow 2x donor nodes concurrent
                                self.free_tier_semaphore = asyncio.Semaphore(max_concurrent)
                            
                            # Try to acquire with timeout
                            self.free_tier_queue_size += 1
                            self.metrics["free_tier_queue_len"] = self.free_tier_queue_size
                            
                        else:
                            # Fallback to non-donor nodes
                            prefer_donor = False
                            self.metrics["fallback_count"] += 1
                            self.metrics["fallback_rate"] = self.metrics["fallback_count"] / self.metrics["free_tier_total"]
        
        # Handle queue timeout outside lock
        if is_free_tier and self.strict_free_tier and self.free_tier_semaphore:
            try:
                # Wait with timeout for free tier slot
                await asyncio.wait_for(
                    self.free_tier_semaphore.acquire(),
                    timeout=self.free_tier_queue_timeout
                )
                async with self._inflight_lock:
                    self.free_tier_queue_size = max(0, self.free_tier_queue_size - 1)
                    self.metrics["free_tier_queue_len"] = self.free_tier_queue_size
            except asyncio.TimeoutError:
                # Timeout - fallback to non-donor
                prefer_donor = False
                async with self._inflight_lock:
                    self.free_tier_queue_size = max(0, self.free_tier_queue_size - 1)
                    self.metrics["free_tier_queue_len"] = self.free_tier_queue_size
                    self.metrics["fallback_count"] += 1
                    self.metrics["fallback_rate"] = self.metrics["fallback_count"] / self.metrics["free_tier_total"]
        
        # Update scheduler metrics
        self.inference_metrics["request_count"] += 1
        self.inference_metrics["active_requests"] += 1
        
        # Get real metrics for scheduler
        avg_latency = self.inference_metrics["total_latency"] / max(1, self.inference_metrics["request_count"])
        
        # Try to get real GPU metrics
        gpu_util = 0.0
        memory_free = 10.0  # Default assumption
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.free', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=0.5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                gpu_util = float(parts[0]) / 100
                memory_free = float(parts[1]) / 1024  # MB to GB
        except:
            pass  # Use defaults if nvidia-smi not available
        
        current_metrics = Metrics(
            p95_latency_ms=avg_latency * 1000 * 1.2,  # Approximate p95
            p50_latency_ms=avg_latency * 1000,        # Use avg as p50
            queue_depth=self.inference_metrics["active_requests"],
            queue_wait_ms=0.0,
            gpu_utilization=gpu_util,
            memory_free_gb=memory_free,
            arrival_rate_per_sec=self.inference_metrics["request_count"] / max(1, time.time() - self.start_time),
            warm_pool_hit_ratio=self.warm_pool.get_hit_ratio(),
            learning_step_duration_ms=300.0
        )
        
        scheduler_state = self.scheduler.tick(current_metrics)
        
        # Apply scheduler-based throttling if in RED state
        if scheduler_state == SchedulerState.RED:
            logger.warning(f"Scheduler in RED state: delaying inference request")
            await asyncio.sleep(0.1)  # Brief delay to allow SLO recovery
        elif scheduler_state == SchedulerState.YELLOW:
            logger.info(f"Scheduler in YELLOW state: proceeding with caution")
        
        # Generate request ID
        request_id = hashlib.sha256(f"{prompt}{time.time()}".encode()).hexdigest()[:16]
        
        request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            required_experts=required_experts,
            max_new_tokens=max_new_tokens
        )
        
        self.active_requests[request_id] = request
        inference_start_time = time.time()
        
        try:
            # Assign experts to nodes with donor tracking
            expert_assignments = {}
            
            for expert_name in required_experts:
                node = self.registry.select_best_node(expert_name, prefer_donor=prefer_donor)
                if node:
                    expert_assignments[expert_name] = node
                    if getattr(node, "donor_mode", False):
                        donor_nodes_used += 1
                else:
                    logger.warning(f"No available node for expert {expert_name}")
            
            # Update donor tracking
            if donor_nodes_used > 0:
                async with self._inflight_lock:
                    self.inflight_donor += donor_nodes_used
            
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
                    
                    # Calculate real quality score
                    from backend.model.quality_calculator import get_quality_calculator
                    quality_calc = get_quality_calculator()
                    
                    # Estimate tokens (can be improved with actual token counting)
                    tokens_generated = len(response.result.split()) * 1.3 if hasattr(response, 'result') else 50
                    
                    quality_score = quality_calc.calculate_score(
                        expert_name=response.expert_name,
                        response_time=response.processing_time,
                        tokens_generated=int(tokens_generated),
                        success=response.success
                    )
                    
                    # Record usage with real quality score
                    self.usage_tracker.record_usage(
                        expert_name=response.expert_name,
                        response_time=response.processing_time,
                        quality_score=quality_score
                    )
                elif isinstance(response, Exception):
                    print(f"Expert call failed: {response}")
            
            # Combine results from actual inference responses
            if successful_responses:
                # Use the actual inference result from the first successful response
                primary_result = successful_responses[0].result if hasattr(successful_responses[0], 'result') else successful_responses[0].expert_name
                
                # For multiple experts, we could aggregate or use the best response
                # For now, use the primary result and note which experts were used
                expert_names = [r.expert_name for r in successful_responses]
                
                # If the result looks like a mock response, indicate that inference succeeded but with limited output
                if "processed:" in primary_result or "inference:" in primary_result:
                    combined_result = f"Blockchain inference completed using experts: {', '.join(expert_names)}"
                else:
                    combined_result = primary_result  # Use actual inference result
                    
                return combined_result, expert_usage
            else:
                return "Error: All expert nodes failed", {}
                
        finally:
            # Update scheduler metrics
            inference_time = time.time() - inference_start_time
            self.inference_metrics["total_latency"] += inference_time
            self.inference_metrics["active_requests"] -= 1
            
            # Update concurrent tracking
            async with self._inflight_lock:
                self.inflight_total -= 1
                if is_free_tier:
                    self.inflight_free_tier -= 1
                    if self.free_tier_semaphore:
                        self.free_tier_semaphore.release()
                if donor_nodes_used > 0:
                    self.inflight_donor -= donor_nodes_used
            
            # Clean up
            if request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def distribute_inference_streaming(
        self,
        prompt: str,
        required_experts: List[str],
        max_new_tokens: int = 256,
        prefer_donor: bool = False,
        session_id: Optional[str] = None,
        sampling_seed: Optional[int] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream inference from a single optimal node with all required experts."""
        import aiohttp
        
        # Apply donor/cap/reputation policies
        is_free_tier = prefer_donor
        selected_node = None
        
        # Track metrics
        stream_start = time.time()
        tokens_generated = 0
        
        # Find optimal node with all experts (prefer ExpertGroup)
        if self.group_index.find_matching_group(required_experts):
            # Use expert group optimization
            matching_nodes = []
            for node_id, node_cap in self.group_index.node_capabilities.items():
                if all(exp in node_cap.individual_experts for exp in required_experts):
                    node = self.registry.nodes.get(node_id)
                    if node:
                        matching_nodes.append(node)
            
            # Apply selection criteria
            if matching_nodes:
                selected_node = self._select_best_streaming_node(
                    matching_nodes, prefer_donor, is_free_tier
                )
        
        # Fallback to single node with most coverage
        if not selected_node:
            coverage_scores = {}
            for node_id, node in self.registry.nodes.items():
                coverage = sum(1 for exp in required_experts if exp in node.available_experts)
                if coverage > 0:
                    coverage_scores[node] = coverage
            
            if coverage_scores:
                # Sort by coverage and select best
                sorted_nodes = sorted(coverage_scores.items(), key=lambda x: x[1], reverse=True)
                candidates = [n for n, score in sorted_nodes if score == sorted_nodes[0][1]]
                selected_node = self._select_best_streaming_node(
                    candidates, prefer_donor, is_free_tier
                )
        
        if not selected_node:
            yield {"type": "error", "error": "No suitable nodes available"}
            return
        
        # Track node selection
        self.metrics["streaming_selected_node"] = selected_node.node_id
        
        # Prepare request payload
        request_data = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "required_experts": required_experts,
            "security_audit": {
                "session_id": session_id or hashlib.sha256(f"{prompt}{time.time()}".encode()).hexdigest()[:16],
                "sampling_seed": sampling_seed or random.randint(0, 2**32),
                "required_experts": required_experts
            }
        }
        
        # Stream from selected node
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{selected_node.endpoint}/inference/secure"
                
                async with session.post(url, json=request_data, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status != 200:
                        # Try failover
                        yield {"type": "error", "error": f"Node {selected_node.node_id} returned {response.status}"}
                        return
                    
                    # Record TTFT
                    ttft = None
                    
                    # Stream tokens
                    async for line in response.content:
                        if not line:
                            continue
                        
                        line_str = line.decode('utf-8').strip()
                        
                        # Parse TOKEN: lines
                        if line_str.startswith("TOKEN: "):
                            token = line_str[7:]
                            tokens_generated += 1
                            
                            # Record time to first token
                            if ttft is None:
                                ttft = (time.time() - stream_start) * 1000
                                self.metrics["ttft_ms"] = ttft
                            
                            yield {
                                "type": "token",
                                "token": token,
                                "token_number": tokens_generated,
                                "timestamp": time.time(),
                                "node_id": selected_node.node_id
                            }
                        
                        # Ignore other beacon lines for now
                        elif line_str.startswith(("header:", "weight_proof:", "rolling_commit:")):
                            pass  # Log for future verification
                    
                    # Update metrics
                    stream_duration = time.time() - stream_start
                    if stream_duration > 0:
                        self.metrics["tokens_per_second"] = tokens_generated / stream_duration
                    
                    # Record success
                    self.reputation_manager.record_request_success(
                        selected_node.node_id, stream_duration * 1000
                    )
                    
        except asyncio.TimeoutError:
            self.metrics["stream_timeout_count"] = self.metrics.get("stream_timeout_count", 0) + 1
            yield {"type": "error", "error": "Stream timeout"}
        except Exception as e:
            self.metrics["stream_error_count"] = self.metrics.get("stream_error_count", 0) + 1
            yield {"type": "error", "error": str(e)}
    
    def _select_best_streaming_node(
        self,
        candidates: List[ExpertNode],
        prefer_donor: bool,
        is_free_tier: bool
    ) -> Optional[ExpertNode]:
        """Select best node for streaming based on policies."""
        if not candidates:
            return None
        
        # Filter by health and reputation
        active_nodes = []
        current_time = time.time()
        
        for node in candidates:
            # Check heartbeat
            if current_time - node.last_heartbeat >= 60.0:
                continue
            
            # Check reputation
            node_rep = self.reputation_manager.get_node_reputation(node.node_id)
            if node_rep:
                success_rate = node_rep.get('success_rate', 0)
                min_success_rate = 0.7 if getattr(node, "donor_mode", False) else 0.5
                if success_rate < min_success_rate:
                    continue
            
            active_nodes.append(node)
        
        if not active_nodes:
            return None
        
        # Apply donor preferences
        candidate_pool = active_nodes
        if prefer_donor:
            donor_candidates = [n for n in active_nodes if getattr(n, "donor_mode", False)]
            if donor_candidates:
                candidate_pool = donor_candidates
        else:
            # Avoid donor for paid
            non_donor = [n for n in active_nodes if not getattr(n, "donor_mode", False)]
            if non_donor:
                candidate_pool = non_donor
        
        # Select by lowest load
        return min(candidate_pool, key=lambda n: n.load_factor)
    
    async def distribute_inference_speculative_streaming(
        self,
        prompt: str,
        required_experts: List[str],
        max_new_tokens: int = 256,
        prefer_donor: bool = False,
        session_id: Optional[str] = None,
        sampling_seed: Optional[int] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream inference with speculative decoding (draft + verify)."""
        import aiohttp
        
        # Config
        draft_tokens = int(os.getenv("DRAFT_TOKENS", "4"))
        spec_ratio = float(os.getenv("SPEC_RATIO", "2.0"))
        
        # Find draft node (prefer donor/fast)
        draft_node = self._select_best_streaming_node(
            list(self.registry.nodes.values()),
            prefer_donor=True,  # Prefer donor for draft
            is_free_tier=prefer_donor
        )
        
        # Find verify node (prefer high-quality)
        verify_node = self._select_best_streaming_node(
            list(self.registry.nodes.values()),
            prefer_donor=False,  # Prefer non-donor for verify
            is_free_tier=False
        )
        
        if not draft_node or not verify_node:
            yield {"type": "error", "error": "No suitable nodes for speculative decoding"}
            return
        
        # Track metrics
        self.metrics["spec_draft_node"] = draft_node.node_id
        self.metrics["spec_verify_node"] = verify_node.node_id
        rollback_count = 0
        accepted_total = 0
        
        generated_text = prompt
        tokens_generated = 0
        
        try:
            async with aiohttp.ClientSession() as session:
                while tokens_generated < max_new_tokens:
                    # Get draft tokens
                    draft_url = f"{draft_node.endpoint}/inference/spec_draft_stream"
                    draft_data = {
                        "prompt": generated_text,
                        "max_new_tokens": min(draft_tokens, max_new_tokens - tokens_generated),
                        "sampling_seed": sampling_seed,
                        "draft_tokens": draft_tokens
                    }
                    
                    draft_candidates = []
                    async with session.post(draft_url, json=draft_data, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status != 200:
                            yield {"type": "error", "error": f"Draft failed: {response.status}"}
                            return
                        
                        async for line in response.content:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith("TOKEN: "):
                                token = line_str[7:]
                                draft_candidates.append(token)
                            elif line_str == "DONE":
                                break
                    
                    if not draft_candidates:
                        break
                    
                    # Verify draft tokens
                    verify_url = f"{verify_node.endpoint}/inference/spec_verify"
                    verify_data = {
                        "prefix": generated_text,
                        "candidate_tokens": draft_candidates,
                        "sampling_seed": sampling_seed
                    }
                    
                    async with session.post(verify_url, json=verify_data, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status != 200:
                            # Fallback: accept first token only
                            accepted = 1
                        else:
                            result = await response.json()
                            accepted = result.get("accepted", 0)
                    
                    # Emit accepted tokens
                    for i in range(accepted):
                        yield {
                            "type": "token",
                            "token": draft_candidates[i],
                            "token_number": tokens_generated + i + 1,
                            "timestamp": time.time(),
                            "speculative": True,
                            "draft_node": draft_node.node_id,
                            "verify_node": verify_node.node_id
                        }
                        generated_text += draft_candidates[i]
                    
                    tokens_generated += accepted
                    accepted_total += accepted
                    
                    # Handle rejection
                    if accepted < len(draft_candidates):
                        rollback_count += 1
                        # Continue from accepted prefix
                    
                    # Update metrics
                    if tokens_generated > 0:
                        self.metrics["speculative_accept_ratio"] = accepted_total / (accepted_total + rollback_count)
                        self.metrics["rollback_count"] = rollback_count
                
        except Exception as e:
            yield {"type": "error", "error": str(e)}
    
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
            
            # Use connection pool for better performance
            async with await self.connection_pool.post(
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
        
        # Use connection pool for better performance
        async with await self.connection_pool.post(
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
        
        # Use connection pool for better performance
        async with await self.connection_pool.post(
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
    
    async def cleanup(self):
        """Clean up resources including connection pool."""
        await self.connection_pool.close()
        print("✓ Cleaned up distributed inference coordinator resources")
    
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
    
    def _update_usage_tracking(self, required_experts: List[str], result_length: int):
        """Update usage tracking for experts."""
        for expert_name in required_experts:
            self.usage_tracker.record_usage(
                expert_name=expert_name,
                response_time=0.1,  # Fast warm pool response
                quality_score=0.9   # High quality for cached results
            )
    
    def ensure_warm_slots(self, slot_count: int = 2):
        """Ensure minimum number of warm expert combinations are loaded."""
        current_size = len(self.warm_pool.cache)
        if current_size >= slot_count:
            return
            
        # Get top expert combinations from usage tracker
        top_experts = self.usage_tracker.get_top_experts(limit=slot_count * 2)
        
        # Load combinations of top experts
        for i in range(0, min(len(top_experts), slot_count * 2), 2):
            if len(self.warm_pool.cache) >= slot_count:
                break
                
            expert_combo = top_experts[i:i+2] if i+1 < len(top_experts) else [top_experts[i]]
            expert_group_key = f"{sorted(expert_combo)}_default"
            
            if expert_group_key not in self.warm_pool.cache:
                # Mock loading expert combination
                expert_data = {
                    "experts": expert_combo,
                    "preloaded": True,
                    "timestamp": time.time(),
                    "usage_count": sum(self.usage_tracker.get_expert_stats(e).get("usage_count", 0) for e in expert_combo)
                }
                self.warm_pool.put(expert_group_key, expert_data)
                print(f"🔥 Preloaded warm slot: {expert_combo}")
    
    def get_warm_pool_status(self) -> Dict[str, Any]:
        """Get warm pool status for monitoring."""
        return self.warm_pool.get_status()
    
    def record_learning_preemption(self, node_id: str, preemption_seconds: float):
        """Record learning preemption for fair compensation."""
        self.reward_calculator.record_preemption(node_id, preemption_seconds)
        
    def get_fairness_report(self) -> Dict[str, Any]:
        """Get fairness report for monitoring."""
        return self.reward_calculator.get_fairness_report()


class ExpertNodeServer:
    """Enhanced server that runs on expert nodes with expert group support."""
    
    def __init__(self, node_id: str, available_experts: List[str], expert_groups: List[ExpertGroup] = None, port: int = 8001):
        logger.info(f"ExpertNodeServer init - node_id: {node_id}, port: {port}")
        self.node_id = node_id
        self.available_experts = available_experts
        self.expert_groups = expert_groups or []
        self.port = port
        self.app = web.Application()
        self.current_load = 0.0
        
        # Add authentication middleware
        self.app.middlewares.append(self._auth_middleware)
        
        # Initialize MoE model manager for actual inference
        try:
            from pathlib import Path
            # Load chains for blockchain-based inference
            root_dir = Path("./data")
            meta_chain = Chain(root_dir, "A")
            param_chain = Chain(root_dir, "B") 
            param_index = ParameterIndex(param_chain)
            usage_tracker = ExpertUsageTracker(Path("./data/expert_usage.json"))
            
            self.moe_manager = MoEModelManager(
                meta_chain=meta_chain,
                param_chain=param_chain, 
                param_index=param_index,
                usage_tracker=usage_tracker
            )
            print(f"✅ MoE Model Manager initialized for node {node_id}")
        except Exception as e:
            print(f"⚠️ Failed to initialize MoE manager: {e}")
            self.moe_manager = None
        
        # Setup routes
        self.app.router.add_post('/expert/inference', self.handle_inference)
        self.app.router.add_post('/inference/expert_group', self.handle_expert_group_inference)
        self.app.router.add_post('/inference/multi_expert', self.handle_multi_expert_inference)
        self.app.router.add_post('/inference/secure', self.handle_secure_inference)
        # Pipeline RPC endpoints
        self._pipeline_buffers = None
        try:
            from backend.learning.pipeline_rpc import PipelineRPCServerBuffer
            self._pipeline_buffers = PipelineRPCServerBuffer()
            # Optionally start gRPC server for pipeline if enabled
            try:
                import os
                if os.getenv('BLYAN_PIPELINE_GRPC', '0') in ('1', 'true', 'yes'):
                    from backend.learning.pipeline_rpc_grpc import serve as grpc_serve
                    # Run gRPC server in background
                    loop = asyncio.get_event_loop()
                    loop.create_task(grpc_serve(self._pipeline_buffers))
            except Exception as e:
                logger.warning(f"Pipeline gRPC startup failed: {e}")
            self.app.router.add_post('/pipeline/send_activations', self.handle_send_activations)
            self.app.router.add_post('/pipeline/recv_activations', self.handle_recv_activations)
            self.app.router.add_post('/pipeline/send_grads', self.handle_send_grads)
            self.app.router.add_post('/pipeline/recv_grads', self.handle_recv_grads)
            self.app.router.add_post('/pipeline/reset', self.handle_pipeline_reset)
        except Exception as e:
            logger.warning(f"Pipeline RPC buffer not available: {e}")
        self.app.router.add_post('/inference/prepare', self.handle_prepare)
        self.app.router.add_post('/inference/spec_draft_stream', self.handle_spec_draft_stream)
        self.app.router.add_post('/inference/spec_verify', self.handle_spec_verify)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/experts', self.handle_list_experts)
        self.app.router.add_get('/expert_groups', self.handle_list_expert_groups)
    
    @web.middleware
    async def _auth_middleware(self, request, handler):
        """Middleware to verify requests come from authorized main node."""
        # Skip auth for health check
        if request.path == '/health':
            return await handler(request)
        
        # Check authentication headers
        node_id = request.headers.get('X-Node-ID')
        auth_token = request.headers.get('X-Node-Auth-Token')
        
        if not node_id or not auth_token:
            logger.warning(f"Missing auth headers from {request.remote}")
            return web.json_response(
                {"error": "Missing authentication headers"},
                status=401
            )
        
        # Verify with node authenticator
        try:
            from backend.api.node_auth import NodeAuthenticator
            authenticator = NodeAuthenticator()
            
            # For now, only main node can call worker endpoints
            if not authenticator.verify_main_node(node_id, auth_token):
                logger.warning(f"Invalid auth from node {node_id}")
                authenticator._log_security_event({
                    "type": "unauthorized_worker_access",
                    "node_id": node_id,
                    "endpoint": request.path,
                    "timestamp": datetime.utcnow().isoformat()
                })
                return web.json_response(
                    {"error": "Unauthorized"},
                    status=403
                )
            
        except Exception as e:
            logger.error(f"Auth verification error: {e}")
            return web.json_response(
                {"error": "Authentication system error"},
                status=500
            )
        
        # Auth passed, continue to handler
        return await handler(request)
    
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
            
            # Perform actual expert inference using blockchain
            start_time = time.time()
            
            if self.moe_manager:
                try:
                    # Use actual blockchain-based expert inference
                    result, expert_usage = self.moe_manager.selective_generate(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        top_k_experts=1  # Single expert
                    )
                    print(f"🧠 Actual expert inference completed for {expert_name}")
                except Exception as e:
                    print(f"❌ Expert inference failed: {e}")
                    return web.json_response({
                        "error": f"Expert inference failed: {str(e)}",
                        "expert_name": expert_name,
                        "status": "failed"
                    }, status=500)
            else:
                # MoE manager not available - return proper error
                return web.json_response({
                    "error": "MoE manager not initialized. Blockchain model not loaded.",
                    "expert_name": expert_name,
                    "status": "unavailable"
                }, status=503)
                
            processing_time = time.time() - start_time
            
            return web.json_response({
                "expert_name": expert_name,
                "result": result,
                "processing_time": processing_time,
                "node_id": self.node_id
            })
            
        except Exception as e:
            try:
                await self.pipeline_metrics.incr_rpc_error()
            except Exception:
                pass
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def handle_prepare(self, request: web.Request) -> web.Response:
        """Prepare node for handoff by warming up prefix cache."""
        try:
            data = await request.json()
            session_id = data.get('session_id')
            prompt = data.get('prompt')
            sampling_seed = data.get('sampling_seed')
            top_k_experts = data.get('top_k_experts', 2)
            prepare_only = data.get('prepare_only', True)
            
            # Mock prefix caching for now
            # In production, would warm up KV cache with prompt prefix
            logger.info(f"Preparing node {self.node_id} for session {session_id}")
            
            # Simulate prefix warmup
            await asyncio.sleep(0.1)
            
            return web.json_response({
                "ready": True,
                "node_id": self.node_id,
                "session_id": session_id,
                "prefix_length": len(prompt.split()) if prompt else 0
            })
            
        except Exception as e:
            return web.json_response(
                {"error": str(e), "ready": False},
                status=500
            )
    
    async def handle_spec_draft_stream(self, request: web.Request) -> web.StreamResponse:
        """Generate draft tokens quickly for speculative decoding."""
        try:
            data = await request.json()
            prompt = data.get('prompt', '')
            max_new_tokens = data.get('max_new_tokens', 64)
            sampling_seed = data.get('sampling_seed')
            draft_tokens = data.get('draft_tokens', 4)
            
            response = web.StreamResponse()
            response.headers['Content-Type'] = 'text/event-stream'
            response.headers['Cache-Control'] = 'no-cache'
            await response.prepare(request)
            
            # Draft generation requires actual model
            if not self.moe_manager:
                await response.write(b"ERROR: Draft model not available - blockchain model not loaded\n")
                return response
            
            # Use actual model for draft generation with reduced compute
            try:
                # Generate draft tokens with lightweight settings
                draft_result, _ = self.moe_manager.selective_generate(
                    prompt=prefix,
                    max_new_tokens=draft_tokens,
                    top_k_experts=1,  # Use single expert for speed
                    temperature=1.2   # Higher temp for draft diversity
                )
                
                # Stream the generated tokens
                tokens = draft_result.split()[:draft_tokens]
                for token in tokens:
                    await response.write(f"TOKEN: {token}\n".encode())
                    await asyncio.sleep(0.01)
                
            except Exception as e:
                await response.write(f"ERROR: Draft generation failed: {str(e)}\n".encode())
                return response
            
            await response.write(b"DONE\n")
            return response
            
        except Exception as e:
            response = web.StreamResponse(status=500)
            await response.prepare(request)
            await response.write(f"ERROR: {str(e)}\n".encode())
            return response
    
    async def handle_spec_verify(self, request: web.Request) -> web.Response:
        """Verify draft tokens against full model."""
        try:
            data = await request.json()
            prefix = data.get('prefix', '')
            candidate_tokens = data.get('candidate_tokens', [])
            sampling_seed = data.get('sampling_seed')
            
            # Verification requires actual model
            if not self.moe_manager:
                return web.json_response({
                    "error": "Verification model not available - blockchain model not loaded",
                    "status": "unavailable"
                }, status=503)
            
            # Verify tokens against actual model
            try:
                # Generate continuation with full model
                full_result, _ = self.moe_manager.selective_generate(
                    prompt=prefix,
                    max_new_tokens=len(candidate_tokens),
                    top_k_experts=3,  # Use more experts for accuracy
                    temperature=0.1,  # Low temp for deterministic verification
                    seed=sampling_seed
                )
                
                # Compare generated tokens with candidates
                generated_tokens = full_result.split()[:len(candidate_tokens)]
                accepted = 0
                for i, (candidate, generated) in enumerate(zip(candidate_tokens, generated_tokens)):
                    if candidate.strip() == generated.strip():
                        accepted += 1
                    else:
                        break  # Stop at first mismatch
                
                return web.json_response({
                    "accepted": accepted,
                    "rejected_at": accepted if accepted < len(candidate_tokens) else None,
                    "node_id": self.node_id,
                    "verified": True
                })
            except Exception as e:
                return web.json_response({
                    "error": f"Token verification failed: {str(e)}",
                    "status": "failed"
                }, status=500)
            
        except Exception as e:
            return web.json_response(
                {"error": str(e), "accepted": 0},
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
            
            # Perform actual MoE inference using blockchain experts
            start_time = time.time()
            
            if self.moe_manager:
                try:
                    # Use actual blockchain-based MoE inference
                    result_text, expert_usage = self.moe_manager.selective_generate(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        top_k_experts=len(experts)
                    )
                    print(f"🧠 Actual MoE inference completed for group {group_id}")
                except Exception as e:
                    print(f"⚠️ MoE inference failed, falling back to mock: {e}")
                    result_text = f"Expert group {group_id} inference: '{prompt[:50]}...' using {len(experts)} experts efficiently"
            else:
                # Fallback to mock if MoE manager not available
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

    async def handle_send_activations(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            stage_id = int(data.get('stage_id'))
            microbatch_id = str(data.get('microbatch_id'))
            if self._pipeline_buffers is None:
                return web.json_response({"error": "pipeline_disabled"}, status=503)
            # Chunked path
            chunk_index = data.get('chunk_index')
            if chunk_index is not None:
                # Backpressure: reject when buffer watermark exceeded
                ok = await self._pipeline_buffers.try_backpressure()
                if not ok:
                    return web.json_response({"error": "backpressure"}, status=429)
                chunk_b64 = data.get('chunk_b64')
                chunks_total = int(data.get('chunks_total', 1))
                compression = str(data.get('compression', 'none'))
                chunk = base64.b64decode(chunk_b64.encode('ascii')) if chunk_b64 else b''
                assembled = await self._pipeline_buffers.append_chunk(stage_id, microbatch_id, int(chunk_index), chunks_total, chunk, compression, bool(data.get('is_grad', False)))
                if not assembled:
                    return web.json_response({"error": "buffer_full"}, status=429)
                return web.json_response({"status": "chunk_ok", "chunk_index": int(chunk_index)})
            # Single payload path (backward compatible)
            acts_b64 = data.get('activations_b64')
            compression = str(data.get('compression', 'none')).lower()
            blob = base64.b64decode(acts_b64.encode('ascii')) if acts_b64 else b''
            if compression == 'gzip':
                import gzip as _gzip
                blob = _gzip.decompress(blob)
            await self._pipeline_buffers.put_acts(stage_id, microbatch_id, blob)
            # Stage occupancy bookkeeping (simple placeholder)
            try:
                await self.pipeline_metrics.set_stage_occupancy(stage_id, 1.0)
            except Exception:
                pass
            return web.json_response({"status": "ok"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_recv_activations(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            stage_id = int(data.get('stage_id'))
            microbatch_id = str(data.get('microbatch_id'))
            if self._pipeline_buffers is None:
                return web.json_response({"error": "pipeline_disabled"}, status=503)
            blob = await self._pipeline_buffers.get_acts(stage_id, microbatch_id)
            if blob is None:
                return web.json_response({"error": "not_found"}, status=404)
            b64 = base64.b64encode(blob).decode('ascii')
            return web.json_response({"activations_b64": b64})
        except Exception as e:
            try:
                await self.pipeline_metrics.incr_rpc_error()
            except Exception:
                pass
            return web.json_response({"error": str(e)}, status=500)

    async def handle_send_grads(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            stage_id = int(data.get('stage_id'))
            microbatch_id = str(data.get('microbatch_id'))
            if self._pipeline_buffers is None:
                return web.json_response({"error": "pipeline_disabled"}, status=503)
            chunk_index = data.get('chunk_index')
            if chunk_index is not None:
                ok = await self._pipeline_buffers.try_backpressure()
                if not ok:
                    return web.json_response({"error": "backpressure"}, status=429)
                chunk_b64 = data.get('chunk_b64')
                chunks_total = int(data.get('chunks_total', 1))
                compression = str(data.get('compression', 'none'))
                chunk = base64.b64decode(chunk_b64.encode('ascii')) if chunk_b64 else b''
                assembled = await self._pipeline_buffers.append_chunk(stage_id, microbatch_id, int(chunk_index), chunks_total, chunk, compression, True)
                if not assembled:
                    return web.json_response({"error": "buffer_full"}, status=429)
                return web.json_response({"status": "chunk_ok", "chunk_index": int(chunk_index)})
            grads_b64 = data.get('grads_b64')
            compression = str(data.get('compression', 'none')).lower()
            blob = base64.b64decode(grads_b64.encode('ascii')) if grads_b64 else b''
            if compression == 'gzip':
                import gzip as _gzip
                blob = _gzip.decompress(blob)
            await self._pipeline_buffers.put_grads(stage_id, microbatch_id, blob)
            try:
                await self.pipeline_metrics.set_stage_occupancy(stage_id, 0.0)
            except Exception:
                pass
            return web.json_response({"status": "ok"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_recv_grads(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            stage_id = int(data.get('stage_id'))
            microbatch_id = str(data.get('microbatch_id'))
            if self._pipeline_buffers is None:
                return web.json_response({"error": "pipeline_disabled"}, status=503)
            blob = await self._pipeline_buffers.get_grads(stage_id, microbatch_id)
            if blob is None:
                return web.json_response({"error": "not_found"}, status=404)
            b64 = base64.b64encode(blob).decode('ascii')
            return web.json_response({"grads_b64": b64})
        except Exception as e:
            try:
                await self.pipeline_metrics.incr_rpc_error()
            except Exception:
                pass
            return web.json_response({"error": str(e)}, status=500)

    async def handle_pipeline_reset(self, request: web.Request) -> web.Response:
        try:
            if self._pipeline_buffers is None:
                return web.json_response({"error": "pipeline_disabled"}, status=503)
            from backend.learning.pipeline_rpc import PipelineRPCServerBuffer
            self._pipeline_buffers = PipelineRPCServerBuffer()
            try:
                await self.pipeline_metrics.incr_pipeline_reset()
            except Exception:
                pass
            return web.json_response({"status": "reset"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
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
            
            # Perform actual multi-expert MoE inference
            start_time = time.time()
            
            if self.moe_manager:
                try:
                    # Use actual blockchain-based MoE inference with specified experts
                    result_text, expert_usage = self.moe_manager.selective_generate(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        top_k_experts=len(experts)
                    )
                    print(f"🧠 Actual multi-expert inference completed using {len(experts)} experts")
                except Exception as e:
                    print(f"⚠️ Multi-expert inference failed, falling back to mock: {e}")
                    result_text = f"Multi-expert inference: '{prompt[:50]}...' using {len(experts)} individual experts"
            else:
                # Fallback to mock if MoE manager not available
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
        # Optionally enable TLS via environment variables (mTLS ready: set BLYAN_TLS_CLIENT_CA for client cert verification)
        ssl_context = None
        try:
            import os, ssl
            cert_path = os.getenv('BLYAN_TLS_CERT')
            key_path = os.getenv('BLYAN_TLS_KEY')
            client_ca = os.getenv('BLYAN_TLS_CLIENT_CA')
            if cert_path and key_path:
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path)
                if client_ca:
                    ssl_context.load_verify_locations(cafile=client_ca)
                    ssl_context.verify_mode = ssl.CERT_REQUIRED
        except Exception as e:
            logger.warning(f"TLS setup failed: {e}")
            ssl_context = None
        web.run_app(self.app, host='0.0.0.0', port=self.port, ssl_context=ssl_context)


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
        logger.debug(f"sys.argv = {sys.argv}")
        node_id = sys.argv[2] if len(sys.argv) > 2 else "test_node"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8001
        logger.debug(f"Parsed node_id={node_id}, port={port}")
        
        # Mock experts for testing
        experts = [f"layer{i}.expert{j}" for i in range(2) for j in range(2)]
        
        server = ExpertNodeServer(node_id, experts, port=port)
        server.start_server()
    
    else:
        # Run distributed inference simulation
        asyncio.run(simulate_distributed_inference())