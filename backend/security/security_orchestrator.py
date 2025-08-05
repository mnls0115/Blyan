"""
Security Orchestrator for Blyan

This module implements production-grade security orchestration including:
1. Automatic failover when integrity verification fails
2. Adaptive beacon randomization and threshold management
3. Node quarantine and automatic recovery
4. Real-time threat monitoring and alerting
5. Security metrics and dashboard integration
"""

from __future__ import annotations

import asyncio
import json
import time
import random
import statistics
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import logging

from .inference_integrity import InferenceIntegrityCoordinator, SecurityBeacon
from backend.p2p.expert_group_optimizer import NodeCapability, ExpertGroupIndex


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    beacon_failure_threshold: int = 2  # Consecutive failures before quarantine
    integrity_score_threshold: float = 0.7  # Minimum acceptable integrity score
    grace_window_seconds: float = 5.0  # Grace period for network jitter
    canary_rotation_interval: int = 100  # Rotate canaries every N requests
    beacon_randomization_level: float = 0.3  # How much to randomize beacon parameters
    auto_quarantine_enabled: bool = True
    max_retry_attempts: int = 3
    node_recovery_timeout: int = 300  # 5 minutes


@dataclass
class NodeSecurityState:
    """Security state tracking for individual nodes."""
    node_id: str
    trust_score: float = 1.0
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    last_failure_time: float = 0.0
    quarantined: bool = False
    quarantine_start: float = 0.0
    integrity_scores: deque = None  # Recent integrity scores
    failure_reasons: List[str] = None
    
    def __post_init__(self):
        if self.integrity_scores is None:
            self.integrity_scores = deque(maxlen=50)  # Keep last 50 scores
        if self.failure_reasons is None:
            self.failure_reasons = []
    
    @property
    def success_rate(self) -> float:
        return self.successful_requests / max(self.total_requests, 1)
    
    @property
    def average_integrity_score(self) -> float:
        return statistics.mean(self.integrity_scores) if self.integrity_scores else 0.0


@dataclass
class SecurityAlert:
    """Security alert for monitoring integration."""
    alert_type: str  # "node_quarantine", "integrity_failure", "attack_detected"
    severity: str    # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    node_id: Optional[str]
    message: str
    timestamp: float
    metadata: Dict[str, Any]
    
    def to_webhook_payload(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "node_id": self.node_id,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class AdaptiveBeaconManager:
    """Manages adaptive beacon randomization and optimization."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.canary_templates = [
            {"trigger": "quantum_physics", "experts": ["layer0.expert0", "layer1.expert1"]},
            {"trigger": "machine_learning", "experts": ["layer0.expert1", "layer1.expert2"]},
            {"trigger": "data_analysis", "experts": ["layer0.expert2", "layer1.expert3"]},
            {"trigger": "python_programming", "experts": ["layer0.expert6", "layer1.expert7"]},
            {"trigger": "language_translation", "experts": ["layer0.expert3", "layer1.expert4"]},
            {"trigger": "mathematical_proof", "experts": ["layer0.expert0", "layer1.expert5"]},
        ]
        self.current_template_index = 0
        self.requests_since_rotation = 0
        
    def get_randomized_beacon_config(self, request_id: str) -> Dict[str, Any]:
        """Generate randomized beacon configuration for a request."""
        # Use request_id as seed for reproducible randomization
        random.seed(hash(request_id) % (2**32))
        
        # Randomize activation checkpoints
        base_checkpoints = [0, 2, 4]  # Default layers
        if random.random() < self.policy.beacon_randomization_level:
            # Add random checkpoint
            extra_checkpoint = random.randint(1, 6)
            base_checkpoints.append(extra_checkpoint)
        
        # Randomize weight proof pages
        total_pages = 100  # Assume 100 pages per expert
        num_proof_pages = random.randint(2, 6)
        proof_pages = random.sample(range(total_pages), num_proof_pages)
        
        # Randomize rolling commitment interval
        base_interval = 3  # Every 3 tokens
        commitment_interval = random.randint(2, 6)
        
        return {
            "activation_checkpoints": sorted(base_checkpoints),
            "weight_proof_pages": proof_pages,
            "rolling_commitment_interval": commitment_interval,
            "beacon_frequency": random.uniform(0.8, 1.2)  # ±20% variation
        }
    
    def get_current_canary_template(self) -> Dict[str, Any]:
        """Get current canary template, rotating as needed."""
        self.requests_since_rotation += 1
        
        if self.requests_since_rotation >= self.policy.canary_rotation_interval:
            self.current_template_index = (self.current_template_index + 1) % len(self.canary_templates)
            self.requests_since_rotation = 0
        
        template = self.canary_templates[self.current_template_index].copy()
        
        # Add time-based variation
        time_variant = int(time.time() / 3600) % 1000  # Change every hour
        template["variant"] = time_variant
        template["trigger"] = f"{template['trigger']}_{time_variant}"
        
        return template


class SecurityOrchestrator:
    """Main security orchestration system."""
    
    def __init__(
        self, 
        integrity_coordinator: InferenceIntegrityCoordinator,
        group_index: ExpertGroupIndex,
        policy: SecurityPolicy = None
    ):
        self.integrity_coordinator = integrity_coordinator
        self.group_index = group_index
        self.policy = policy or SecurityPolicy()
        
        # State tracking
        self.node_states: Dict[str, NodeSecurityState] = {}
        self.quarantined_nodes: Set[str] = set()
        self.security_alerts: deque = deque(maxlen=1000)  # Keep last 1000 alerts
        
        # Components
        self.beacon_manager = AdaptiveBeaconManager(self.policy)
        self.alert_callbacks: List[Callable[[SecurityAlert], None]] = []
        
        # Metrics
        self.total_requests = 0
        self.failed_requests = 0
        self.failover_count = 0
        self.average_integrity_score = 0.0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def register_alert_callback(self, callback: Callable[[SecurityAlert], None]):
        """Register callback for security alerts (e.g., webhook, Slack)."""
        self.alert_callbacks.append(callback)
    
    def _emit_alert(self, alert: SecurityAlert):
        """Emit security alert to registered callbacks."""
        self.security_alerts.append(alert)
        self.logger.warning(f"Security Alert: {alert.alert_type} - {alert.message}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def _get_node_state(self, node_id: str) -> NodeSecurityState:
        """Get or create node security state."""
        if node_id not in self.node_states:
            self.node_states[node_id] = NodeSecurityState(node_id=node_id)
        return self.node_states[node_id]
    
    def _update_node_metrics(self, node_id: str, success: bool, integrity_score: float = None, failure_reason: str = None):
        """Update node security metrics."""
        state = self._get_node_state(node_id)
        state.total_requests += 1
        
        if success:
            state.successful_requests += 1
            state.consecutive_failures = 0
            if integrity_score is not None:
                state.integrity_scores.append(integrity_score)
        else:
            state.consecutive_failures += 1
            state.last_failure_time = time.time()
            if failure_reason:
                state.failure_reasons.append(failure_reason)
                state.failure_reasons = state.failure_reasons[-10:]  # Keep last 10 reasons
        
        # Update trust score (exponential moving average)
        alpha = 0.1  # Learning rate
        new_score = 1.0 if success else 0.0
        state.trust_score = alpha * new_score + (1 - alpha) * state.trust_score
    
    def _should_quarantine_node(self, node_id: str) -> bool:
        """Determine if node should be quarantined."""
        if not self.policy.auto_quarantine_enabled:
            return False
        
        state = self._get_node_state(node_id)
        
        # Check consecutive failures
        if state.consecutive_failures >= self.policy.beacon_failure_threshold:
            return True
        
        # Check integrity score trend
        if len(state.integrity_scores) >= 5:
            recent_avg = statistics.mean(list(state.integrity_scores)[-5:])
            if recent_avg < self.policy.integrity_score_threshold:
                return True
        
        # Check trust score
        if state.trust_score < 0.3:  # Very low trust
            return True
        
        return False
    
    def quarantine_node(self, node_id: str, reason: str):
        """Quarantine a node due to security issues."""
        if node_id in self.quarantined_nodes:
            return  # Already quarantined
        
        self.quarantined_nodes.add(node_id)
        state = self._get_node_state(node_id)
        state.quarantined = True
        state.quarantine_start = time.time()
        
        alert = SecurityAlert(
            alert_type="node_quarantine",
            severity="HIGH",
            node_id=node_id,
            message=f"Node {node_id} quarantined: {reason}",
            timestamp=time.time(),
            metadata={
                "reason": reason,
                "consecutive_failures": state.consecutive_failures,
                "trust_score": state.trust_score,
                "success_rate": state.success_rate
            }
        )
        
        self._emit_alert(alert)
        self.logger.warning(f"Quarantined node {node_id}: {reason}")
    
    def can_use_node(self, node_id: str) -> bool:
        """Check if node can be used for inference."""
        if node_id in self.quarantined_nodes:
            # Check if quarantine period has expired
            state = self._get_node_state(node_id)
            if time.time() - state.quarantine_start > self.policy.node_recovery_timeout:
                self.attempt_node_recovery(node_id)
            else:
                return False
        
        return node_id not in self.quarantined_nodes
    
    def attempt_node_recovery(self, node_id: str):
        """Attempt to recover a quarantined node."""
        if node_id not in self.quarantined_nodes:
            return
        
        state = self._get_node_state(node_id)
        
        # Reset some metrics for recovery attempt
        state.consecutive_failures = 0
        state.quarantined = False
        self.quarantined_nodes.remove(node_id)
        
        alert = SecurityAlert(
            alert_type="node_recovery",
            severity="MEDIUM",
            node_id=node_id,
            message=f"Attempting recovery for node {node_id}",
            timestamp=time.time(),
            metadata={"quarantine_duration": time.time() - state.quarantine_start}
        )
        
        self._emit_alert(alert)
        self.logger.info(f"Attempting recovery for node {node_id}")
    
    async def secure_inference_with_failover(
        self,
        prompt: str,
        required_experts: List[str],
        max_new_tokens: int = 64,
        preferred_region: str = "default"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute secure inference with automatic failover on integrity failures.
        """
        self.total_requests += 1
        
        # Generate adaptive beacon configuration
        request_id = f"req_{int(time.time()*1000)}_{random.randint(1000, 9999)}"
        beacon_config = self.beacon_manager.get_randomized_beacon_config(request_id)
        
        # Find available nodes (excluding quarantined ones)
        candidates = self.group_index.find_optimal_nodes(set(required_experts))
        available_candidates = [
            (node, group) for node, group in candidates 
            if self.can_use_node(node.node_id)
        ]
        
        if not available_candidates:
            self.failed_requests += 1
            return "Error: No available secure nodes", {
                "error": "All nodes quarantined or unavailable",
                "quarantined_nodes": list(self.quarantined_nodes)
            }
        
        # Try nodes in order of preference
        last_error = None
        attempts = 0
        
        for node, group in available_candidates[:self.policy.max_retry_attempts]:
            attempts += 1
            
            try:
                # Initialize audit with adaptive configuration
                audit_context = self.integrity_coordinator.initialize_audit(
                    request_id=f"{request_id}_attempt_{attempts}",
                    prompt=prompt,
                    required_experts=required_experts
                )
                
                # Apply adaptive beacon configuration
                audit_context.activation_checkpoints = beacon_config["activation_checkpoints"]
                
                # Mock secure inference call (in production, this would call the actual node)
                result, security_report = await self._mock_secure_inference_call(
                    node, audit_context, prompt, max_new_tokens, beacon_config
                )
                
                # Analyze security results
                integrity_score = security_report.get("integrity_score", 0.0)
                trust_level = security_report.get("trust_level", "UNKNOWN")
                anomalies = security_report.get("anomalies", [])
                
                # Update metrics
                success = (
                    integrity_score >= self.policy.integrity_score_threshold and
                    trust_level in ["HIGH", "MEDIUM"] and
                    len(anomalies) == 0
                )
                
                self._update_node_metrics(node.node_id, success, integrity_score)
                
                if success:
                    # Success - update global metrics and return
                    self.average_integrity_score = (
                        0.9 * self.average_integrity_score + 0.1 * integrity_score
                    )
                    
                    return result, {
                        "success": True,
                        "node_used": node.node_id,
                        "attempts": attempts,
                        "security_verification": security_report,
                        "beacon_config": beacon_config,
                        "failover_occurred": attempts > 1
                    }
                
                else:
                    # Integrity failure - check if should quarantine
                    failure_reason = f"Integrity failure: score={integrity_score:.3f}, anomalies={len(anomalies)}"
                    self._update_node_metrics(node.node_id, False, failure_reason=failure_reason)
                    
                    if self._should_quarantine_node(node.node_id):
                        self.quarantine_node(node.node_id, failure_reason)
                    
                    # Emit security alert
                    alert = SecurityAlert(
                        alert_type="integrity_failure",
                        severity="MEDIUM",
                        node_id=node.node_id,
                        message=f"Integrity verification failed: {failure_reason}",
                        timestamp=time.time(),
                        metadata={
                            "integrity_score": integrity_score,
                            "anomalies": anomalies,
                            "trust_level": trust_level
                        }
                    )
                    self._emit_alert(alert)
                    
                    last_error = failure_reason
                    continue  # Try next node
            
            except Exception as e:
                # Network or other error
                error_msg = f"Node communication error: {str(e)}"
                self._update_node_metrics(node.node_id, False, failure_reason=error_msg)
                last_error = error_msg
                continue
        
        # All nodes failed
        self.failed_requests += 1
        self.failover_count += 1
        
        return f"Error: All available nodes failed. Last error: {last_error}", {
            "error": "All nodes failed",
            "attempts": attempts,
            "last_error": last_error,
            "available_nodes": len(available_candidates),
            "quarantined_nodes": list(self.quarantined_nodes)
        }
    
    async def _mock_secure_inference_call(
        self, 
        node: NodeCapability, 
        audit_context, 
        prompt: str, 
        max_new_tokens: int,
        beacon_config: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Mock secure inference call for demo purposes."""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Simulate various security outcomes based on node trust
        node_state = self._get_node_state(node.node_id)
        
        # Simulate integrity score based on node history
        base_score = node_state.trust_score
        noise = random.uniform(-0.2, 0.2)
        integrity_score = max(0.0, min(1.0, base_score + noise))
        
        # Simulate anomalies
        anomalies = []
        if integrity_score < 0.6:
            anomalies.append("activation_mismatch")
        if random.random() < 0.1:  # 10% chance of routing anomaly
            anomalies.append("routing_deviation")
        
        # Determine trust level
        if integrity_score >= 0.9 and len(anomalies) == 0:
            trust_level = "HIGH"
        elif integrity_score >= 0.7 and len(anomalies) <= 1:
            trust_level = "MEDIUM"
        else:
            trust_level = "LOW"
        
        security_report = {
            "verification_enabled": True,
            "beacon_count": len(beacon_config["activation_checkpoints"]) + 3,  # header + footer + weight
            "integrity_score": integrity_score,
            "trust_level": trust_level,
            "anomalies": anomalies,
            "verified_components": ["activation_beacons", "rolling_commitments", "weight_proofs"],
            "beacon_config_applied": beacon_config
        }
        
        result_text = f"Secure inference result from {node.node_id}: {prompt[:30]}..."
        
        return result_text, security_report
    
    def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive security metrics for dashboard."""
        current_time = time.time()
        
        # Node trust scores
        node_trust_scores = {
            node_id: state.trust_score 
            for node_id, state in self.node_states.items()
        }
        
        # Recent alerts by severity
        recent_alerts = list(self.security_alerts)[-100:]  # Last 100 alerts
        alert_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for alert in recent_alerts:
            alert_counts[alert.severity] += 1
        
        # Integrity score distribution
        all_integrity_scores = []
        for state in self.node_states.values():
            all_integrity_scores.extend(list(state.integrity_scores))
        
        integrity_stats = {}
        if all_integrity_scores:
            integrity_stats = {
                "mean": statistics.mean(all_integrity_scores),
                "median": statistics.median(all_integrity_scores),
                "p95": sorted(all_integrity_scores)[int(0.95 * len(all_integrity_scores))] if len(all_integrity_scores) > 20 else 0,
                "p99": sorted(all_integrity_scores)[int(0.99 * len(all_integrity_scores))] if len(all_integrity_scores) > 100 else 0
            }
        
        # Failure analysis
        failure_reasons = defaultdict(int)
        for state in self.node_states.values():
            for reason in state.failure_reasons:
                failure_reasons[reason] += 1
        
        return {
            "overview": {
                "total_requests": self.total_requests,
                "failed_requests": self.failed_requests,
                "success_rate": 1.0 - (self.failed_requests / max(self.total_requests, 1)),
                "failover_count": self.failover_count,
                "average_integrity_score": self.average_integrity_score,
                "quarantined_nodes": len(self.quarantined_nodes),
                "active_nodes": len([s for s in self.node_states.values() if not s.quarantined])
            },
            "node_metrics": {
                node_id: {
                    "trust_score": state.trust_score,
                    "success_rate": state.success_rate,
                    "total_requests": state.total_requests,
                    "consecutive_failures": state.consecutive_failures,
                    "quarantined": state.quarantined,
                    "average_integrity": state.average_integrity_score
                }
                for node_id, state in self.node_states.items()
            },
            "security_alerts": {
                "total": len(recent_alerts),
                "by_severity": alert_counts,
                "recent": [
                    {
                        "type": alert.alert_type,
                        "severity": alert.severity,
                        "message": alert.message,
                        "timestamp": alert.timestamp,
                        "node_id": alert.node_id
                    }
                    for alert in recent_alerts[-10:]  # Last 10 alerts
                ]
            },
            "integrity_distribution": integrity_stats,
            "failure_analysis": dict(failure_reasons),
            "policy_settings": asdict(self.policy),
            "quarantined_nodes": [
                {
                    "node_id": node_id,
                    "quarantine_duration": current_time - self.node_states[node_id].quarantine_start,
                    "reason": self.node_states[node_id].failure_reasons[-1] if self.node_states[node_id].failure_reasons else "Unknown"
                }
                for node_id in self.quarantined_nodes
            ]
        }


# Webhook integration example
async def send_slack_alert(alert: SecurityAlert):
    """Example webhook integration for Slack notifications."""
    webhook_payload = {
        "text": f"🚨 Blyan Security Alert: {alert.alert_type}",
        "attachments": [
            {
                "color": "danger" if alert.severity in ["HIGH", "CRITICAL"] else "warning",
                "fields": [
                    {"title": "Severity", "value": alert.severity, "short": True},
                    {"title": "Node", "value": alert.node_id or "N/A", "short": True},
                    {"title": "Message", "value": alert.message, "short": False}
                ],
                "timestamp": int(alert.timestamp)
            }
        ]
    }
    
    # In production, send to actual Slack webhook
    print(f"[MOCK SLACK] {json.dumps(webhook_payload, indent=2)}")


# Factory function
def create_security_orchestrator(
    integrity_coordinator: InferenceIntegrityCoordinator,
    group_index: ExpertGroupIndex,
    policy: SecurityPolicy = None
) -> SecurityOrchestrator:
    """Create security orchestrator with default configuration."""
    orchestrator = SecurityOrchestrator(integrity_coordinator, group_index, policy)
    
    # Register default alert handlers
    orchestrator.register_alert_callback(lambda alert: asyncio.create_task(send_slack_alert(alert)))
    
    return orchestrator