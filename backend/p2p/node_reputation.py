#!/usr/bin/env python3
"""
Node Reputation and Failover System
Tracks node reliability and provides automatic failover

### PRODUCTION FEATURES ###
- Node reputation scoring
- Automatic failover on failure
- Exponential backoff for failed nodes
- Health check monitoring
- Performance-based routing
"""

import time
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class NodeMetrics:
    """Performance metrics for a node."""
    node_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_failure: Optional[float] = None
    consecutive_failures: int = 0
    reputation_score: float = 100.0  # 0-100 scale
    is_healthy: bool = True
    last_health_check: float = 0.0
    
    # Time-windowed metrics for recent performance
    window_start_time: float = field(default_factory=time.time)
    window_requests: int = 0
    window_successes: int = 0
    window_duration_seconds: float = 600.0  # 10 minute window
    
    # Node registration time for grace period
    registered_at: float = field(default_factory=time.time)
    grace_period_seconds: float = 300.0  # 5 minute grace period for new nodes
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate with time window and grace period."""
        current_time = time.time()
        
        # Grace period for new nodes - assume high success rate
        if current_time - self.registered_at < self.grace_period_seconds:
            if self.total_requests == 0:
                return 0.9  # 90% assumed success for new nodes
            # Blend actual and assumed for smooth transition
            actual_rate = self.successful_requests / self.total_requests if self.total_requests > 0 else 0
            grace_weight = 1 - ((current_time - self.registered_at) / self.grace_period_seconds)
            return actual_rate * (1 - grace_weight) + 0.9 * grace_weight
        
        # Check if window needs reset
        if current_time - self.window_start_time > self.window_duration_seconds:
            # Window expired, use overall metrics
            if self.total_requests == 0:
                return 1.0
            return self.successful_requests / self.total_requests
        
        # Use windowed metrics for recent performance
        if self.window_requests == 0:
            # No recent requests, fall back to overall
            if self.total_requests == 0:
                return 1.0
            return self.successful_requests / self.total_requests
        
        # Weight recent performance more heavily (70% recent, 30% historical)
        recent_rate = self.window_successes / self.window_requests
        if self.total_requests > self.window_requests:
            historical_rate = ((self.successful_requests - self.window_successes) / 
                             (self.total_requests - self.window_requests))
            return 0.7 * recent_rate + 0.3 * historical_rate
        return recent_rate
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def p95_response_time(self) -> float:
        """Calculate 95th percentile response time."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    def record_success(self, response_time: float):
        """Record a successful request."""
        current_time = time.time()
        
        # Check if window needs reset
        if current_time - self.window_start_time > self.window_duration_seconds:
            self.window_start_time = current_time
            self.window_requests = 0
            self.window_successes = 0
        
        self.total_requests += 1
        self.successful_requests += 1
        self.window_requests += 1
        self.window_successes += 1
        self.total_response_time += response_time
        self.response_times.append(response_time)
        self.consecutive_failures = 0
        
        # Update reputation (slowly increase on success)
        # Faster recovery during grace period
        if current_time - self.registered_at < self.grace_period_seconds:
            self.reputation_score = min(100, self.reputation_score + 1.0)  # Faster recovery for new nodes
        else:
            self.reputation_score = min(100, self.reputation_score + 0.5)
        
    def record_failure(self):
        """Record a failed request."""
        current_time = time.time()
        
        # Check if window needs reset
        if current_time - self.window_start_time > self.window_duration_seconds:
            self.window_start_time = current_time
            self.window_requests = 0
            self.window_successes = 0
        
        self.total_requests += 1
        self.failed_requests += 1
        self.window_requests += 1
        self.consecutive_failures += 1
        self.last_failure = current_time
        
        # Update reputation (decrease on failure)
        # Less penalty during grace period
        if current_time - self.registered_at < self.grace_period_seconds:
            penalty = min(5, self.consecutive_failures)  # Gentler penalty for new nodes
        else:
            penalty = min(10, self.consecutive_failures * 2)
        self.reputation_score = max(0, self.reputation_score - penalty)
        
        # Mark unhealthy after too many failures (more lenient for new nodes)
        failure_threshold = 5 if current_time - self.registered_at < self.grace_period_seconds else 3
        if self.consecutive_failures >= failure_threshold:
            self.is_healthy = False


class NodeReputationManager:
    """Manages node reputation and failover with background health monitoring."""
    
    def __init__(self, health_check_interval: int = 30, persistence_path: Optional[str] = None):
        self.node_metrics: Dict[str, NodeMetrics] = {}
        self.blacklist: Dict[str, float] = {}  # node_id -> blacklist_until
        self.health_check_interval = health_check_interval  # seconds
        self.blacklist_duration_base = 60  # base blacklist duration in seconds
        
        # Concurrency protection
        self._metrics_lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Persistence
        self.persistence_path = persistence_path or "./data/node_reputation.json"
        self._load_metrics()
        
        # Node endpoints for health checks
        self.node_endpoints: Dict[str, str] = {}  # node_id -> endpoint
        
    def get_or_create_metrics(self, node_id: str) -> NodeMetrics:
        """Get or create metrics for a node."""
        if node_id not in self.node_metrics:
            self.node_metrics[node_id] = NodeMetrics(node_id=node_id)
        return self.node_metrics[node_id]
    
    def record_request_success(self, node_id: str, response_time: float):
        """Record a successful request to a node."""
        metrics = self.get_or_create_metrics(node_id)
        metrics.record_success(response_time)
        logger.debug(f"Node {node_id} success: {response_time:.2f}s, reputation: {metrics.reputation_score:.1f}")
    
    def record_request_failure(self, node_id: str):
        """Record a failed request to a node."""
        metrics = self.get_or_create_metrics(node_id)
        metrics.record_failure()
        
        # Apply exponential backoff blacklisting
        if metrics.consecutive_failures >= 3:
            backoff_duration = self.blacklist_duration_base * (2 ** (metrics.consecutive_failures - 3))
            self.blacklist_node(node_id, backoff_duration)
        
        logger.warning(f"Node {node_id} failure #{metrics.consecutive_failures}, reputation: {metrics.reputation_score:.1f}")
    
    def blacklist_node(self, node_id: str, duration: float):
        """Temporarily blacklist a node."""
        self.blacklist[node_id] = time.time() + duration
        logger.warning(f"Blacklisted node {node_id} for {duration:.0f}s")
    
    def is_node_available(self, node_id: str) -> bool:
        """Check if a node is available (not blacklisted)."""
        if node_id in self.blacklist:
            if time.time() < self.blacklist[node_id]:
                return False
            else:
                # Blacklist expired
                del self.blacklist[node_id]
                logger.info(f"Node {node_id} removed from blacklist")
        return True
    
    def get_node_reputation(self, node_id: str) -> float:
        """Get reputation score for a node."""
        if node_id in self.node_metrics:
            return self.node_metrics[node_id].reputation_score
        return 50.0  # Default reputation for new nodes
    
    def rank_nodes(self, node_ids: List[str]) -> List[Tuple[str, float]]:
        """Rank nodes by reputation and availability."""
        available_nodes = []
        
        for node_id in node_ids:
            if not self.is_node_available(node_id):
                continue
                
            reputation = self.get_node_reputation(node_id)
            metrics = self.node_metrics.get(node_id)
            
            # Calculate composite score
            if metrics:
                # Factor in success rate and response time
                score = (
                    reputation * 0.5 +
                    metrics.success_rate * 30 +
                    (1.0 / (1.0 + metrics.avg_response_time)) * 20
                )
            else:
                score = reputation
                
            available_nodes.append((node_id, score))
        
        # Sort by score (highest first)
        available_nodes.sort(key=lambda x: x[1], reverse=True)
        return available_nodes
    
    async def health_check(self, node_id: str, endpoint: str) -> bool:
        """Perform health check on a node."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{endpoint}/health", timeout=5) as response:
                    if response.status == 200:
                        metrics = self.get_or_create_metrics(node_id)
                        metrics.is_healthy = True
                        metrics.last_health_check = time.time()
                        
                        # Slowly recover reputation on successful health check
                        if metrics.reputation_score < 50:
                            metrics.reputation_score = min(50, metrics.reputation_score + 5)
                        
                        return True
        except Exception as e:
            logger.error(f"Health check failed for node {node_id}: {e}")
        
        metrics = self.get_or_create_metrics(node_id)
        metrics.is_healthy = False
        metrics.last_health_check = time.time()
        return False
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all node metrics."""
        summary = {}
        
        for node_id, metrics in self.node_metrics.items():
            summary[node_id] = {
                "reputation": metrics.reputation_score,
                "success_rate": f"{metrics.success_rate:.2%}",
                "avg_response_time": f"{metrics.avg_response_time:.2f}s",
                "p95_response_time": f"{metrics.p95_response_time:.2f}s",
                "total_requests": metrics.total_requests,
                "consecutive_failures": metrics.consecutive_failures,
                "is_healthy": metrics.is_healthy,
                "is_blacklisted": node_id in self.blacklist
            }
        
        return summary
    
    async def start_health_monitoring(self):
        """Start background health check task."""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Started background health monitoring")
        return self._health_check_task
    
    async def stop_health_monitoring(self):
        """Stop background health check task."""
        self._shutdown = True
        if self._health_check_task:
            await self._health_check_task
            self._health_check_task = None
        logger.info("Stopped background health monitoring")
    
    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while not self._shutdown:
            try:
                await self._perform_health_checks()
                self._save_metrics()  # Persist after each check cycle
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _perform_health_checks(self):
        """Perform health checks on all registered nodes."""
        async with self._metrics_lock:
            nodes_to_check = list(self.node_endpoints.keys())
        
        if not nodes_to_check:
            return
        
        logger.debug(f"Performing health checks on {len(nodes_to_check)} nodes")
        
        # Check nodes concurrently
        tasks = []
        for node_id in nodes_to_check:
            if node_id in self.node_endpoints:
                tasks.append(self._check_node_health(node_id, self.node_endpoints[node_id]))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_node_health(self, node_id: str, endpoint: str):
        """Check health of a single node with p95 tracking."""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{endpoint}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        async with self._metrics_lock:
                            metrics = self.get_or_create_metrics(node_id)
                            metrics.is_healthy = True
                            metrics.last_health_check = time.time()
                            # Record as successful interaction
                            metrics.record_success(response_time)
                        logger.debug(f"Node {node_id} healthy: {response_time:.2f}s")
                    else:
                        raise Exception(f"Health check returned {response.status}")
                        
        except Exception as e:
            async with self._metrics_lock:
                metrics = self.get_or_create_metrics(node_id)
                metrics.is_healthy = False
                metrics.last_health_check = time.time()
                metrics.record_failure()
            logger.warning(f"Node {node_id} health check failed: {e}")
    
    def register_node_endpoint(self, node_id: str, endpoint: str):
        """Register a node endpoint for health monitoring."""
        self.node_endpoints[node_id] = endpoint
        logger.info(f"Registered node {node_id} at {endpoint} for health monitoring")
    
    def _save_metrics(self):
        """Persist metrics to disk."""
        import json
        from pathlib import Path
        
        try:
            Path(self.persistence_path).parent.mkdir(parents=True, exist_ok=True)
            
            data = {}
            for node_id, metrics in self.node_metrics.items():
                data[node_id] = {
                    'reputation_score': metrics.reputation_score,
                    'total_requests': metrics.total_requests,
                    'successful_requests': metrics.successful_requests,
                    'failed_requests': metrics.failed_requests,
                    'is_healthy': metrics.is_healthy,
                    'last_health_check': metrics.last_health_check,
                    'response_times': list(metrics.response_times)[-10:]  # Keep last 10
                }
            
            with open(self.persistence_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def _load_metrics(self):
        """Load metrics from disk."""
        import json
        from pathlib import Path
        
        try:
            if Path(self.persistence_path).exists():
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                    
                for node_id, metrics_data in data.items():
                    metrics = NodeMetrics(node_id=node_id)
                    metrics.reputation_score = metrics_data.get('reputation_score', 50.0)
                    metrics.total_requests = metrics_data.get('total_requests', 0)
                    metrics.successful_requests = metrics_data.get('successful_requests', 0)
                    metrics.failed_requests = metrics_data.get('failed_requests', 0)
                    metrics.is_healthy = metrics_data.get('is_healthy', True)
                    metrics.last_health_check = metrics_data.get('last_health_check', 0)
                    
                    # Restore response times
                    saved_times = metrics_data.get('response_times', [])
                    for rt in saved_times:
                        metrics.response_times.append(rt)
                    
                    self.node_metrics[node_id] = metrics
                    
                logger.info(f"Loaded metrics for {len(self.node_metrics)} nodes")
                
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")


class FailoverCoordinator:
    """Coordinates failover between nodes."""
    
    def __init__(self, reputation_manager: NodeReputationManager):
        self.reputation_manager = reputation_manager
        self.retry_attempts = 3
        self.retry_delay = 1.0  # seconds
        
    async def execute_with_failover(
        self,
        primary_node_id: str,
        fallback_node_ids: List[str],
        execute_fn: Any,
        *args,
        **kwargs
    ) -> Tuple[bool, Any, str]:
        """
        Execute a function with automatic failover.
        
        Returns:
            - success: Whether execution succeeded
            - result: The result if successful
            - node_id: The node that successfully executed
        """
        all_nodes = [primary_node_id] + fallback_node_ids
        
        for attempt in range(len(all_nodes)):
            # Rank nodes by reputation
            ranked_nodes = self.reputation_manager.rank_nodes(all_nodes[attempt:])
            
            if not ranked_nodes:
                logger.error("No available nodes for failover")
                break
            
            # Try the best available node
            node_id, score = ranked_nodes[0]
            logger.info(f"Attempting node {node_id} (score: {score:.1f})")
            
            try:
                start_time = time.time()
                result = await execute_fn(node_id, *args, **kwargs)
                response_time = time.time() - start_time
                
                # Record success
                self.reputation_manager.record_request_success(node_id, response_time)
                return True, result, node_id
                
            except Exception as e:
                logger.error(f"Node {node_id} failed: {e}")
                self.reputation_manager.record_request_failure(node_id)
                
                # Wait before trying next node
                if attempt < len(all_nodes) - 1:
                    await asyncio.sleep(self.retry_delay)
        
        return False, None, None
    
    async def execute_with_retry(
        self,
        node_id: str,
        execute_fn: Any,
        *args,
        **kwargs
    ) -> Tuple[bool, Any]:
        """
        Execute a function with retry logic.
        """
        for attempt in range(self.retry_attempts):
            try:
                start_time = time.time()
                result = await execute_fn(node_id, *args, **kwargs)
                response_time = time.time() - start_time
                
                # Record success
                self.reputation_manager.record_request_success(node_id, response_time)
                return True, result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.retry_attempts} failed for node {node_id}: {e}")
                
                if attempt < self.retry_attempts - 1:
                    # Exponential backoff
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    # Final failure
                    self.reputation_manager.record_request_failure(node_id)
        
        return False, None


# Global instances
_reputation_manager = None
_failover_coordinator = None


def get_reputation_manager() -> NodeReputationManager:
    """Get or create reputation manager singleton."""
    global _reputation_manager
    if _reputation_manager is None:
        _reputation_manager = NodeReputationManager()
    return _reputation_manager


def get_failover_coordinator() -> FailoverCoordinator:
    """Get or create failover coordinator singleton."""
    global _failover_coordinator
    if _failover_coordinator is None:
        _failover_coordinator = FailoverCoordinator(get_reputation_manager())
    return _failover_coordinator