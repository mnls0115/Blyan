"""
Unified Metrics Collection

Consolidates all metrics collection using the block runtime metrics system.
This replaces scattered metrics implementations across the codebase.
"""

from typing import Dict, Any, Optional, List
import time
from dataclasses import dataclass, field
from collections import defaultdict
import threading

from backend.runtime.block.metrics import MetricsCollector, MetricWindow


@dataclass
class UnifiedMetrics:
    """
    Unified metrics that aggregates from all components.
    
    This replaces scattered metrics collection with a single source of truth.
    """
    
    # Block runtime metrics
    runtime_collector: MetricsCollector
    
    # Component-specific collectors
    inference_metrics: Dict[str, Any] = field(default_factory=dict)
    cache_metrics: Dict[str, Any] = field(default_factory=dict)
    p2p_metrics: Dict[str, Any] = field(default_factory=dict)
    blockchain_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Aggregated windows
    request_latencies: MetricWindow = field(default_factory=lambda: MetricWindow(window_size=1000))
    token_latencies: MetricWindow = field(default_factory=lambda: MetricWindow(window_size=1000))
    
    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def __post_init__(self):
        if self.runtime_collector is None:
            self.runtime_collector = MetricsCollector(enable_detailed=True)
    
    def record_inference_request(
        self,
        session_id: str,
        model_id: str,
        num_experts: int,
        num_tokens: int,
        latency_ms: float,
        cache_hits: int = 0,
        cache_misses: int = 0
    ):
        """Record metrics for an inference request."""
        with self._lock:
            # Update runtime collector
            self.runtime_collector.increment("inference_requests")
            self.runtime_collector.increment("tokens_generated", num_tokens)
            self.runtime_collector.increment("expert_loads", num_experts)
            self.runtime_collector.increment("cache_hits", cache_hits)
            self.runtime_collector.increment("cache_misses", cache_misses)
            self.runtime_collector.record_latency("inference_latency_ms", latency_ms)
            
            # Update aggregated windows
            self.request_latencies.add(latency_ms)
            
            # Update inference metrics
            if model_id not in self.inference_metrics:
                self.inference_metrics[model_id] = {
                    "requests": 0,
                    "tokens": 0,
                    "total_latency_ms": 0,
                    "cache_hits": 0,
                    "cache_misses": 0
                }
            
            self.inference_metrics[model_id]["requests"] += 1
            self.inference_metrics[model_id]["tokens"] += num_tokens
            self.inference_metrics[model_id]["total_latency_ms"] += latency_ms
            self.inference_metrics[model_id]["cache_hits"] += cache_hits
            self.inference_metrics[model_id]["cache_misses"] += cache_misses
    
    def record_token_generation(
        self,
        token_id: int,
        latency_ms: float,
        expert_usage: Dict[str, float]
    ):
        """Record metrics for single token generation."""
        with self._lock:
            self.runtime_collector.record_latency("token_generation_ms", latency_ms)
            self.token_latencies.add(latency_ms)
            
            # Track expert usage
            for expert_id, usage in expert_usage.items():
                self.runtime_collector.increment(f"expert_usage_{expert_id}", int(usage * 100))
    
    def record_cache_operation(
        self,
        operation: str,  # "hit", "miss", "eviction", "prefetch"
        expert_id: str,
        latency_ms: float = 0
    ):
        """Record cache operation metrics."""
        with self._lock:
            if operation == "hit":
                self.runtime_collector.increment("cache_hits")
            elif operation == "miss":
                self.runtime_collector.increment("cache_misses")
            elif operation == "eviction":
                self.runtime_collector.increment("cache_evictions")
            elif operation == "prefetch":
                self.runtime_collector.increment("cache_prefetches")
            
            if latency_ms > 0:
                self.runtime_collector.record_latency(f"cache_{operation}_latency_ms", latency_ms)
            
            # Update cache metrics
            if expert_id not in self.cache_metrics:
                self.cache_metrics[expert_id] = defaultdict(int)
            self.cache_metrics[expert_id][operation] += 1
    
    def record_p2p_operation(
        self,
        operation: str,  # "register", "unregister", "heartbeat", "inference"
        node_id: str,
        success: bool,
        latency_ms: float = 0
    ):
        """Record P2P operation metrics."""
        with self._lock:
            metric_name = f"p2p_{operation}_{'success' if success else 'failure'}"
            self.runtime_collector.increment(metric_name)
            
            if latency_ms > 0:
                self.runtime_collector.record_latency(f"p2p_{operation}_latency_ms", latency_ms)
            
            # Update P2P metrics
            if node_id not in self.p2p_metrics:
                self.p2p_metrics[node_id] = {
                    "operations": defaultdict(int),
                    "successes": 0,
                    "failures": 0,
                    "total_latency_ms": 0
                }
            
            self.p2p_metrics[node_id]["operations"][operation] += 1
            if success:
                self.p2p_metrics[node_id]["successes"] += 1
            else:
                self.p2p_metrics[node_id]["failures"] += 1
            self.p2p_metrics[node_id]["total_latency_ms"] += latency_ms
    
    def record_blockchain_operation(
        self,
        operation: str,  # "read", "write", "verify"
        chain_id: str,
        success: bool,
        latency_ms: float = 0,
        gas_used: Optional[int] = None
    ):
        """Record blockchain operation metrics."""
        with self._lock:
            metric_name = f"blockchain_{operation}_{'success' if success else 'failure'}"
            self.runtime_collector.increment(metric_name)
            
            if latency_ms > 0:
                self.runtime_collector.record_latency(f"blockchain_{operation}_latency_ms", latency_ms)
            
            # Update blockchain metrics
            if chain_id not in self.blockchain_metrics:
                self.blockchain_metrics[chain_id] = {
                    "operations": defaultdict(int),
                    "successes": 0,
                    "failures": 0,
                    "total_latency_ms": 0,
                    "total_gas": 0
                }
            
            self.blockchain_metrics[chain_id]["operations"][operation] += 1
            if success:
                self.blockchain_metrics[chain_id]["successes"] += 1
            else:
                self.blockchain_metrics[chain_id]["failures"] += 1
            self.blockchain_metrics[chain_id]["total_latency_ms"] += latency_ms
            
            if gas_used:
                self.blockchain_metrics[chain_id]["total_gas"] += gas_used
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get comprehensive aggregated metrics."""
        with self._lock:
            # Get runtime metrics
            runtime_metrics = self.runtime_collector.get_metrics()
            
            # Calculate inference aggregates
            total_inference_requests = sum(
                m["requests"] for m in self.inference_metrics.values()
            )
            total_tokens = sum(
                m["tokens"] for m in self.inference_metrics.values()
            )
            
            # Calculate cache performance
            total_cache_ops = defaultdict(int)
            for expert_metrics in self.cache_metrics.values():
                for op, count in expert_metrics.items():
                    total_cache_ops[op] += count
            
            cache_hit_ratio = (
                total_cache_ops["hit"] / (total_cache_ops["hit"] + total_cache_ops["miss"])
                if (total_cache_ops["hit"] + total_cache_ops["miss"]) > 0
                else 0
            )
            
            # Calculate P2P health
            total_p2p_success = sum(m["successes"] for m in self.p2p_metrics.values())
            total_p2p_failure = sum(m["failures"] for m in self.p2p_metrics.values())
            p2p_success_rate = (
                total_p2p_success / (total_p2p_success + total_p2p_failure)
                if (total_p2p_success + total_p2p_failure) > 0
                else 0
            )
            
            # Build aggregated response
            return {
                "runtime": runtime_metrics,
                "inference": {
                    "total_requests": total_inference_requests,
                    "total_tokens": total_tokens,
                    "models": self.inference_metrics,
                    "request_latency_stats": self.request_latencies.get_stats(),
                    "token_latency_stats": self.token_latencies.get_stats()
                },
                "cache": {
                    "hit_ratio": cache_hit_ratio,
                    "operations": dict(total_cache_ops),
                    "by_expert": self.cache_metrics
                },
                "p2p": {
                    "success_rate": p2p_success_rate,
                    "total_successes": total_p2p_success,
                    "total_failures": total_p2p_failure,
                    "by_node": self.p2p_metrics
                },
                "blockchain": self.blockchain_metrics,
                "timestamp": time.time()
            }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        metrics = self.get_aggregated_metrics()
        
        # Runtime metrics (already formatted)
        lines.append(self.runtime_collector.export_prometheus())
        
        # Inference metrics
        lines.append(f"# Inference Metrics")
        lines.append(f"inference_total_requests {metrics['inference']['total_requests']}")
        lines.append(f"inference_total_tokens {metrics['inference']['total_tokens']}")
        
        if metrics['inference']['request_latency_stats']['count'] > 0:
            stats = metrics['inference']['request_latency_stats']
            lines.append(f"inference_request_latency_p50 {stats['p50']}")
            lines.append(f"inference_request_latency_p95 {stats['p95']}")
            lines.append(f"inference_request_latency_p99 {stats['p99']}")
        
        # Cache metrics
        lines.append(f"# Cache Metrics")
        lines.append(f"cache_hit_ratio {metrics['cache']['hit_ratio']}")
        for op, count in metrics['cache']['operations'].items():
            lines.append(f"cache_operation_{op}_total {count}")
        
        # P2P metrics
        lines.append(f"# P2P Metrics")
        lines.append(f"p2p_success_rate {metrics['p2p']['success_rate']}")
        lines.append(f"p2p_total_successes {metrics['p2p']['total_successes']}")
        lines.append(f"p2p_total_failures {metrics['p2p']['total_failures']}")
        
        # Blockchain metrics
        lines.append(f"# Blockchain Metrics")
        for chain_id, chain_metrics in metrics['blockchain'].items():
            lines.append(f"blockchain_{chain_id}_successes {chain_metrics['successes']}")
            lines.append(f"blockchain_{chain_id}_failures {chain_metrics['failures']}")
            if chain_metrics['total_gas'] > 0:
                lines.append(f"blockchain_{chain_id}_gas_used {chain_metrics['total_gas']}")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.runtime_collector.reset()
            self.inference_metrics.clear()
            self.cache_metrics.clear()
            self.p2p_metrics.clear()
            self.blockchain_metrics.clear()
            self.request_latencies.values.clear()
            self.token_latencies.values.clear()


# Global metrics instance
_global_metrics: Optional[UnifiedMetrics] = None


def get_unified_metrics() -> UnifiedMetrics:
    """Get or create global unified metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = UnifiedMetrics(
            runtime_collector=MetricsCollector(enable_detailed=True)
        )
    return _global_metrics


# Convenience functions for common operations
def record_inference(session_id: str, model_id: str, **kwargs):
    """Record inference metrics."""
    get_unified_metrics().record_inference_request(session_id, model_id, **kwargs)


def record_cache(operation: str, expert_id: str, **kwargs):
    """Record cache metrics."""
    get_unified_metrics().record_cache_operation(operation, expert_id, **kwargs)


def record_p2p(operation: str, node_id: str, **kwargs):
    """Record P2P metrics."""
    get_unified_metrics().record_p2p_operation(operation, node_id, **kwargs)


def record_blockchain(operation: str, chain_id: str, **kwargs):
    """Record blockchain metrics."""
    get_unified_metrics().record_blockchain_operation(operation, chain_id, **kwargs)


def get_metrics() -> Dict[str, Any]:
    """Get all metrics."""
    return get_unified_metrics().get_aggregated_metrics()


def export_prometheus() -> str:
    """Export metrics in Prometheus format."""
    return get_unified_metrics().export_prometheus()