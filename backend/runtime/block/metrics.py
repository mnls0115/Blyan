"""
Metrics Layer Implementation

Unified metrics collection and reporting for block runtime.
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Point-in-time metric snapshot."""
    timestamp: float
    value: float


@dataclass
class MetricWindow:
    """Sliding window for metric aggregation."""
    window_size: int = 100
    values: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add(self, value: float) -> None:
        """Add a value to the window."""
        self.values.append(MetricSnapshot(time.time(), value))
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistical summary of the window."""
        if not self.values:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
        
        values = [s.value for s in self.values]
        values.sort()
        n = len(values)
        
        return {
            "count": n,
            "mean": sum(values) / n,
            "min": values[0],
            "max": values[-1],
            "p50": values[int(n * 0.5)],
            "p95": values[int(n * 0.95)] if n > 20 else values[-1],
            "p99": values[int(n * 0.99)] if n > 100 else values[-1]
        }


class MetricsCollector:
    """
    Centralized metrics collection for block runtime.
    """
    
    def __init__(self, enable_detailed: bool = True):
        self.enable_detailed = enable_detailed
        self._lock = threading.Lock()
        
        # Core metrics
        self.counters: Dict[str, int] = {
            "inference_requests": 0,
            "tokens_generated": 0,
            "expert_loads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "verification_successes": 0,
            "verification_failures": 0,
            "stream_tokens": 0,
            "cancelled_sessions": 0
        }
        
        # Latency metrics (sliding windows)
        self.latencies: Dict[str, MetricWindow] = {
            "first_token_latency_ms": MetricWindow(),
            "token_generation_ms": MetricWindow(),
            "expert_fetch_ms": MetricWindow(),
            "verification_ms": MetricWindow(),
            "streaming_latency_ms": MetricWindow()
        }
        
        # Throughput metrics
        self.throughput: Dict[str, MetricWindow] = {
            "tokens_per_second": MetricWindow(),
            "requests_per_second": MetricWindow()
        }
        
        # Resource metrics
        self.resources: Dict[str, float] = {
            "memory_cache_mb": 0,
            "disk_cache_mb": 0,
            "active_sessions": 0,
            "queue_depth": 0
        }
        
        # Error tracking
        self.errors: Dict[str, int] = {}
        
        # Start time for rate calculations
        self.start_time = time.time()
        self.last_reset = time.time()
    
    def increment(self, metric: str, value: int = 1) -> None:
        """Increment a counter metric."""
        with self._lock:
            if metric not in self.counters:
                self.counters[metric] = 0
            self.counters[metric] += value
    
    def record_latency(self, metric: str, latency_ms: float) -> None:
        """Record a latency measurement."""
        with self._lock:
            if metric not in self.latencies:
                self.latencies[metric] = MetricWindow()
            self.latencies[metric].add(latency_ms)
    
    def record_throughput(self, metric: str, value: float) -> None:
        """Record a throughput measurement."""
        with self._lock:
            if metric not in self.throughput:
                self.throughput[metric] = MetricWindow()
            self.throughput[metric].add(value)
    
    def update_resource(self, metric: str, value: float) -> None:
        """Update a resource metric."""
        with self._lock:
            self.resources[metric] = value
    
    def record_error(self, error_type: str) -> None:
        """Record an error occurrence."""
        with self._lock:
            if error_type not in self.errors:
                self.errors[error_type] = 0
            self.errors[error_type] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            uptime_seconds = time.time() - self.start_time
            
            # Calculate rates
            tokens_per_second = (
                self.counters["tokens_generated"] / uptime_seconds
                if uptime_seconds > 0 else 0
            )
            
            requests_per_second = (
                self.counters["inference_requests"] / uptime_seconds
                if uptime_seconds > 0 else 0
            )
            
            # Calculate cache hit ratio
            total_cache_ops = self.counters["cache_hits"] + self.counters["cache_misses"]
            cache_hit_ratio = (
                self.counters["cache_hits"] / total_cache_ops
                if total_cache_ops > 0 else 0
            )
            
            # Build metrics dict
            metrics = {
                "uptime_seconds": uptime_seconds,
                "counters": dict(self.counters),
                "rates": {
                    "tokens_per_second": tokens_per_second,
                    "requests_per_second": requests_per_second
                },
                "cache": {
                    "hit_ratio": cache_hit_ratio,
                    "hits": self.counters["cache_hits"],
                    "misses": self.counters["cache_misses"]
                },
                "resources": dict(self.resources),
                "errors": dict(self.errors)
            }
            
            # Add detailed latency stats if enabled
            if self.enable_detailed:
                metrics["latencies"] = {
                    name: window.get_stats()
                    for name, window in self.latencies.items()
                }
                
                metrics["throughput"] = {
                    name: window.get_stats()
                    for name, window in self.throughput.items()
                }
            
            # Add SLO compliance
            metrics["slo"] = self._calculate_slo_compliance()
            
            return metrics
    
    def _calculate_slo_compliance(self) -> Dict[str, Any]:
        """Calculate SLO compliance metrics."""
        # Define SLO targets
        slo_targets = {
            "first_token_p95_ms": 500,  # 500ms p95 first token
            "tokens_per_second": 10,    # 10 tokens/sec minimum
            "cache_hit_ratio": 0.8,     # 80% cache hit ratio
            "error_rate": 0.01          # 1% error rate max
        }
        
        # Calculate compliance
        compliance = {}
        
        # First token latency
        if "first_token_latency_ms" in self.latencies:
            stats = self.latencies["first_token_latency_ms"].get_stats()
            if stats["count"] > 0:
                compliance["first_token_p95"] = {
                    "target": slo_targets["first_token_p95_ms"],
                    "actual": stats["p95"],
                    "compliant": stats["p95"] <= slo_targets["first_token_p95_ms"]
                }
        
        # Tokens per second
        uptime = time.time() - self.start_time
        if uptime > 0:
            tps = self.counters["tokens_generated"] / uptime
            compliance["tokens_per_second"] = {
                "target": slo_targets["tokens_per_second"],
                "actual": tps,
                "compliant": tps >= slo_targets["tokens_per_second"]
            }
        
        # Cache hit ratio
        total_cache = self.counters["cache_hits"] + self.counters["cache_misses"]
        if total_cache > 0:
            hit_ratio = self.counters["cache_hits"] / total_cache
            compliance["cache_hit_ratio"] = {
                "target": slo_targets["cache_hit_ratio"],
                "actual": hit_ratio,
                "compliant": hit_ratio >= slo_targets["cache_hit_ratio"]
            }
        
        # Error rate
        total_requests = self.counters["inference_requests"]
        if total_requests > 0:
            total_errors = sum(self.errors.values())
            error_rate = total_errors / total_requests
            compliance["error_rate"] = {
                "target": slo_targets["error_rate"],
                "actual": error_rate,
                "compliant": error_rate <= slo_targets["error_rate"]
            }
        
        # Overall compliance
        if compliance:
            compliant_count = sum(1 for c in compliance.values() if c.get("compliant", False))
            compliance["overall"] = {
                "compliant_metrics": compliant_count,
                "total_metrics": len(compliance),
                "compliance_rate": compliant_count / len(compliance)
            }
        
        return compliance
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            for key in self.counters:
                self.counters[key] = 0
            
            for window in self.latencies.values():
                window.values.clear()
            
            for window in self.throughput.values():
                window.values.clear()
            
            self.errors.clear()
            self.last_reset = time.time()
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = self.get_metrics()
        lines = []
        
        # Counters
        for name, value in metrics["counters"].items():
            lines.append(f"block_runtime_{name}_total {value}")
        
        # Rates
        for name, value in metrics["rates"].items():
            lines.append(f"block_runtime_{name} {value}")
        
        # Cache metrics
        lines.append(f"block_runtime_cache_hit_ratio {metrics['cache']['hit_ratio']}")
        
        # Resources
        for name, value in metrics["resources"].items():
            lines.append(f"block_runtime_{name} {value}")
        
        # Latency percentiles
        if "latencies" in metrics:
            for metric_name, stats in metrics["latencies"].items():
                if stats["count"] > 0:
                    lines.append(f"block_runtime_{metric_name}_p50 {stats['p50']}")
                    lines.append(f"block_runtime_{metric_name}_p95 {stats['p95']}")
                    lines.append(f"block_runtime_{metric_name}_p99 {stats['p99']}")
        
        # SLO compliance
        if "slo" in metrics and "overall" in metrics["slo"]:
            overall = metrics["slo"]["overall"]
            lines.append(f"block_runtime_slo_compliance_rate {overall['compliance_rate']}")
        
        return "\n".join(lines)