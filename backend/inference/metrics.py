"""
Unified Inference Metrics
=========================
Single source of truth for inference performance tracking.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Unified metrics for all inference operations."""
    
    request_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Token metrics
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_tokens: int = 0
    
    # Performance metrics
    latency_ms: float = 0.0
    time_to_first_token_ms: Optional[float] = None
    tokens_per_second: float = 0.0
    
    # Resource metrics
    gpu_utilization: Optional[float] = None
    memory_used_gb: Optional[float] = None
    
    # Pipeline metrics (for distributed)
    pipeline_stages: Optional[Dict[int, str]] = None
    nodes_used: Optional[List[str]] = None
    
    # Quality metrics
    success: bool = False
    error: Optional[str] = None
    cache_hit: bool = False
    
    # Cost metrics
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    
    def finalize(self) -> None:
        """Calculate final metrics."""
        self.end_time = time.time()
        self.latency_ms = (self.end_time - self.start_time) * 1000
        self.total_tokens = self.prompt_tokens + self.generated_tokens
        
        if self.generated_tokens > 0 and self.latency_ms > 0:
            self.tokens_per_second = self.generated_tokens / (self.latency_ms / 1000)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_second": self.tokens_per_second,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "success": self.success,
            "cache_hit": self.cache_hit,
            "error": self.error,
            "gpu_utilization": self.gpu_utilization,
            "memory_used_gb": self.memory_used_gb,
            "pipeline_stages": self.pipeline_stages,
            "nodes_used": self.nodes_used,
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost
        }


class MetricsCollector:
    """Centralized metrics collection and aggregation."""
    
    def __init__(self, max_history: int = 1000):
        self.history: List[InferenceMetrics] = []
        self.max_history = max_history
        
    def record(self, metrics: InferenceMetrics) -> None:
        """Record metrics instance."""
        self.history.append(metrics)
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated metrics summary."""
        if not self.history:
            return {"message": "No metrics available"}
        
        successful = [m for m in self.history if m.success]
        failed = [m for m in self.history if not m.success]
        
        if successful:
            avg_latency = sum(m.latency_ms for m in successful) / len(successful)
            avg_tokens = sum(m.generated_tokens for m in successful) / len(successful)
            avg_tps = sum(m.tokens_per_second for m in successful if m.tokens_per_second > 0) / max(1, len([m for m in successful if m.tokens_per_second > 0]))
            cache_hit_rate = sum(1 for m in successful if m.cache_hit) / len(successful)
        else:
            avg_latency = 0
            avg_tokens = 0
            avg_tps = 0
            cache_hit_rate = 0
        
        return {
            "total_requests": len(self.history),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / len(self.history) if self.history else 0,
            "avg_latency_ms": avg_latency,
            "avg_tokens_generated": avg_tokens,
            "avg_tokens_per_second": avg_tps,
            "cache_hit_rate": cache_hit_rate,
            "total_tokens_generated": sum(m.generated_tokens for m in successful),
            "total_cost": sum(m.actual_cost for m in successful)
        }
    
    def get_recent(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent metrics."""
        recent = self.history[-count:] if len(self.history) > count else self.history
        return [m.to_dict() for m in recent]


# Global metrics collector instance
_global_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    return _global_collector


def create_metrics(request_id: str, prompt: str = "") -> InferenceMetrics:
    """
    Create new metrics instance.
    
    Args:
        request_id: Unique request identifier
        prompt: Optional prompt for token counting
        
    Returns:
        New InferenceMetrics instance
    """
    metrics = InferenceMetrics(request_id=request_id)
    
    if prompt:
        from backend.common.costs import TokenCostCalculator
        metrics.prompt_tokens = TokenCostCalculator.count_tokens(prompt)
    
    return metrics