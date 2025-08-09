#!/usr/bin/env python3
"""
Performance Metrics Dashboard
Tracks p95/p99 latency, throughput, and GPU utilization
"""

import time
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    timestamp: float
    request_id: str
    latency_ms: float
    tokens_generated: int
    gpu_memory_mb: float
    gpu_utilization: float
    model_type: str  # 'main', 'teacher', 'sentinel'
    batch_size: int
    kv_cache_hit: bool
    experts_used: int

class PerformanceTracker:
    """
    Tracks performance metrics with sliding window.
    """
    
    def __init__(self, window_size: int = 1000, persist_path: Optional[Path] = None):
        self.window_size = window_size
        self.persist_path = persist_path or Path("./logs/performance_metrics.json")
        
        # Metrics storage (sliding window)
        self.metrics: deque = deque(maxlen=window_size)
        
        # Aggregated stats
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.time()
        
        # Load historical data if exists
        self._load_historical()
    
    def record_request(
        self,
        request_id: str,
        latency_ms: float,
        tokens_generated: int,
        model_type: str = 'main',
        batch_size: int = 1,
        kv_cache_hit: bool = False,
        experts_used: int = 0
    ):
        """Record a single request's performance."""
        
        # Get GPU metrics if available
        gpu_memory_mb = 0
        gpu_utilization = 0
        
        if torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / 1e6
                # GPU utilization would require nvidia-ml-py
                gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            except:
                pass
        
        metric = PerformanceMetric(
            timestamp=time.time(),
            request_id=request_id,
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization,
            model_type=model_type,
            batch_size=batch_size,
            kv_cache_hit=kv_cache_hit,
            experts_used=experts_used
        )
        
        self.metrics.append(metric)
        self.total_requests += 1
        self.total_tokens += tokens_generated
        
        # Persist periodically
        if self.total_requests % 100 == 0:
            self._persist_metrics()
    
    def get_percentiles(self, percentiles: List[int] = [50, 90, 95, 99]) -> Dict[str, float]:
        """Get latency percentiles."""
        if not self.metrics:
            return {f"p{p}": 0 for p in percentiles}
        
        latencies = [m.latency_ms for m in self.metrics]
        
        result = {}
        for p in percentiles:
            result[f"p{p}"] = np.percentile(latencies, p)
        
        return result
    
    def get_throughput(self, window_seconds: int = 60) -> Dict[str, float]:
        """Get throughput metrics."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Filter recent metrics
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {
                "requests_per_second": 0,
                "tokens_per_second": 0
            }
        
        time_span = current_time - recent_metrics[0].timestamp
        if time_span == 0:
            time_span = 1
        
        total_requests = len(recent_metrics)
        total_tokens = sum(m.tokens_generated for m in recent_metrics)
        
        return {
            "requests_per_second": total_requests / time_span,
            "tokens_per_second": total_tokens / time_span
        }
    
    def get_model_breakdown(self) -> Dict[str, Any]:
        """Get performance breakdown by model type."""
        breakdown = {}
        
        for model_type in ['main', 'teacher', 'sentinel']:
            model_metrics = [m for m in self.metrics if m.model_type == model_type]
            
            if model_metrics:
                latencies = [m.latency_ms for m in model_metrics]
                breakdown[model_type] = {
                    "count": len(model_metrics),
                    "avg_latency_ms": np.mean(latencies),
                    "p95_latency_ms": np.percentile(latencies, 95),
                    "total_tokens": sum(m.tokens_generated for m in model_metrics)
                }
            else:
                breakdown[model_type] = {
                    "count": 0,
                    "avg_latency_ms": 0,
                    "p95_latency_ms": 0,
                    "total_tokens": 0
                }
        
        return breakdown
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get KV-cache hit rate statistics."""
        if not self.metrics:
            return {"hit_rate": 0, "total_requests": 0}
        
        cache_hits = sum(1 for m in self.metrics if m.kv_cache_hit)
        
        return {
            "hit_rate": cache_hits / len(self.metrics),
            "cache_hits": cache_hits,
            "cache_misses": len(self.metrics) - cache_hits,
            "total_requests": len(self.metrics)
        }
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU utilization statistics."""
        if not self.metrics:
            return {
                "avg_memory_mb": 0,
                "peak_memory_mb": 0,
                "avg_utilization": 0
            }
        
        memories = [m.gpu_memory_mb for m in self.metrics if m.gpu_memory_mb > 0]
        utilizations = [m.gpu_utilization for m in self.metrics if m.gpu_utilization > 0]
        
        return {
            "avg_memory_mb": np.mean(memories) if memories else 0,
            "peak_memory_mb": np.max(memories) if memories else 0,
            "avg_utilization": np.mean(utilizations) if utilizations else 0,
            "current_memory_mb": torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        }
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """Get MoE expert usage statistics."""
        expert_metrics = [m for m in self.metrics if m.experts_used > 0]
        
        if not expert_metrics:
            return {
                "avg_experts_used": 0,
                "total_expert_requests": 0
            }
        
        return {
            "avg_experts_used": np.mean([m.experts_used for m in expert_metrics]),
            "max_experts_used": max(m.experts_used for m in expert_metrics),
            "total_expert_requests": len(expert_metrics),
            "expert_usage_rate": len(expert_metrics) / len(self.metrics)
        }
    
    def _persist_metrics(self):
        """Save metrics to disk."""
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to serializable format
            data = {
                "timestamp": datetime.now().isoformat(),
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "uptime_seconds": time.time() - self.start_time,
                "percentiles": self.get_percentiles(),
                "throughput": self.get_throughput(),
                "model_breakdown": self.get_model_breakdown(),
                "cache_stats": self.get_cache_stats(),
                "gpu_stats": self.get_gpu_stats(),
                "expert_stats": self.get_expert_stats()
            }
            
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")
    
    def _load_historical(self):
        """Load historical metrics from disk."""
        if self.persist_path.exists():
            try:
                with open(self.persist_path) as f:
                    data = json.load(f)
                    # Could restore some aggregated stats here
                    logger.info(f"Loaded historical metrics from {self.persist_path}")
            except Exception as e:
                logger.error(f"Failed to load historical metrics: {e}")

class PerformanceDashboard:
    """
    Real-time performance dashboard.
    """
    
    def __init__(self):
        self.tracker = PerformanceTracker()
        self.alert_thresholds = {
            "p95_latency_ms": 1000,  # Alert if p95 > 1s
            "gpu_memory_mb": 15000,   # Alert if memory > 15GB
            "throughput_rps": 1       # Alert if RPS < 1
        }
    
    def get_dashboard(self) -> Dict[str, Any]:
        """Get complete dashboard data."""
        
        percentiles = self.tracker.get_percentiles()
        throughput = self.tracker.get_throughput()
        gpu_stats = self.tracker.get_gpu_stats()
        
        # Check for alerts
        alerts = []
        if percentiles.get('p95', 0) > self.alert_thresholds['p95_latency_ms']:
            alerts.append(f"⚠️ High p95 latency: {percentiles['p95']:.0f}ms")
        
        if gpu_stats['current_memory_mb'] > self.alert_thresholds['gpu_memory_mb']:
            alerts.append(f"⚠️ High GPU memory: {gpu_stats['current_memory_mb']:.0f}MB")
        
        if throughput['requests_per_second'] < self.alert_thresholds['throughput_rps']:
            alerts.append(f"⚠️ Low throughput: {throughput['requests_per_second']:.2f} RPS")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime": self._format_uptime(),
            "summary": {
                "total_requests": self.tracker.total_requests,
                "total_tokens": self.tracker.total_tokens,
                "avg_tokens_per_request": self.tracker.total_tokens / max(1, self.tracker.total_requests)
            },
            "latency": {
                **percentiles,
                "unit": "ms"
            },
            "throughput": {
                **throughput,
                "unit": "per_second"
            },
            "model_performance": self.tracker.get_model_breakdown(),
            "cache": self.tracker.get_cache_stats(),
            "gpu": gpu_stats,
            "experts": self.tracker.get_expert_stats(),
            "alerts": alerts,
            "healthy": len(alerts) == 0
        }
    
    def _format_uptime(self) -> str:
        """Format uptime as human-readable string."""
        uptime_seconds = time.time() - self.tracker.start_time
        return str(timedelta(seconds=int(uptime_seconds)))
    
    def print_dashboard(self):
        """Print dashboard to console."""
        dashboard = self.get_dashboard()
        
        print("\n" + "=" * 60)
        print("PERFORMANCE DASHBOARD")
        print("=" * 60)
        
        print(f"\nUptime: {dashboard['uptime']}")
        print(f"Total Requests: {dashboard['summary']['total_requests']:,}")
        print(f"Total Tokens: {dashboard['summary']['total_tokens']:,}")
        
        print("\n📊 Latency Percentiles (ms):")
        for key, value in dashboard['latency'].items():
            if key != 'unit':
                print(f"  {key}: {value:.1f}")
        
        print("\n🚀 Throughput:")
        print(f"  Requests/sec: {dashboard['throughput']['requests_per_second']:.2f}")
        print(f"  Tokens/sec: {dashboard['throughput']['tokens_per_second']:.1f}")
        
        print("\n🧠 Model Breakdown:")
        for model_type, stats in dashboard['model_performance'].items():
            if stats['count'] > 0:
                print(f"  {model_type}:")
                print(f"    Requests: {stats['count']}")
                print(f"    Avg latency: {stats['avg_latency_ms']:.1f}ms")
                print(f"    P95 latency: {stats['p95_latency_ms']:.1f}ms")
        
        print("\n💾 Cache Stats:")
        cache = dashboard['cache']
        print(f"  Hit Rate: {cache['hit_rate']:.1%}")
        print(f"  Hits/Misses: {cache['cache_hits']}/{cache['cache_misses']}")
        
        print("\n🎮 GPU Stats:")
        gpu = dashboard['gpu']
        print(f"  Current Memory: {gpu['current_memory_mb']:.0f}MB")
        print(f"  Peak Memory: {gpu['peak_memory_mb']:.0f}MB")
        
        if dashboard['alerts']:
            print("\n⚠️ ALERTS:")
            for alert in dashboard['alerts']:
                print(f"  {alert}")
        else:
            print("\n✅ System Healthy")

# Singleton instance
_dashboard = None

def get_dashboard() -> PerformanceDashboard:
    """Get or create performance dashboard."""
    global _dashboard
    if _dashboard is None:
        _dashboard = PerformanceDashboard()
    return _dashboard

if __name__ == "__main__":
    # Test dashboard
    dashboard = get_dashboard()
    tracker = dashboard.tracker
    
    # Simulate some requests
    import random
    
    print("Simulating performance metrics...")
    
    for i in range(100):
        # Simulate different model types
        model_type = random.choice(['main', 'teacher', 'sentinel'])
        
        # Simulate varying latencies
        if model_type == 'main':
            latency = random.uniform(100, 500)  # Main model slower
        elif model_type == 'teacher':
            latency = random.uniform(50, 200)   # Teacher faster (INT8)
        else:
            latency = random.uniform(30, 100)   # Sentinel fastest (INT8)
        
        tracker.record_request(
            request_id=f"req_{i:04d}",
            latency_ms=latency,
            tokens_generated=random.randint(10, 100),
            model_type=model_type,
            batch_size=random.randint(1, 8),
            kv_cache_hit=random.random() > 0.3,
            experts_used=random.randint(0, 4)
        )
    
    # Display dashboard
    dashboard.print_dashboard()
    
    # Save metrics
    tracker._persist_metrics()
    print(f"\nMetrics saved to {tracker.persist_path}")