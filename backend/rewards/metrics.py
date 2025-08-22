#!/usr/bin/env python3
"""
Reward System Metrics and Monitoring
Provides Prometheus metrics and health checks for the reward system.
"""

import time
import logging
from typing import Dict, Optional
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry

logger = logging.getLogger(__name__)

# Create a custom registry for reward metrics
registry = CollectorRegistry()

# Counters
rewards_inference_total = Counter(
    'rewards_inference_bly_total',
    'Total BLY distributed for inference',
    registry=registry
)

rewards_learning_total = Counter(
    'rewards_learning_bly_total',
    'Total BLY distributed for learning',
    registry=registry
)

rewards_validation_total = Counter(
    'rewards_validation_bly_total',
    'Total BLY distributed for validation',
    registry=registry
)

rewards_dataset_total = Counter(
    'rewards_dataset_bly_total',
    'Total BLY distributed for datasets',
    registry=registry
)

# Gauges
bucket_utilization = Gauge(
    'rewards_bucket_utilization',
    'Budget bucket utilization percentage',
    ['bucket_type'],
    registry=registry
)

bucket_available = Gauge(
    'rewards_bucket_available_bly',
    'Available BLY in budget bucket',
    ['bucket_type'],
    registry=registry
)

backpay_queue_size = Gauge(
    'rewards_backpay_queue_size',
    'Number of items in backpay queue',
    ['bucket_type'],
    registry=registry
)

backpay_queue_bly = Gauge(
    'rewards_backpay_queue_bly',
    'Total BLY amount in backpay queue',
    ['bucket_type'],
    registry=registry
)

learning_unpaid_claims = Gauge(
    'rewards_learning_unpaid_claims',
    'Number of unpaid learning reward claims',
    registry=registry
)

# Histograms
reward_distribution_time = Histogram(
    'rewards_distribution_duration_seconds',
    'Time taken to distribute rewards',
    ['distribution_type'],
    registry=registry
)

tokens_processed_histogram = Histogram(
    'rewards_tokens_processed',
    'Tokens processed per inference claim',
    buckets=(1000, 5000, 10000, 50000, 100000, 500000, 1000000),
    registry=registry
)

improvement_percentage_histogram = Histogram(
    'rewards_improvement_percentage',
    'Improvement percentage for learning claims',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
    registry=registry
)

# Summary
reward_amount_summary = Summary(
    'rewards_amount_bly',
    'Reward amounts distributed',
    ['reward_type'],
    registry=registry
)


class RewardMetricsCollector:
    """Collects and updates reward system metrics."""
    
    def __init__(self, allocator=None, distributor=None):
        """Initialize metrics collector."""
        self.allocator = allocator
        self.distributor = distributor
        self.last_update = 0
        self.update_interval = 10  # Update every 10 seconds
    
    def update_metrics(self):
        """Update all metrics from current system state."""
        current_time = time.time()
        
        # Rate limit updates
        if current_time - self.last_update < self.update_interval:
            return
        
        try:
            # Update bucket metrics if allocator available
            if self.allocator:
                self._update_bucket_metrics()
            
            # Update distribution metrics if distributor available
            if self.distributor:
                self._update_distribution_metrics()
            
            self.last_update = current_time
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def _update_bucket_metrics(self):
        """Update bucket-related metrics."""
        status = self.allocator.get_bucket_status()
        metrics = self.allocator.get_metrics()
        
        for bucket_type, info in status['buckets'].items():
            # Update utilization
            bucket_utilization.labels(bucket_type=bucket_type).set(
                info['utilization_24h']
            )
            
            # Update available budget
            bucket_available.labels(bucket_type=bucket_type).set(
                info['available']
            )
            
            # Update backpay queue
            backpay_queue_size.labels(bucket_type=bucket_type).set(
                info['backpay_queue_size']
            )
            
            backpay_queue_bly.labels(bucket_type=bucket_type).set(
                info['backpay_queue_bly']
            )
        
        # Update learning unpaid claims
        learning_queue = metrics['backpay_queue_size'].get('learning', 0)
        learning_unpaid_claims.set(learning_queue)
    
    def _update_distribution_metrics(self):
        """Update distribution-related metrics."""
        stats = self.distributor.get_distribution_stats()
        
        # Update totals by type
        for claim_type, type_stats in stats.get('by_type', {}).items():
            amount = type_stats.get('amount_bly', 0)
            
            if claim_type == 'inference':
                rewards_inference_total._value.set(amount)
            elif claim_type == 'learning':
                rewards_learning_total._value.set(amount)
            elif claim_type == 'validation':
                rewards_validation_total._value.set(amount)
            elif claim_type == 'dataset':
                rewards_dataset_total._value.set(amount)
    
    def record_inference_claim(self, tokens: int, amount_bly: float):
        """Record an inference reward claim."""
        tokens_processed_histogram.observe(tokens)
        reward_amount_summary.labels(reward_type='inference').observe(amount_bly)
        rewards_inference_total.inc(amount_bly)
    
    def record_learning_claim(self, improvement_pct: float, amount_bly: float):
        """Record a learning reward claim."""
        improvement_percentage_histogram.observe(improvement_pct)
        reward_amount_summary.labels(reward_type='learning').observe(amount_bly)
        rewards_learning_total.inc(amount_bly)
    
    def record_validation_claim(self, tasks: int, amount_bly: float):
        """Record a validation reward claim."""
        reward_amount_summary.labels(reward_type='validation').observe(amount_bly)
        rewards_validation_total.inc(amount_bly)
    
    def record_dataset_claim(self, gb_size: float, amount_bly: float):
        """Record a dataset reward claim."""
        reward_amount_summary.labels(reward_type='dataset').observe(amount_bly)
        rewards_dataset_total.inc(amount_bly)
    
    def record_distribution_time(self, distribution_type: str, duration: float):
        """Record time taken for distribution."""
        reward_distribution_time.labels(distribution_type=distribution_type).observe(duration)
    
    def get_health_status(self) -> Dict:
        """Get health status of reward system."""
        health = {
            'status': 'healthy',
            'issues': [],
            'metrics': {}
        }
        
        if self.allocator:
            metrics = self.allocator.get_metrics()
            
            # Check for alerts
            for alert in metrics.get('alerts', []):
                health['issues'].append(alert)
                if alert['type'] in ['low_utilization', 'high_backpay']:
                    health['status'] = 'degraded'
            
            # Add key metrics
            health['metrics']['bucket_health'] = metrics.get('budget_health', {})
            health['metrics']['total_backpay'] = sum(
                metrics.get('backpay_queue_bly', {}).values()
            )
        
        if self.distributor:
            stats = self.distributor.get_distribution_stats()
            health['metrics']['pending_claims'] = stats.get('pending_claims', 0)
            health['metrics']['processed_claims'] = stats.get('processed_claims', 0)
        
        # Check for critical issues
        if health['metrics'].get('total_backpay', 0) > 200000:
            health['status'] = 'critical'
            health['issues'].append({
                'type': 'critical_backpay',
                'message': 'Backpay queue critically high'
            })
        
        return health


def create_grafana_dashboard() -> Dict:
    """Create Grafana dashboard configuration for reward metrics."""
    return {
        "dashboard": {
            "title": "BLY Rewards System",
            "panels": [
                {
                    "title": "Budget Utilization",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rewards_bucket_utilization",
                            "legendFormat": "{{bucket_type}}"
                        }
                    ]
                },
                {
                    "title": "Rewards Distributed (24h)",
                    "type": "piechart",
                    "targets": [
                        {"expr": "increase(rewards_inference_bly_total[24h])", "legendFormat": "Inference"},
                        {"expr": "increase(rewards_learning_bly_total[24h])", "legendFormat": "Learning"},
                        {"expr": "increase(rewards_validation_bly_total[24h])", "legendFormat": "Validation"},
                        {"expr": "increase(rewards_dataset_bly_total[24h])", "legendFormat": "Dataset"}
                    ]
                },
                {
                    "title": "Backpay Queue",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rewards_backpay_queue_bly",
                            "legendFormat": "{{bucket_type}}"
                        }
                    ]
                },
                {
                    "title": "Learning Improvements Distribution",
                    "type": "heatmap",
                    "targets": [
                        {
                            "expr": "rewards_improvement_percentage",
                            "legendFormat": "Improvement %"
                        }
                    ]
                },
                {
                    "title": "Per 1k Tokens BLY Rate",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "rewards_inference_bly_total / (rewards_tokens_processed / 1000)",
                            "legendFormat": "BLY per 1k tokens"
                        }
                    ]
                },
                {
                    "title": "Effective BLY per 1% Improvement",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "rewards_learning_bly_total / rewards_improvement_percentage",
                            "legendFormat": "BLY per 1% improvement"
                        }
                    ]
                }
            ]
        }
    }


if __name__ == "__main__":
    # Example: Start metrics server
    from prometheus_client import start_http_server
    import time
    
    # Start Prometheus metrics server
    start_http_server(9090, registry=registry)
    print("Metrics server started on port 9090")
    
    # Create collector (would get actual instances in production)
    collector = RewardMetricsCollector()
    
    # Simulate some metrics
    collector.record_inference_claim(100000, 100.0)
    collector.record_learning_claim(2.5, 1250.0)
    
    # Keep server running
    while True:
        collector.update_metrics()
        time.sleep(10)