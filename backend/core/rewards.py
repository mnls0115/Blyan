#!/usr/bin/env python3
"""Fair reward system with expected value equalization across all nodes."""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import json


@dataclass
class RewardComponent:
    """Individual components of reward calculation."""
    base_price: float           # P_i: Request unit price
    tokens_generated: int       # Work done (tokens or FLOPs)
    quality_score: float        # Q: PoL improvement/accuracy (0.8-1.2)
    speed_multiplier: float     # S: Response time bonus/penalty (0.5-1.5)
    integrity_factor: float     # I: Security verification pass (0.0 or 1.0)
    cost_normalizer: float      # 1/C: Inverse of estimated resource cost


@dataclass
class NodePerformanceMetrics:
    """Performance tracking for fair reward distribution."""
    node_id: str
    total_rewards: float = 0.0
    total_gpu_seconds: float = 0.0
    request_count: int = 0
    
    # Revenue rate tracking
    revenue_per_gpu_second: float = 0.0
    recent_revenue_samples: deque = None
    
    # Preemption compensation
    preemption_credit: float = 0.0
    preemption_count: int = 0
    total_preemption_seconds: float = 0.0
    
    # Routing bias for fairness
    routing_bias: float = 0.0  # -1.0 to +1.0, adjusted by feedback loop
    
    # Trust and reliability
    trust_score: float = 1.0
    slo_violation_count: int = 0
    integrity_pass_rate: float = 1.0
    
    def __post_init__(self):
        if self.recent_revenue_samples is None:
            self.recent_revenue_samples = deque(maxlen=50)


class FairRewardCalculator:
    """Calculates rewards with expected value equalization."""
    
    def __init__(self, 
                 target_slo_ms: float = 300,
                 preemption_credit_rate: float = 0.1,  # Credit per second of preemption
                 rebalance_interval: int = 3600):      # Rebalance every hour
        
        self.target_slo_ms = target_slo_ms
        self.preemption_credit_rate = preemption_credit_rate
        self.rebalance_interval = rebalance_interval
        
        # Node performance tracking
        self.node_metrics: Dict[str, NodePerformanceMetrics] = {}
        self.last_rebalance_time = time.time()
        
        # Market dynamics
        self.base_request_price = 1.0
        self.surge_multiplier = 1.0
        
        # Fairness parameters
        self.exploration_epsilon = 0.05  # 5% random allocation for cold nodes
        self.max_bias_adjustment = 0.1   # Max 10% bias change per rebalance
        
    def compute_reward(self, 
                      node_id: str,
                      request_id: str,
                      tokens_generated: int,
                      quality_score: float,
                      actual_latency_ms: float,
                      integrity_passed: bool,
                      estimated_cost: float,
                      request_price: float = None) -> Tuple[float, RewardComponent]:
        """Compute fair reward with all normalization factors."""
        
        if request_price is None:
            request_price = self.base_request_price * self.surge_multiplier
            
        # Calculate reward components
        components = RewardComponent(
            base_price=request_price,
            tokens_generated=tokens_generated,
            quality_score=max(0.8, min(1.2, quality_score)),  # Clamp quality
            speed_multiplier=self._calculate_speed_multiplier(actual_latency_ms),
            integrity_factor=1.0 if integrity_passed else 0.0,
            cost_normalizer=1.0 / max(estimated_cost, 1e-6)
        )
        
        # Calculate final reward
        reward = (components.base_price * 
                 components.tokens_generated * 
                 components.quality_score * 
                 components.speed_multiplier * 
                 components.integrity_factor * 
                 components.cost_normalizer)
        
        # Apply preemption credit bonus
        if node_id in self.node_metrics:
            credit_bonus = min(self.node_metrics[node_id].preemption_credit * 0.1, reward * 0.2)
            reward += credit_bonus
            self.node_metrics[node_id].preemption_credit = max(0, 
                self.node_metrics[node_id].preemption_credit - credit_bonus / 0.1)
        
        # Record performance metrics
        self._record_performance(node_id, reward, estimated_cost, actual_latency_ms, integrity_passed)
        
        return reward, components
    
    def _calculate_speed_multiplier(self, actual_latency_ms: float) -> float:
        """Calculate speed bonus/penalty based on SLO performance."""
        if actual_latency_ms <= 0:
            return 1.0
            
        ratio = self.target_slo_ms / actual_latency_ms
        return max(0.5, min(1.5, ratio))  # 50% penalty to 50% bonus
    
    def _record_performance(self, 
                           node_id: str, 
                           reward: float, 
                           cost: float, 
                           latency_ms: float,
                           integrity_passed: bool):
        """Record node performance for fairness tracking."""
        
        if node_id not in self.node_metrics:
            self.node_metrics[node_id] = NodePerformanceMetrics(node_id=node_id)
            
        metrics = self.node_metrics[node_id]
        
        # Update totals
        metrics.total_rewards += reward
        metrics.total_gpu_seconds += cost  # Assuming cost approximates GPU-seconds
        metrics.request_count += 1
        
        # Update revenue rate
        if metrics.total_gpu_seconds > 0:
            metrics.revenue_per_gpu_second = metrics.total_rewards / metrics.total_gpu_seconds
            metrics.recent_revenue_samples.append(metrics.revenue_per_gpu_second)
        
        # Update trust metrics
        if latency_ms > self.target_slo_ms * 1.5:  # 50% over SLO = violation
            metrics.slo_violation_count += 1
            
        # Update integrity rate (EMA)
        alpha = 0.1
        metrics.integrity_pass_rate = (alpha * (1.0 if integrity_passed else 0.0) + 
                                      (1 - alpha) * metrics.integrity_pass_rate)
        
        # Update trust score based on reliability
        violation_rate = metrics.slo_violation_count / max(metrics.request_count, 1)
        metrics.trust_score = max(0.1, 1.0 - violation_rate * 0.5) * metrics.integrity_pass_rate
    
    def record_preemption(self, node_id: str, preemption_seconds: float):
        """Record learning preemption for compensation."""
        if node_id not in self.node_metrics:
            self.node_metrics[node_id] = NodePerformanceMetrics(node_id=node_id)
            
        metrics = self.node_metrics[node_id]
        credit = preemption_seconds * self.preemption_credit_rate
        
        metrics.preemption_credit += credit
        metrics.preemption_count += 1
        metrics.total_preemption_seconds += preemption_seconds
        
        print(f"💳 Node {node_id} earned {credit:.3f} preemption credit ({preemption_seconds:.1f}s)")
    
    def get_routing_probabilities(self, 
                                 candidate_nodes: List[str],
                                 context: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate fair routing probabilities with expected value equalization."""
        
        if not candidate_nodes:
            return {}
            
        # Rebalance if needed
        current_time = time.time()
        if current_time - self.last_rebalance_time > self.rebalance_interval:
            self._rebalance_routing_weights()
            self.last_rebalance_time = current_time
        
        # Calculate base scores for each node
        scores = {}
        for node_id in candidate_nodes:
            base_score = self._calculate_base_routing_score(node_id, context)
            fairness_adjustment = self._get_fairness_adjustment(node_id)
            scores[node_id] = base_score * (1.0 + fairness_adjustment)
        
        # Convert to probabilities with softmax
        k = 2.0  # Temperature parameter
        exp_scores = {node: np.exp(k * score) for node, score in scores.items()}
        total_exp = sum(exp_scores.values())
        
        probabilities = {node: exp_score / total_exp for node, exp_score in exp_scores.items()}
        
        # Apply exploration (random allocation for fairness)
        exploration_prob = self.exploration_epsilon / len(candidate_nodes)
        for node_id in candidate_nodes:
            probabilities[node_id] = ((1 - self.exploration_epsilon) * probabilities[node_id] + 
                                     exploration_prob)
        
        return probabilities
    
    def _calculate_base_routing_score(self, node_id: str, context: Dict[str, Any] = None) -> float:
        """Calculate base routing score (latency, warm hits, trust)."""
        if node_id not in self.node_metrics:
            return 0.5  # Default score for new nodes
            
        metrics = self.node_metrics[node_id]
        
        # Base factors (these would come from actual node status)
        latency_score = context.get('latency_score', 0.8) if context else 0.8
        warm_hit_score = context.get('warm_hit_score', 0.7) if context else 0.7
        cost_efficiency = context.get('cost_efficiency', 1.0) if context else 1.0
        
        # Combine with trust and reliability
        score = (0.3 * latency_score + 
                0.2 * warm_hit_score + 
                0.3 * metrics.trust_score + 
                0.2 * cost_efficiency)
        
        return max(0.1, min(1.0, score))
    
    def _get_fairness_adjustment(self, node_id: str) -> float:
        """Get fairness adjustment based on revenue rate deviation."""
        if node_id not in self.node_metrics:
            return 0.0
            
        return self.node_metrics[node_id].routing_bias
    
    def _rebalance_routing_weights(self):
        """Rebalance routing weights to equalize expected revenue per GPU-second."""
        
        if len(self.node_metrics) < 2:
            return
            
        # Calculate current revenue rates
        revenue_rates = {}
        for node_id, metrics in self.node_metrics.items():
            if metrics.total_gpu_seconds > 0:
                revenue_rates[node_id] = metrics.revenue_per_gpu_second
            else:
                revenue_rates[node_id] = 0.0
        
        if not revenue_rates:
            return
            
        # Calculate average revenue rate
        avg_revenue_rate = np.mean(list(revenue_rates.values()))
        
        if avg_revenue_rate <= 0:
            return
            
        print(f"🔄 Rebalancing routing weights. Average revenue rate: {avg_revenue_rate:.4f}")
        
        # Adjust routing bias for each node
        for node_id, current_rate in revenue_rates.items():
            deviation = (avg_revenue_rate - current_rate) / avg_revenue_rate
            
            # Clamp adjustment to prevent oscillation
            adjustment = max(-self.max_bias_adjustment, 
                           min(self.max_bias_adjustment, deviation))
            
            old_bias = self.node_metrics[node_id].routing_bias
            self.node_metrics[node_id].routing_bias += adjustment
            
            # Clamp final bias
            self.node_metrics[node_id].routing_bias = max(-1.0, 
                min(1.0, self.node_metrics[node_id].routing_bias))
            
            print(f"  Node {node_id}: rate={current_rate:.4f}, "
                  f"bias {old_bias:.3f} → {self.node_metrics[node_id].routing_bias:.3f}")
    
    def get_fairness_report(self) -> Dict[str, Any]:
        """Generate fairness report for monitoring."""
        if not self.node_metrics:
            return {"status": "no_data"}
            
        # Calculate revenue rate statistics
        revenue_rates = []
        node_summaries = []
        
        for node_id, metrics in self.node_metrics.items():
            rate = metrics.revenue_per_gpu_second
            revenue_rates.append(rate)
            
            node_summaries.append({
                "node_id": node_id,
                "revenue_per_gpu_sec": rate,
                "total_rewards": metrics.total_rewards,
                "request_count": metrics.request_count,
                "preemption_credit": metrics.preemption_credit,
                "routing_bias": metrics.routing_bias,
                "trust_score": metrics.trust_score,
                "slo_violations": metrics.slo_violation_count
            })
        
        if revenue_rates:
            revenue_std = np.std(revenue_rates)
            revenue_mean = np.mean(revenue_rates)
            coefficient_of_variation = revenue_std / revenue_mean if revenue_mean > 0 else float('inf')
        else:
            revenue_std = 0
            revenue_mean = 0
            coefficient_of_variation = 0
            
        return {
            "fairness_metrics": {
                "revenue_rate_std": revenue_std,
                "revenue_rate_mean": revenue_mean,
                "coefficient_of_variation": coefficient_of_variation,
                "fairness_score": max(0, 1.0 - coefficient_of_variation)  # Lower CV = more fair
            },
            "node_summaries": sorted(node_summaries, key=lambda x: x["revenue_per_gpu_sec"], reverse=True),
            "last_rebalance": self.last_rebalance_time,
            "total_nodes": len(self.node_metrics)
        }
    
    def update_surge_pricing(self, demand_multiplier: float):
        """Update surge pricing based on demand prediction."""
        self.surge_multiplier = max(0.5, min(3.0, demand_multiplier))
        print(f"💰 Surge pricing updated: {self.surge_multiplier:.2f}x")


# Global reward calculator instance
_global_reward_calculator: Optional[FairRewardCalculator] = None

def get_reward_calculator() -> FairRewardCalculator:
    """Get global reward calculator instance."""
    global _global_reward_calculator
    if _global_reward_calculator is None:
        _global_reward_calculator = FairRewardCalculator()
    return _global_reward_calculator


# Example usage and testing
if __name__ == "__main__":
    calculator = FairRewardCalculator()
    
    # Simulate different nodes with varying performance
    nodes = ["node1", "node2", "node3"]
    
    print("=== Simulating node performance over time ===")
    
    # Simulate 100 requests
    for i in range(100):
        # Random request parameters
        node_id = np.random.choice(nodes)
        tokens = np.random.randint(50, 200)
        quality = np.random.normal(1.0, 0.1)
        
        # Simulate different node characteristics
        if node_id == "node1":  # Fast, reliable node
            latency = np.random.normal(200, 30)
            cost = np.random.normal(2.0, 0.3)
            integrity = np.random.random() > 0.05
        elif node_id == "node2":  # Slower but efficient node
            latency = np.random.normal(350, 50)
            cost = np.random.normal(1.5, 0.2)
            integrity = np.random.random() > 0.02
        else:  # node3 - variable performance
            latency = np.random.normal(300, 100)
            cost = np.random.normal(2.5, 0.5)
            integrity = np.random.random() > 0.1
            
        reward, components = calculator.compute_reward(
            node_id=node_id,
            request_id=f"req_{i}",
            tokens_generated=tokens,
            quality_score=quality,
            actual_latency_ms=latency,
            integrity_passed=integrity,
            estimated_cost=cost
        )
        
        # Simulate some preemptions for learning nodes
        if i % 20 == 0 and np.random.random() > 0.7:
            preemption_time = np.random.uniform(30, 120)
            calculator.record_preemption(node_id, preemption_time)
    
    # Generate fairness report
    print("\n=== Fairness Report ===")
    report = calculator.get_fairness_report()
    
    print(f"Fairness score: {report['fairness_metrics']['fairness_score']:.3f}")
    print(f"Revenue coefficient of variation: {report['fairness_metrics']['coefficient_of_variation']:.3f}")
    print("\nNode performance:")
    
    for node in report['node_summaries']:
        print(f"  {node['node_id']}: "
              f"{node['revenue_per_gpu_sec']:.4f} rewards/gpu-sec, "
              f"bias: {node['routing_bias']:+.3f}, "
              f"trust: {node['trust_score']:.2f}")
    
    # Test routing probabilities
    print("\n=== Routing Probabilities ===")
    probs = calculator.get_routing_probabilities(nodes)
    for node_id, prob in probs.items():
        print(f"  {node_id}: {prob:.3f}")