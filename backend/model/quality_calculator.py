"""
Quality Score Calculator - Production Implementation
Calculates real quality scores based on performance metrics
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class InferenceMetrics:
    """Metrics for a single inference request"""
    expert_name: str
    response_time: float  # seconds
    tokens_generated: int
    prompt_tokens: int
    throughput: float  # tokens/second
    memory_usage: float  # GB
    temperature: float = 0.7
    success: bool = True
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class QualityScoreCalculator:
    """
    Calculate quality scores based on multiple factors:
    - Response time / latency
    - Throughput (tokens/second)
    - Success rate
    - Model perplexity (if available)
    - Resource efficiency
    - Historical performance
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize quality calculator
        
        Args:
            history_size: Number of historical metrics to keep for averaging
        """
        self.history_size = history_size
        self.metrics_history: Dict[str, deque] = {}  # expert_name -> deque of metrics
        self.success_counts: Dict[str, Dict[str, int]] = {}  # expert_name -> {success: count, total: count}
        
        # Performance thresholds (can be configured)
        self.thresholds = {
            'excellent_throughput': 50.0,  # tokens/second
            'good_throughput': 30.0,
            'acceptable_throughput': 15.0,
            'excellent_latency': 0.5,  # seconds
            'good_latency': 1.0,
            'acceptable_latency': 2.0,
            'min_success_rate': 0.95,
            'target_success_rate': 0.99,
        }
    
    def calculate_score(
        self,
        expert_name: str,
        response_time: float,
        tokens_generated: int = 0,
        prompt_tokens: int = 0,
        memory_usage: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None,
        use_history: bool = True
    ) -> float:
        """
        Calculate quality score for an inference
        
        Args:
            expert_name: Name of the expert/model
            response_time: Total response time in seconds
            tokens_generated: Number of tokens generated
            prompt_tokens: Number of prompt tokens
            memory_usage: Memory used in GB
            success: Whether inference succeeded
            error_message: Error message if failed
            use_history: Whether to use historical data
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        # Calculate throughput
        throughput = tokens_generated / response_time if response_time > 0 else 0
        
        # Create metrics object
        metrics = InferenceMetrics(
            expert_name=expert_name,
            response_time=response_time,
            tokens_generated=tokens_generated,
            prompt_tokens=prompt_tokens,
            throughput=throughput,
            memory_usage=memory_usage,
            success=success,
            error_message=error_message
        )
        
        # Store in history
        self._update_history(metrics)
        
        # Calculate component scores
        scores = {}
        
        # 1. Throughput score (0.3 weight)
        scores['throughput'] = self._calculate_throughput_score(throughput)
        
        # 2. Latency score (0.3 weight)
        scores['latency'] = self._calculate_latency_score(response_time)
        
        # 3. Success rate score (0.2 weight)
        if use_history:
            scores['success_rate'] = self._calculate_success_rate_score(expert_name)
        else:
            scores['success_rate'] = 1.0 if success else 0.0
        
        # 4. Efficiency score (0.1 weight)
        scores['efficiency'] = self._calculate_efficiency_score(
            tokens_generated, memory_usage, response_time
        )
        
        # 5. Consistency score (0.1 weight)
        if use_history:
            scores['consistency'] = self._calculate_consistency_score(expert_name)
        else:
            scores['consistency'] = 0.8  # Default for new experts
        
        # Apply failure penalty
        if not success:
            penalty = 0.5  # 50% penalty for failures
            for key in scores:
                scores[key] *= penalty
        
        # Calculate weighted average
        weights = {
            'throughput': 0.3,
            'latency': 0.3,
            'success_rate': 0.2,
            'efficiency': 0.1,
            'consistency': 0.1
        }
        
        quality_score = sum(scores[k] * weights[k] for k in weights)
        
        # Log detailed scores for debugging
        logger.debug(f"Quality scores for {expert_name}: {scores}")
        logger.debug(f"Final quality score: {quality_score:.3f}")
        
        return min(1.0, max(0.0, quality_score))
    
    def _calculate_throughput_score(self, throughput: float) -> float:
        """Calculate score based on throughput"""
        if throughput >= self.thresholds['excellent_throughput']:
            return 1.0
        elif throughput >= self.thresholds['good_throughput']:
            return 0.85 + 0.15 * (throughput - self.thresholds['good_throughput']) / (
                self.thresholds['excellent_throughput'] - self.thresholds['good_throughput']
            )
        elif throughput >= self.thresholds['acceptable_throughput']:
            return 0.6 + 0.25 * (throughput - self.thresholds['acceptable_throughput']) / (
                self.thresholds['good_throughput'] - self.thresholds['acceptable_throughput']
            )
        else:
            # Below acceptable
            return max(0.1, 0.6 * throughput / self.thresholds['acceptable_throughput'])
    
    def _calculate_latency_score(self, latency: float) -> float:
        """Calculate score based on latency (lower is better)"""
        if latency <= self.thresholds['excellent_latency']:
            return 1.0
        elif latency <= self.thresholds['good_latency']:
            # Linear interpolation between excellent and good
            return 0.85 + 0.15 * (self.thresholds['good_latency'] - latency) / (
                self.thresholds['good_latency'] - self.thresholds['excellent_latency']
            )
        elif latency <= self.thresholds['acceptable_latency']:
            return 0.6 + 0.25 * (self.thresholds['acceptable_latency'] - latency) / (
                self.thresholds['acceptable_latency'] - self.thresholds['good_latency']
            )
        else:
            # Exponential decay for high latency
            return max(0.1, 0.6 * np.exp(-(latency - self.thresholds['acceptable_latency']) / 2))
    
    def _calculate_success_rate_score(self, expert_name: str) -> float:
        """Calculate score based on historical success rate"""
        if expert_name not in self.success_counts:
            return 0.95  # Default for new experts
        
        counts = self.success_counts[expert_name]
        if counts['total'] == 0:
            return 0.95
        
        success_rate = counts['success'] / counts['total']
        
        if success_rate >= self.thresholds['target_success_rate']:
            return 1.0
        elif success_rate >= self.thresholds['min_success_rate']:
            return 0.8 + 0.2 * (success_rate - self.thresholds['min_success_rate']) / (
                self.thresholds['target_success_rate'] - self.thresholds['min_success_rate']
            )
        else:
            return max(0.0, success_rate)
    
    def _calculate_efficiency_score(
        self,
        tokens_generated: int,
        memory_usage: float,
        response_time: float
    ) -> float:
        """Calculate efficiency score (tokens per GB-second)"""
        if memory_usage <= 0 or response_time <= 0:
            return 0.8  # Default
        
        # Efficiency = tokens / (memory * time)
        efficiency = tokens_generated / (memory_usage * response_time)
        
        # Normalize (assuming 100 tokens/GB-sec is excellent)
        target_efficiency = 100.0
        score = min(1.0, efficiency / target_efficiency)
        
        return score
    
    def _calculate_consistency_score(self, expert_name: str) -> float:
        """Calculate consistency based on variance in historical performance"""
        if expert_name not in self.metrics_history:
            return 0.8  # Default
        
        history = self.metrics_history[expert_name]
        if len(history) < 3:
            return 0.8  # Not enough data
        
        # Calculate coefficient of variation for throughput
        throughputs = [m.throughput for m in history if m.success]
        if len(throughputs) < 2:
            return 0.5
        
        mean_throughput = np.mean(throughputs)
        std_throughput = np.std(throughputs)
        
        if mean_throughput <= 0:
            return 0.5
        
        # Coefficient of variation (lower is better)
        cv = std_throughput / mean_throughput
        
        # Convert to score (0 CV = 1.0 score, 0.5 CV = 0.5 score)
        consistency_score = max(0.0, min(1.0, 1.0 - cv))
        
        return consistency_score
    
    def _update_history(self, metrics: InferenceMetrics):
        """Update metrics history"""
        expert_name = metrics.expert_name
        
        # Initialize if needed
        if expert_name not in self.metrics_history:
            self.metrics_history[expert_name] = deque(maxlen=self.history_size)
            self.success_counts[expert_name] = {'success': 0, 'total': 0}
        
        # Add to history
        self.metrics_history[expert_name].append(metrics)
        
        # Update success counts
        self.success_counts[expert_name]['total'] += 1
        if metrics.success:
            self.success_counts[expert_name]['success'] += 1
    
    def get_expert_stats(self, expert_name: str) -> Dict[str, Any]:
        """Get statistics for an expert"""
        if expert_name not in self.metrics_history:
            return {
                'expert_name': expert_name,
                'total_requests': 0,
                'success_rate': 0.0,
                'avg_throughput': 0.0,
                'avg_latency': 0.0,
                'quality_score': 0.5
            }
        
        history = list(self.metrics_history[expert_name])
        counts = self.success_counts[expert_name]
        
        successful = [m for m in history if m.success]
        
        stats = {
            'expert_name': expert_name,
            'total_requests': counts['total'],
            'success_rate': counts['success'] / counts['total'] if counts['total'] > 0 else 0.0,
            'avg_throughput': np.mean([m.throughput for m in successful]) if successful else 0.0,
            'avg_latency': np.mean([m.response_time for m in successful]) if successful else 0.0,
            'p95_latency': np.percentile([m.response_time for m in successful], 95) if successful else 0.0,
            'quality_score': self.calculate_score(
                expert_name=expert_name,
                response_time=np.mean([m.response_time for m in successful]) if successful else 10.0,
                tokens_generated=int(np.mean([m.tokens_generated for m in successful])) if successful else 0,
                success=len(successful) > 0,
                use_history=True
            )
        }
        
        return stats
    
    def reset_expert_stats(self, expert_name: str):
        """Reset statistics for an expert"""
        if expert_name in self.metrics_history:
            del self.metrics_history[expert_name]
        if expert_name in self.success_counts:
            del self.success_counts[expert_name]


# Singleton instance
_quality_calculator = None

def get_quality_calculator() -> QualityScoreCalculator:
    """Get singleton quality calculator instance"""
    global _quality_calculator
    if _quality_calculator is None:
        _quality_calculator = QualityScoreCalculator()
    return _quality_calculator